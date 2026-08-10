#!/usr/bin/env python3
# Copyright    2026  Xiaomi Corp.        (authors:  Han Zhu)
#
# See ../../LICENSE for clarification regarding multiple authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Training loop for OmniVoice.

Wraps the HuggingFace Accelerate training loop with checkpoint saving/resuming,
evaluation, gradient accumulation, and learning rate scheduling.
Launched via ``omnivoice.cli.train``.

中文说明:
    OmniVoice 的训练循环。基于 HuggingFace Accelerate 封装, 在其之上补充了:
      - checkpoint 保存与恢复 (断点续训);
      - 定期评估 (evaluation);
      - 梯度累积 (gradient accumulation, 用小显存模拟大 batch);
      - 学习率调度 (learning rate scheduling, 含 warmup)。
    由 ``omnivoice.cli.train`` 启动: cli 负责解析配置、构建模型和 dataloader,
    本文件的 OmniTrainer 负责实际的"取 batch -> forward -> backward -> 更新权重"循环。
"""

import logging
import math
import os
import sys
import time
from datetime import timedelta
from typing import Any, Optional

import torch
# accelerate 让同一份训练代码能在不同硬件配置上跑
from accelerate import Accelerator, DistributedDataParallelKwargs
from accelerate.utils import DeepSpeedPlugin, InitProcessGroupKwargs, set_seed
from torch.utils.data import DataLoader
from transformers import (
    get_cosine_schedule_with_warmup,
    get_constant_schedule_with_warmup,
)

from omnivoice.training.checkpoint import TrainLogger, load_checkpoint
from omnivoice.training.checkpoint import save_checkpoint as engine_save_checkpoint

logger = logging.getLogger(__name__)


def _to_device(batch, device):
    """Move all tensors in a batch dict to the target device.
    把 batch 字典里的所有 Tensor 搬到模型所在设备，非 Tensor 数据保持不变
    
    batch 是一个 Python dict，保存一批训练/验证数据。数据示例：
    {
        "input_ids": Tensor(...),
        "audio_mask": Tensor(...),
        "labels": Tensor(...),
        "attention_mask": Tensor(...),
    }
    """
    return {
        k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v
        for k, v in batch.items()
    }


class OmniTrainer:
    def __init__(
        self,
        model: torch.nn.Module,
        config: Any,  # TrainingConfig
        train_dataloader: DataLoader,
        eval_dataloader: Optional[DataLoader] = None,
        tokenizer: Optional[Any] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        lr_scheduler: Optional[Any] = None,
    ):
        """
        Args:
            model: 要训练的模型，核心方法有：
                -  model(**batch)：实际通过 __call__() 调用 OmniVoice.forward()
                - .parameters()：获取模型参数：parameter.data 获取模型权重；parameter.grad获取梯度
                - .train()：将模型设置为训练模式
                - .eval()：将模型设置为评估模式
            config: 训练配置

            【注意】数据集加载
            train_dataloader: 训练数据集加载器
            eval_dataloader: 模型验证集的 DataLoader
            tokenizer: 分词器
            optimizer: 优化器，
                - .step() 根据梯度更新模型参数
                - .zero_grad() 清空梯度
            lr_scheduler: 学习率调度器
        """

        # 训练配置
        self.config = config
        # 要训练的模型？
        self.model = model
        # 分词器
        self.tokenizer = tokenizer
        # 训练数据集加载器
        self.train_dataloader = train_dataloader
        # 模型验证集的 DataLoader
        self.eval_dataloader = eval_dataloader

        # 1. Initialize Accelerator 协助分布式训练
        self.accelerator = self._init_accelerator()

        # 2. Setup Optimizer & Scheduler if not provided
        #   设置训练时候的核心组件: 优化器和学习率调度器
        if optimizer is None:
            # 优化器 通过 self.model.parameters()获取并绑定模型参数
            self.optimizer, self.lr_scheduler = self.create_optimizer_and_scheduler()
        else:
            self.optimizer = optimizer
            self.lr_scheduler = lr_scheduler

        # 3. DeepSpeed Hack (Batch Size fix)
        if self.accelerator.distributed_type == "DEEPSPEED":
            self.accelerator.state.deepspeed_plugin.deepspeed_config[
                "train_micro_batch_size_per_gpu"
            ] = 1

        # 4. Prepare with Accelerator
        # 元组解包：把 prepare() 返回的 3 个对象依次赋给左侧 3 个变量，数量必须严格匹配。
        (self.model, self.optimizer, self.lr_scheduler,) = self.accelerator.prepare(
            self.model,
            self.optimizer, # 绑定了 model.Parameters() 的优化器
            self.lr_scheduler, # 绑定了 optimizer
        )

        self.global_step = 0
        self.epoch = 0

    def _init_accelerator(self) -> Accelerator:
        """Initialize Accelerator, DeepSpeed, and Logging.
            Accelerator: 加速器，用于分布式训练
            DeepSpeed: 深度学习加速库，用于分布式训练
        """
        # TF32 setup
        if getattr(self.config, "allow_tf32", False):
            # matmul 是 matrix multiplication 的缩写
            # set_float32_matmul_precision 是指执行 float32 矩阵乘法时，GPU 内部计算使用多高的精度
            torch.set_float32_matmul_precision("high")

        # Init handlers Accelerate 的分布式训练配置
        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=False) # 默认所有模型参数都会参与本次训练
        init_kwargs = InitProcessGroupKwargs(timeout=timedelta(minutes=60)) # 设置分布式进程通信超时时间为 60 分钟

        # DeepSpeed setup
        deepspeed_plugin = None
        if self.config.use_deepspeed and self.config.deepspeed_config:
            if not os.path.exists(self.config.deepspeed_config):
                raise FileNotFoundError(
                    f"DeepSpeed config not found: {self.config.deepspeed_config}"
                )
            deepspeed_plugin = DeepSpeedPlugin(
                hf_ds_config=self.config.deepspeed_config,
                gradient_accumulation_steps=self.config.gradient_accumulation_steps,
                gradient_clipping=self.config.max_grad_norm,
            )

        accelerator = Accelerator(
            # 累积多少个 batch 的梯度后更新一次权重
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            # 混合精度类型，例如 bf16、fp16 或 no
            mixed_precision=self.config.mixed_precision,
            # 使用 TensorBoard 记录训练指标
            log_with="tensorboard",
            # TensorBoard 日志等输出文件的保存目录
            project_dir=self.config.output_dir,
            # 不由 Accelerate 自动推进调度器，训练循环中手动调用 lr_scheduler.step()
            step_scheduler_with_optimizer=False,
            # 传入 DDP 和分布式进程通信配置
            kwargs_handlers=[ddp_kwargs, init_kwargs],
            # DeepSpeed 配置，未启用时为 None
            deepspeed_plugin=deepspeed_plugin,
            # 不让 Accelerate 自动把一个 batch 再拆分给多张 GPU
            split_batches=False,
        )

        # Logging setup
        if accelerator.is_main_process:
            os.makedirs(self.config.output_dir, exist_ok=True)
            # Try to save config if it has the method
            if hasattr(self.config, "save_to_json"):
                self.config.save_to_json(
                    os.path.join(self.config.output_dir, "initial_config.json")
                )

            logging.basicConfig(
                format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
                datefmt="%m/%d/%Y %H:%M:%S",
                level=logging.INFO,
                handlers=[
                    logging.StreamHandler(sys.stdout),
                    logging.FileHandler(
                        os.path.join(self.config.output_dir, "train.log")
                    ),
                ],
            )
        else:
            logging.basicConfig(level=logging.ERROR)

        logger.info(f"Loaded Config: {self.config}")
        set_seed(self.config.seed)
        accelerator.init_trackers("tensorboard")
        return accelerator

    def create_optimizer_and_scheduler(self):
        """Default AdamW + configurable LR Scheduler.
            重要概念:
                AdamW: 一种优化器 Optimizer，用于训练模型
                Learning Rate Scheduler: 一种调度器，用于调整学习率
        """
        optimizer = torch.optim.AdamW(
            self.model.parameters(), # 权重，【重要】获取模型参数，优化器在这里跟模型参数绑定
            # AdamW 的基础/峰值学习率；训练中 lr_scheduler 会动态调整实际学习率
            # SGD 的基础理解：新权重 = 旧权重 - 学习率 × 梯度。SGD 随机梯度下降（Stochastic Gradient Descent）
            lr=self.config.learning_rate, 
            weight_decay=self.config.weight_decay, # todo
        )

        if self.config.warmup_type == "ratio":
            final_warmup_steps = math.ceil(self.config.steps * self.config.warmup_ratio)
        else:
            final_warmup_steps = self.config.warmup_steps

        if self.config.lr_scheduler_type == "constant":
            lr_scheduler = get_constant_schedule_with_warmup(
                optimizer=optimizer,
                num_warmup_steps=final_warmup_steps,
            )
        else:
            lr_scheduler = get_cosine_schedule_with_warmup(
                optimizer=optimizer,
                num_warmup_steps=final_warmup_steps,
                num_training_steps=self.config.steps,
            )
        return optimizer, lr_scheduler # TODO lr_scheduler 是啥？

    def save_checkpoint(self, step):
        """Wrapper for engine save_checkpoint."""
        # 保存 checkpoint
        engine_save_checkpoint(
            self.accelerator,
            self.model,
            self.tokenizer,
            self.config.output_dir,
            step,
            self.config.keep_last_n_checkpoints,
        )
        # Save config copy for convenience 保存配置文件
        if self.accelerator.is_main_process and hasattr(self.config, "save_to_json"):
            checkpoint_dir = os.path.join(self.config.output_dir, f"checkpoint-{step}")
            self.config.save_to_json(os.path.join(checkpoint_dir, "train_config.json"))

    def load_checkpoint(self, checkpoint_path):
        """Wrapper for loading."""
        step = load_checkpoint(self.accelerator, checkpoint_path)
        self.global_step = step
        logger.info(f"Resumed from step {self.global_step}")
        return step

    def evaluate(self):
        """Evaluation loop. 【注意】这个方法只做评估和记录，不影响训练进程
        """
        # 如果没有验证集则直接返回
        if self.eval_dataloader is None:
            return {}

        # 设置模型为评估模式。eval和推理的关系：“验证”和“推理”是eval模式下的两种使用场景，不是两种独立的模型模式。
        self.model.eval()
        logger.info(f"Running evaluation at step {self.global_step}...")

        # 初始化空值、累加验证集的 loss
        local_loss_sum = torch.tensor(0.0, device=self.accelerator.device)
        eval_count = 0

        # no_grad 表示不自动求梯度，不会记录和计算梯度，减少显存占用、提升推理速度
        with torch.no_grad():
            for eval_batch in self.eval_dataloader: # 遍历验证集的 DataLoader
                eval_batch = _to_device(eval_batch, self.accelerator.device)
                # 调用链路：self.model(**eval_batch) -> nn.Module.__call__() -> OmniVoice.forward()
                # 其中， model 是一个可调用的 nn.Module 对象、所以会调用到 nn.Module.__call__() 方法
                # 结果类型是 OmniVoiceModelOutput
                outputs = self.model(**eval_batch)
                # 获取 loss 并累加（另外，除了 loss，还有 logits 是模型的输出
                #【重要】loss 是一个标量 tensor，比如 tensor(2.35)
                # detach() 返回结果的 形状、值、类型都和 loss 相同，修改了计算图和梯度相关的信息
                local_loss_sum += outputs.loss.detach() 
                eval_count += 1 # 累加验证集的样本数量

        if eval_count > 0:
            local_mean = local_loss_sum / eval_count # 计算平均损失；张量除法、每个元素都和 eval_count 相除
        else:
            local_mean = torch.tensor(0.0, device=self.accelerator.device) # 防御代码，忽略

        # 每个gpu都有自己的 mean，这里是聚合所有gpu的mean：结果是一个 tensor，形状是 (num_gpus, )【注意】
        all_means = self.accelerator.gather(local_mean) 
        # 计算所有gpu的mean的平均值为最终的 loss
        # .mean() 是 tensor 的 方法，计算所有元素的平均值，结果是张量标量
        # .item() 是 tensor 的 方法，将张量标量转换成 python 数值， .item() 只能用只有一个元素的 tensor
        final_eval_loss = all_means.mean().item()

        # 记录评估指标
        eval_metrics = {"eval/loss": final_eval_loss}
        self.accelerator.log(eval_metrics, step=self.global_step)
        logger.info(f"Eval Loss: {final_eval_loss:.4f}")

        self.accelerator.wait_for_everyone() # 等待所有gpu都完成评估
        self.model.train() # 切换回训练模式
        return eval_metrics

    def train(self):
        """Main training loop.
        【重要】核心训练代码

        基本流程:
          step 1: 准备训练状态 (可选断点续训、设置 epoch、日志、切换训练模式)。
          step 2: 循环取 batch, 数据取完则进入下一个 epoch。
          step 3: forward 得到 loss, backward 计算/累积梯度。
          step 4: 累积到位后裁剪梯度、更新权重和学习率, global_step + 1。
          step 5: 按配置间隔记录日志、跑验证、保存 checkpoint。
          step 6: 达到 config.steps 后保存最终 checkpoint 并收尾。
        """
        logger.info("Starting Training Loop...")

        # ===== step 1: 准备训练状态 =====
        # Resume if configured 从指定路径加载 checkpoint
        if self.config.resume_from_checkpoint:
            self.load_checkpoint(self.config.resume_from_checkpoint)

        # Handle IterableDataset Epochs 【重要】每个 epoch 都打乱数据获取顺序
        if hasattr(self.train_dataloader.dataset, "set_epoch"): # 如果 dataset 有 set_epoch 方法
            # 更新 epoch 值，这个值影响数据获取的随机性、不同的 epoch 影响数据获取顺序
            self.train_dataloader.dataset.set_epoch(self.epoch)
        # Logger
        train_logger = TrainLogger(
            self.accelerator, 
            self.config.steps, 
            self.config.logging_steps
        )
        train_logger.start(self.global_step)

        # 切换到训练模式，对应 model.eval() 模式、进行推理和评估
        self.model.train()
        # 创建 dataloader 迭代器
        # 把可迭代对象转成迭代器，方便遍历。另，实现  def __iter__(self) 的类的对象就是一个可迭代对象
        train_iterator = iter(self.train_dataloader)

        logging_start_time = time.time()
        logging_start_step = self.global_step

        #【重要】
        # 创建一个标量张量 0.0 记录训练 loss，形状是 标量张量
        tr_loss = torch.tensor(0.0).to(self.accelerator.device)
        logging_loss_scalar = 0.0

        # ===== step 2: 主循环, 按 global_step 控制训练何时结束 =====
        """
        1. 迭代并获取数据，也有数据打乱逻辑
        2. forward() -> 获取 loss -> backward() 计算梯度 -如果
        """

        #【重要】训练程度/进程是通过配置的 steps 数量控制的
        # 比如 config.steps = 320、数据集有 100 个batch(1000个数据、batch_size 是10)、那么训练4个epoch后结束
        while self.global_step < self.config.steps:
            try:
                # 遍历迭代器，获取一个 batch 的数据、batch_size 大小是按照 batch_tokens 获取的
                batch = next(train_iterator)
            except StopIteration:
                # 【重要】这里并不是 异常情况，而是数据迭代完了
                self.epoch += 1
                logger.info(f"Epoch {self.epoch} starting. Resetting dataloader...")
                if hasattr(self.train_dataloader.dataset, "set_epoch"):
                    # 【重要】重新设置 epoch、打乱数据获取顺序
                    self.train_dataloader.dataset.set_epoch(self.epoch)
                # 【重要】重新创建迭代器
                train_iterator = iter(self.train_dataloader)
                batch = next(train_iterator) # 获取下一个 batch 的数据

            batch = _to_device(batch, self.accelerator.device)

            # ===== step 3: forward 求 loss, backward 计算/累积梯度 =====
            with self.accelerator.accumulate(self.model): 
                # 使用当前权重进行一次 forward
                outputs = self.model(**batch)
                # 获取 loss
                loss = outputs.loss
                tr_loss += loss.detach()
                # 【重要】
                # 计算如何调整参数来优化权重：写入 param.grad
                # Optimizer: optimizer.step() 用 grad 更新权重 -> optimizer.zero_grad() 清空 grad
                self.accelerator.backward(loss)

                # ===== step 4: 梯度累积到位后才真正更新权重 =====
                # sync_gradients 是判断多少个batch进行一次权重更新
                if self.accelerator.sync_gradients:
                    #【注释】梯度裁剪说明
                    # 是否配置了梯度裁剪，裁剪是防止梯度过大导致训练不稳定
                    # 参数更新量 = 学习率 × 梯度，所以梯度过大的时候 optimizer 调整的参数量可能过大、导致loss波动过大
                    grad_norm = 0.0
                    if self.config.max_grad_norm > 0:
                        # 【重要】如何进行梯度裁剪：返回的结果是裁剪前梯度范数
                        grad_norm = self.accelerator.clip_grad_norm_( # 裁剪梯度，norm 是 范数 的英文
                            # 梯度保存模型参数中： parameter.grad是backward() 计算出的梯度； parameter.data是当前权重
                            self.model.parameters(),
                            #【重要】裁剪规则：这个限制的是所有的梯度元素的 L2范数，如果超过则所有梯度元素按照相同比例进行缩放
                            self.config.max_grad_norm
                        )
                        grad_norm = (
                            grad_norm.item() if grad_norm is not None else 0.0
                        )

                    # 更新模型权重，基本逻辑是 新权重 = 旧权重 - 学习率 × 梯度
                    # 权重参数更新程度 与 【学习率 × 梯度】 正相关
                    self.optimizer.step()
                    # 更新学习率，这个值是先大后小
                    self.lr_scheduler.step()
                    # 清空本轮梯度
                    self.optimizer.zero_grad()
                    # 完成一个 step 的处理，递增 step，global_step 来判断训练啥时候结束
                    # 【注意】这里的 step 处理可能累加了多个 batch 数据的 forward/loss/backward/.step 计算
                    self.global_step += 1

                    # ===== step 5: 按间隔记录日志 / 验证 / 保存 =====
                    # Logging
                    current_lr = self.lr_scheduler.get_last_lr()[0] # 获取最新计算出的学习率
                    train_logger.update(
                        step=self.global_step, 
                        loss=loss.item(), 
                        lr=current_lr
                    )

                    # 都是记录日志，可先不看
                    if self.global_step % self.config.logging_steps == 0:
                        elapsed = time.time() - logging_start_time
                        steps_per_sec = (
                            (self.global_step - logging_start_step) / elapsed
                            if elapsed > 0
                            else 0
                        )

                        tr_loss_scalar = self.accelerator.gather(tr_loss).mean().item()
                        current_interval_loss = tr_loss_scalar - logging_loss_scalar
                        avg_loss = current_interval_loss / (
                            self.config.logging_steps
                            * self.config.gradient_accumulation_steps
                        )
                        logging_loss_scalar = tr_loss_scalar

                        logs = {
                            "train/loss": avg_loss,
                            "train/learning_rate": current_lr,
                            "train/grad_norm": grad_norm,
                            "train/epoch": self.epoch,
                            "train/steps_per_sec": steps_per_sec,
                        }
                        train_logger.log_metrics(step=self.global_step, metrics=logs)

                        logging_start_time = time.time()
                        logging_start_step = self.global_step

                    # Evaluate
                    # 到达评估间隔: 跑一遍验证集, 只统计 loss 不更新权重
                    if (
                        self.eval_dataloader is not None # 如果评测数据集不为空
                        and self.global_step % self.config.eval_steps == 0 # 如果当前步数是评估间隔，eval_steps 是评估间隔步数
                    ):
                        # 1. 仅仅是评估，并没有对训练流程产生印象，结果可一哦你过来参考：
                        #   1.1 如果 loss 持续变小则说明训练结果更好了
                        #   1.2 如果 loss 上升但是之前训练loss下降、说明训练过拟合
                        self.evaluate()

                    # Save
                    # 【重要】到达保存间隔: 保存 checkpoint, 支持后续断点续训
                    if self.global_step % self.config.save_steps == 0:
                        self.save_checkpoint(self.global_step)

        # ===== step 6: 训练结束, 保存最终 checkpoint 并收尾 =====
        # Final Save
        self.save_checkpoint(self.global_step)
        train_logger.close()
        self.accelerator.end_training()

