# 第二十三章：扩展知识六 —— 常用张量与模型方法速查

阅读 OmniVoice 这类基于 Transformer 的 TTS 源码时，反复出现的往往是一小组 PyTorch 张量操作和模型模块方法。理解这组方法，就能读懂大部分数据在进入 Transformer 前后的形状变化。本章按用途归类，给出每个方法的作用、形状规则、最小示例，以及它在 OmniVoice 中的具体位置。

本章覆盖的方法可以先按用途分组：

```mermaid
flowchart LR
    subgraph createStage["创建张量"]
        A1["torch.full"]
        A2["torch.arange"]
        A3["torch.full_like"]
    end

    subgraph shapeStage["形状变换"]
        B1["索引 / 切片"]
        B2["unsqueeze / squeeze"]
        B3["view / reshape"]
        B4["repeat"]
    end

    subgraph mathStage["逐元素与归约"]
        C1["广播 + / *"]
        C2["sum(dim=)"]
    end

    subgraph mergeStage["选择与拼接"]
        D1["torch.where"]
        D2["torch.cat"]
    end

    subgraph moduleStage["模型模块"]
        E1["nn.Embedding + weight"]
        E2["nn.Linear"]
        E3["register_buffer"]
        E4["get_input_embeddings"]
    end

    createStage --> shapeStage --> mathStage --> mergeStage --> moduleStage
    moduleStage -.-> L["图例：橙=方法分组"]

    classDef core fill:#FFF2CC,stroke:#D88B00,stroke-width:2px,color:#3B2A00;
    classDef note fill:#FFFDF4,stroke:#B7A870,stroke-dasharray:3 3,color:#5A5134;
    class A1,A2,A3,B1,B2,B3,B4,C1,C2,D1,D2,E1,E2,E3,E4 core;
    class L note;
    style createStage fill:#FFFDF4,stroke:#D88B00,stroke-dasharray:6 4,color:#3B2A00;
    style shapeStage fill:#FFFDF4,stroke:#D88B00,stroke-dasharray:6 4,color:#3B2A00;
    style mathStage fill:#FFFDF4,stroke:#D88B00,stroke-dasharray:6 4,color:#3B2A00;
    style mergeStage fill:#FFFDF4,stroke:#D88B00,stroke-dasharray:6 4,color:#3B2A00;
    style moduleStage fill:#FFFDF4,stroke:#D88B00,stroke-dasharray:6 4,color:#3B2A00;
```

> 💡 **小科普：Tensor、PyTorch、Transformer 不是一回事**
>
> Tensor 是多维数字容器（数据结构）；PyTorch 是操作 Tensor、构建神经网络的框架；Transformer 是一种神经网络架构。本章的方法大多来自 PyTorch，用来在 Transformer 前后整理 Tensor 的形状和数值。

## 1. 创建张量

### `torch.full(size, fill_value)`

创建一个指定形状、所有元素都等于同一个值的张量。

```text
torch.full((1, 8, 3), 1024)
→ 形状 (1,8,3) 的张量，每个元素都是 1024
```

在 OmniVoice 中，它用于初始化"全是 MASK"的目标音频区域：

```python
target_audio_tokens = torch.full(
    (1, self.config.num_audio_codebook, num_target_tokens),
    self.config.audio_mask_id,   # 1024
    dtype=torch.long,
    device=self.device,
)
```

- 第 1 个参数 `size` 是形状元组 `(1, C, T)`。
- 第 2 个参数 `fill_value` 是填充值，这里是 `audio_mask_id=1024`。
- `dtype=torch.long` 表示整数 token ID；`device` 保证与模型同设备。

生成后是一个 `(1, 8, T)` 的全 MASK 矩阵，等待迭代解码逐步填入真实 codec token。

### `torch.arange(n)`

生成 `0, 1, 2, ..., n-1` 的一维整数序列。

```text
torch.arange(8) → [0, 1, 2, 3, 4, 5, 6, 7]
```

OmniVoice 用它构造 codebook 层偏移量：

```python
torch.arange(config.num_audio_codebook) * config.audio_vocab_size
# [0, 1, ..., 7] * 1025 → [0, 1025, 2050, ..., 7175]
```

### `torch.full_like(x, value)`

按照参照张量 `x` 的形状和设备创建一个填满 `value` 的新张量，无需手写 `size`。

```python
probs = torch.full_like(logits, float("-inf"))
# 与 logits 形状相同，全部填 -inf，常用于采样前初始化
```

## 2. 形状变换

这一类方法只改变张量的"组织方式"，不做数学计算（`repeat` 会复制数据）。

| 方法 | 作用 | 形状变化 |
| --- | --- | --- |
| 索引 / 切片 | 取子张量 | 整数索引会消除该维度 |
| `unsqueeze(dim)` | 插入长度 1 的新维度 | 维数 +1 |
| `squeeze(dim)` | 去掉长度 1 的维度 | 维数 -1 |
| `view` / `reshape` | 重新排列同一批数据的形状 | 元素总数不变 |
| `repeat(...)` | 按倍数复制数据 | 各维乘以对应倍数 |

### 索引与切片

`input_ids[:, 0, :]` 表示：所有 batch、第 0 个 codebook、所有序列位置。整数索引 `0` 会消除 codebook 维：

```text
(B, C, S) --input_ids[:, 0, :]--> (B, S)
```

如果写成切片 `input_ids[:, 0:1, :]`，则保留该维为长度 1，得到 `(B, 1, S)`。

### `unsqueeze(dim)` 与 `squeeze(dim)`

`unsqueeze` 在指定位置插入一个长度为 1 的维度，数值不变，只增加一层组织结构：

```text
(2, 5) --unsqueeze(1)--> (2, 1, 5)
```

OmniVoice 中常用它为广播准备维度：

```python
audio_mask.unsqueeze(1)    # (B,S) -> (B,1,S)，广播到 C 层
audio_mask.unsqueeze(-1)   # (B,S) -> (B,S,1)，广播到 H 维
ref_audio_tokens.unsqueeze(0)  # (C,T) -> (1,C,T)，补 batch 维
```

`squeeze` 是反向操作，去掉长度为 1 的维度，例如把 tokenizer 输出的 `(1,C,T)` 压回 `(C,T)`：

```python
...audio_codes.squeeze(0)
```

### `view` / `reshape`

在元素总数不变的前提下重新排列形状。`-1` 表示该位置由其余维度自动推断。

```python
self.codebook_layer_offsets.view(1, -1, 1)   # (8,) -> (1,8,1)
audio_logits = logits_flat.view(B, C, S, V)   # 展平的 logits 还原成四维
```

`view` 不改变数值，也不做转置，只是换一种"读法"。

> 💡 **小科普：`view` 和 `reshape` 有什么区别？**
>
> 两者语义相同，都用于改变形状。`view` 要求张量在内存中连续，否则会报错；`reshape` 在必要时会自动复制数据以保证成功。读代码时可以先都理解成"改形状"。

### `repeat(...)`

按给定倍数复制数据，参数是每个维度的**重复次数**，不是目标大小。

```text
x.shape = (1, 3)
x.repeat(8, 1) → (8, 3)   # 第 0 维复制 8 份，第 1 维不变
```

OmniVoice 用它把单层文本 token 复制到 8 层，以便和音频 token 组成规则的 `(B, C, S)`：

```python
.input_ids.repeat(self.config.num_audio_codebook, 1)  # (1,N) -> (8,N)
.unsqueeze(0)                                          # (8,N) -> (1,8,N)
```

## 3. 逐元素运算与广播

### 广播（broadcasting）

当两个张量形状不同但"可对齐"时，PyTorch 会把长度为 1 的维度自动扩展到另一方的长度，无需真正复制数据。

```text
input_ids:               (B, C, S)
audio_mask.unsqueeze(1): (B, 1, S)
                              ↓ 广播
                         (B, C, S)
```

对齐规则从右向左逐维比较：维度相等，或其中一方为 1，即可广播。

### 逐元素乘 `*` 与加 `+`

`*` 是逐元素相乘，不是矩阵乘法。布尔张量参与运算时按 `True=1`、`False=0` 处理：

```text
input_ids:  [23, 56, 91, 42]
audio_mask: [ F,  T,  T,  F]
相乘:       [ 0, 56, 91,  0]   # 保留 True 位置，清零 False 位置
```

OmniVoice 用乘法保留音频位置的 ID，用加法叠加层偏移：

```python
shifted_ids = (
    input_ids * audio_mask.unsqueeze(1)          # 保留音频 ID
) + self.codebook_layer_offsets.view(1, -1, 1)   # 加层偏移得到全局行号
```

### `sum(dim=k)`

沿指定维度求和，该维度会被"压掉"。维度编号必须结合**当前完整张量的 shape**理解。

```text
(B, C, S, H) --sum(dim=1)--> (B, S, H)   # 沿 codebook 维求和
```

OmniVoice 用它把同一时间帧的 8 层 codebook 向量合并成一个向量：

```python
audio_embeds = self.audio_embeddings(shifted_ids).sum(dim=1)
```

这里求和的是训练学到的高维向量（本项目 `H=1024`），不是 token ID，因此不同层组合求和后一般仍然可区分。

## 4. 选择与拼接

这两个方法都会"把多份数据放到一起"，但机制完全不同：`torch.where` 逐位置二选一、形状不变；`torch.cat` 沿某维首尾接长。

### `torch.where(cond, a, b)`

对每个位置，`cond` 为真取 `a`，为假取 `b`。三者形状需可广播，输出形状不变。

```python
torch.where(audio_mask.unsqueeze(-1), audio_embeds, text_embeds)
```

- `audio_mask.unsqueeze(-1)`：`(B,S)` → `(B,S,1)`，再广播到 `H`。
- `True` 位置取音频向量，`False` 位置取文本向量。
- 输入输出都是 `(B,S,H)`，长度不变。

它用于在文本 embedding 和音频 embedding 之间**逐位置选择**，而不是把两段接起来。

### `torch.cat(tensors, dim=k)`

把多个张量沿"已有维度"首尾拼接，其余维度必须一致。拼接会让该维度变长。

```python
cond_input_ids = torch.cat(parts, dim=2)
# style + text + ref_audio + target，沿序列维 S 拼成 (1,8,S)
```

例如四段长度为 4、5、2、3 的片段沿 `S` 拼接后，`S = 4+5+2+3 = 14`。

> 💡 **小科普：`torch.where` 与 `torch.cat` 的关键区别**
>
> `torch.where` 是"同一位置在两个候选之间二选一"，序列长度不变；`torch.cat` 是"把不同片段接成更长的序列"，长度相加。OmniVoice 里文本和音频的拼接发生在更早的 **token ID 层**（`torch.cat`），而文本 / 音频 embedding 的合并发生在 **向量层**（`torch.where`）。

## 5. 模型模块

前面几类都是纯张量操作，不含可训练参数。这一类是 `nn.Module` 子模块，内部保存模型权重。

### `nn.Embedding(num, dim)` 与 `.weight`

`nn.Embedding` 是"整数 ID → 向量"的查找表模块。构造时立即创建一张形状为 `(num, dim)` 的可训练权重矩阵，初值随机：

```python
self.audio_embeddings = nn.Embedding(8 * 1025, H)
# self.audio_embeddings.weight.shape = (8200, H)
```

- `.weight` 是内部的 `nn.Parameter`（可训练参数），即那张向量表。
- 查表等价于按行号索引：`embedding(ids)` ≈ `embedding.weight[ids]`。
- 输入形状 `(*)`，输出在末尾追加向量维 `(*, H)`，不发生转置。

调用 `self.audio_embeddings(ids)` 会经过 `nn.Module.__call__()` 触发前向查表，无需手写 `.forward()`。

> 💡 **小科普：`nn.Embedding(...)` 不是赋空值**
>
> 这行代码是在构造模块并立即分配权重矩阵，只是初值为随机数。训练时被梯度更新，或 `from_pretrained()` 从 checkpoint 覆盖后，这张表才具有实际意义。

### `nn.Linear(in, out, bias=False)`

全连接层，做一次 `y = x @ weight.T (+ bias)` 的线性变换，权重形状 `(out, in)`。OmniVoice 用它把 Transformer 的 hidden states 一次性投影成 8 层 codebook 的 logits：

```python
self.audio_heads = nn.Linear(
    self.config.llm_config.hidden_size,               # in = H
    config.num_audio_codebook * config.audio_vocab_size,  # out = 8×1025
    bias=False,
)
```

输出再 `view` 成 `(B, C, S, V)`，得到每层 codebook 各自的分类分布。这说明输入侧用"求和"合并信息，输出侧用独立预测头分层还原，两者是不同模块。

### `register_buffer(name, tensor)`

注册一个"随模型走但不参与训练"的张量。它会跟随模型移动到 CPU / GPU、进入 `state_dict`，但优化器不会更新它。

```python
self.register_buffer(
    "codebook_layer_offsets",
    torch.arange(config.num_audio_codebook) * config.audio_vocab_size,
)
```

`codebook_layer_offsets` 是固定的行号规则，因此适合作为 buffer，而不是可训练参数。

### `get_input_embeddings()` / `set_input_embeddings()`

Hugging Face 模型的标准接口，返回或替换输入文本 Embedding 模块。OmniVoice 把它转发给内部 LLM 主干：

```python
def get_input_embeddings(self):
    return self.llm.get_input_embeddings()
```

它返回的是**模块**而非向量，再传入 token ID 调用一次才做查表。使用标准接口而非直接访问 `weight`，可让 `resize_token_embeddings()`、checkpoint 加载保存、输入输出权重共享等通用流程自动复用。

### `.to(device)`

把张量或模块搬到指定设备（CPU / GPU）。参与同一次运算的张量必须在同一设备上。

```python
ref_audio_tokens.unsqueeze(0).to(self.device)
```

## 6. 参数与 buffer 的区别

理解一个张量"是否被训练、是否随模型保存"，是读模型代码的关键。

| 类型 | 例子 | 可训练 | 进入 state_dict | 说明 |
| --- | --- | --- | --- | --- |
| 可训练参数 `nn.Parameter` | `audio_embeddings.weight`、`nn.Linear.weight` | 是 | 是 | 训练时被梯度更新，构成"模型权重" |
| buffer | `codebook_layer_offsets` | 否 | 是 | 固定规则或统计量，随模型移动但不训练 |
| 普通中间张量 | `shifted_ids`、`audio_mask` | 否 | 否 | 前向计算的临时结果 |

"模型权重"指的是可训练参数里的那些数字，模型学到的能力就存在其中。像 `torch.where`、`sum`、`unsqueeze`、`view` 这类操作只整理数据，不携带权重；真正携带权重的是 `nn.Embedding`、`nn.Linear` 等模块。

## 7. 组合示例：一行代码里多个方法怎么串起来

前面的方法很少单独出现，源码里经常是好几个嵌在一行。读这类代码的窍门是**从最里层往外层逐步算形状和数值**。下面用几个 OmniVoice 里的真实例子演示。

### 示例一：用 RMS 计算参考音频响度

`create_voice_clone_prompt` 里有一行计算参考音频响度：

```python
ref_rms = float(np.sqrt(np.mean(ref_wav ** 2)))
```

其中 `ref_wav` 是形状 `(1, T)` 的波形数组。从里到外拆开：

```text
ref_wav ** 2        逐元素平方 → (1, T)，每个采样点各自平方
np.mean(...)        不带 axis → 对全部元素求平均 → 标量（均方）
np.sqrt(...)        开平方 → 标量（均方根 RMS）
float(...)          numpy 标量 → Python float
```

三个关键点：

- **`** 2` 是逐元素幂**，不是矩阵乘方。`(1,T)` 平方后仍是 `(1,T)`，负数平方变正。
- **`np.mean` 不传 `axis`，所以对整个数组求平均，结果是标量**（`()` 形状）。若写成 `np.mean(x, axis=1)` 才会保留通道轴得到 `(1,)`。
- **`np.sqrt` 是逐元素开方**，这里作用在标量上，得到最终的 RMS 标量。

一个具体数字例子：

```text
ref_wav = [[0.2, -0.5, 0.1]]        # (1, 3)

** 2   → [[0.04, 0.25, 0.01]]        # 逐元素平方
mean   → (0.04 + 0.25 + 0.01) / 3 = 0.1   # 全体平均 → 标量
sqrt   → 0.316...                    # 开方 → RMS
```

> 💡 **小科普：`**`、`np.mean`、`np.sqrt` 都是逐元素或归约操作**
>
> `**` 和 `np.sqrt` 是逐元素运算，形状不变；`np.mean` 是归约操作，会把被平均的轴压掉（不带 `axis` 时压成标量）。这三个组合起来就是"平方 → 求均值 → 开方"的 RMS 公式。

#### 为什么 RMS 能衡量响度

响度可以理解为声波振动的能量强弱。波形 `ref_wav` 的每个采样点是振幅，也就是偏离静音 `0` 的位移，振幅摆动越大声音越响。难点在于振幅有正有负（声波上下振动），直接求平均会正负抵消、趋近 `0`，反映不出强弱。RMS 的三步正好解决这个问题：

```text
① 平方：把正负都变成正数，并放大大振幅——平方值就是每个采样点的“能量”
② 平均：对整段能量取平均，得到“平均能量”，代表整体有多“用力”
③ 开方：把平方放大的量纲拉回到与振幅一致，得到可直接比较的响度标量
```

物理上，声音的能量正比于振幅的平方，RMS 正是"平均能量再开方"，因此它比"直接平均"或"平均绝对值"更贴近人对响度的感受，是音频领域衡量响度的标准指标。这也解释了代码里的用途：`ref_rms` 就是这段音频"平均有多响"的量化值，用来放大过轻的参考音频、并给输出做音量归一。

### 示例二：`shifted_ids` 的层偏移

音频分支里把局部 token ID 变成共享大表的全局行号：

```python
shifted_ids = (
    input_ids * audio_mask.unsqueeze(1)
) + self.codebook_layer_offsets.view(1, -1, 1)
```

从里到外：

```text
audio_mask.unsqueeze(1)          (B,S) → (B,1,S)      # 加广播维
input_ids * ...                  (B,C,S)，文本位置清零 # 逐元素乘 + 广播
codebook_layer_offsets.view(...) (C,) → (1,C,1)       # 改形状便于广播
两者相加                          (B,C,S)              # 广播加法
```

同一行里就串了 `unsqueeze`、广播乘、`view`、广播加四个操作。

### 示例三：`audio_embeds` 的查表加求和

```python
audio_embeds = self.audio_embeddings(shifted_ids).sum(dim=1)
```

```text
self.audio_embeddings(shifted_ids)   (B,C,S) → (B,C,S,H)   # Embedding 查表
.sum(dim=1)                          (B,C,S,H) → (B,S,H)   # 沿 codebook 维求和
```

一行里是"查表 + 归约"两步。读法同样是先看里层的查表输出形状，再看外层 `sum` 压掉哪一维。

## 8. 方法速查表

| 方法 | 一句话作用 | OmniVoice 中的典型位置 |
| --- | --- | --- |
| `torch.full` | 建指定形状、同值张量 | 初始化全 MASK 目标区域 |
| `torch.arange` | 生成 0..n-1 序列 | 计算 codebook 层偏移 |
| `torch.full_like` | 按参照张量建同形张量 | 采样前初始化 `-inf` |
| 索引 / 切片 | 取子张量，整数索引消除该维 | `input_ids[:, 0, :]` 取文本层 |
| `unsqueeze` | 插入长度 1 的维度 | 为广播加 codebook / batch 维 |
| `squeeze` | 去掉长度 1 的维度 | 压回 `(C,T)` |
| `view` / `reshape` | 改形状（元素总数不变） | 层偏移 `(8,)`→`(1,8,1)`、logits 还原 |
| `repeat` | 按倍数复制数据 | 文本复制到 8 层 |
| 广播 `+ *` | 形状对齐后逐元素运算 | mask 保留 ID、叠加层偏移 |
| `sum(dim=)` | 沿某维求和并压掉该维 | 合并 8 层 codebook 向量 |
| `torch.where` | 逐位置二选一，长度不变 | 文本 / 音频 embedding 选择 |
| `torch.cat` | 沿某维拼接，长度相加 | 拼接 style/text/ref/target token |
| `nn.Embedding` + `.weight` | ID→向量查表，权重可训练 | 音频 / 文本 embedding |
| `nn.Linear` | 线性投影 | `audio_heads` 输出 logits |
| `register_buffer` | 注册不训练的随模型张量 | `codebook_layer_offsets` |
| `get_input_embeddings` | 取文本 Embedding 模块 | 转发给内部 LLM |
| `.to(device)` | 搬到指定设备 | 张量对齐到模型设备 |
