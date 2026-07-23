# `_prepare_inference_inputs` 的 `input_ids` 结构

`_generate_iterative()` 会为 batch 中的每条任务调用一次
`_prepare_inference_inputs()`，结果组成 `inputs_list`：

```python
inputs_list = [
    {"input_ids": Tensor(1, C, S0), "audio_mask": Tensor(1, S0)},
    {"input_ids": Tensor(1, C, S1), "audio_mask": Tensor(1, S1)},
    # ...
]
```

不同任务的序列长度 `S` 可以不同，后续再统一 padding 到 `max_c_len`。

## `input_ids` 的四段结构

```text
[style] + [text] + [可选 ref audio] + [target MASK]
```

### 1. `style`：怎么说

`style` 由以下信息组成：

```text
[是否去噪] + [目标语言] + [声音风格]
```

对应结构：

```text
<|denoise|>
<|lang_start|>语言<|lang_end|>
<|instruct_start|>风格指令<|instruct_end|>
```

例如：

```python
lang = "en"
instruct = "female, whisper"
denoise = True
```

得到：

```text
<|denoise|><|lang_start|>en<|lang_end|><|instruct_start|>female, whisper<|instruct_end|>
```

### 2. `text`：说什么

`text` 由参考音频文本和目标生成文本组成：

```text
ref_text + target text
```

并包在文本边界标记中：

```text
<|text_start|>ref_text target text<|text_end|>
```

如果没有 `ref_text`，其中只包含目标生成文本。

### 3. `ref audio` 与 `target MASK`

- `ref audio`：可选的参考音频 token，形状为 `(C, T_ref)`。
- `target MASK`：等待模型生成的目标音频区域，初始值全部为 `audio_mask_id`，当前是 `1024`。

## 简化示例

为了便于观察，假设只有两个 codebook：

```text
位置：       style       text          ref audio     target
Codebook 0：[101,102] [201,202,203]    [11,12]     [1024,1024]
Codebook 1：[101,102] [201,202,203]    [21,22]     [1024,1024]
```

对应的 `input_ids` 为：

```python
[
    [
        [101, 102, 201, 202, 203, 11, 12, 1024, 1024],
        [101, 102, 201, 202, 203, 21, 22, 1024, 1024],
    ]
]
```

形状为 `(1, 2, 9)`；OmniVoice 实际使用 `C=8`。

对应的 `audio_mask` 为：

```python
[[False, False, False, False, False, True, True, True, True]]
```

`False` 表示文本位置，`True` 表示参考音频或目标音频位置。
