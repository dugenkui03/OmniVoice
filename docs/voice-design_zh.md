# Voice Design（声音设计）

Voice Design 模式允许你通过 speaker attributes（说话人属性，即 `instruct` 参数）描述想要的说话人，不需要参考音频。模型会即时生成匹配的声音。

## 快速示例

```python
import torch
from omnivoice import OmniVoice

model = OmniVoice.from_pretrained(
    "k2-fsa/OmniVoice",
    device_map="cuda:0",
    dtype=torch.float16
)

audio = model.generate(
    text="This is a test for voice design.",
    instruct="female, young adult, high pitch, british accent",
)
```

## 工作方式

`instruct` 参数接受用逗号分隔的说话人属性字符串。每个属性属于一个**类别**（性别、年龄、音高、风格、口音或方言）。同一个类别中一次只能选择一个属性；不同类别的属性可以自由组合。

模型会自动检测 instruct 文本的语言，并在内部进行标准化；你可以使用英文、中文，或者中英混写。

## 支持的属性

### 性别

| 英文 | 中文 |
|---------|---------|
| male | 男 |
| female | 女 |

### 年龄

| 英文 | 中文 |
|---------|---------|
| child | 儿童 |
| teenager | 少年 |
| young adult | 青年 |
| middle-aged | 中年 |
| elderly | 老年 |

### 音高

| 英文 | 中文 |
|---------|---------|
| very low pitch | 极低音调 |
| low pitch | 低音调 |
| moderate pitch | 中音调 |
| high pitch | 高音调 |
| very high pitch | 极高音调 |

### 风格

| 英文 | 中文 |
|---------|---------|
| whisper | 耳语 |

### 英语口音

仅当合成文本为英文时有效。

| 口音 |
|--------|
| american accent |
| british accent |
| australian accent |
| canadian accent |
| indian accent |
| chinese accent |
| korean accent |
| japanese accent |
| portuguese accent |
| russian accent |

### 中文方言

仅当合成文本为中文时有效。

| 方言 |
|---------|
| 河南话 |
| 陕西话 |
| 四川话 |
| 贵州话 |
| 云南话 |
| 桂林话 |
| 济南话 |
| 石家庄话 |
| 甘肃话 |
| 宁夏话 |
| 青岛话 |
| 东北话 |

## 编写 Instruct 字符串

用逗号分隔属性（英文使用半角 `,`，中文使用全角 `，`；模型会自动修复不匹配的逗号）。

```text
# 英文
"female, young adult, high pitch, british accent"

# 中文
"女，青年，高音调，四川话"

# 混写（会自动标准化）
"female, young adult, 四川话"
```

### 提示

- **跨类别自由组合**：例如 `"male, elderly, low pitch, whisper"`。
- **交给模型决定**：可以省略你不关心的属性，模型会补全其余属性。例如只写 `"female"` 也是有效的。
- **大小写不敏感**：`"Male"`、`"MALE"` 和 `"male"` 都能被接受，代码会把它们标准化成小写。
- **口音 vs 方言**：英语口音只作用于英文语音，中文方言只作用于中文语音。
- **属性组合效果**：受训练数据限制，某些属性组合可能效果不好，模型可能会忽略组合中的部分属性。如果输出不符合预期，可以尝试简化 instruct 字符串。
