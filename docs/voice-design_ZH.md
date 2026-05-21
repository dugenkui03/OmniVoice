# 声音设计 (Voice Design)

声音设计（Voice Design）模式允许您通过描述说话人的特征属性（`instruct` 参数）来定制音色 —— 无需提供任何参考音频。模型会实时生成符合描述的声音。

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
    text="这是一次声音设计的测试。",
    instruct="女，青年，高音调，四川话",
)
```

## 工作原理

`instruct` 参数接收以逗号分隔的说话人特征属性字符串。
每个属性都属于某个特定的**类别**（性别、年龄、音调、风格、英文口音或中文方言）。在同一个类别中，每次只能选择一个属性；不同类别的属性可以自由组合。

模型会自动检测指令文本的语言并在内部进行标准化 —— 您可以使用英文、中文或两者混合进行书写。

## 支持的属性列表

### 性别 (Gender)

| 英文 | 中文 |
|---------|---------|
| male | 男 |
| female | 女 |

### 年龄 (Age)

| 英文 | 中文 |
|---------|---------|
| child | 儿童 |
| teenager | 少年 |
| young adult | 青年 |
| middle-aged | 中年 |
| elderly | 老年 |

### 音调 (Pitch)

| 英文 | 中文 |
|---------|---------|
| very low pitch | 极低音调 |
| low pitch | 低音调 |
| moderate pitch | 中音调 |
| high pitch | 高音调 |
| very high pitch | 极高音调 |

### 风格 (Style)

| 英文 | 中文 |
|---------|---------|
| whisper | 耳语 |

### 英文口音 (English Accent)

**仅在合成的文本内容为英文时生效。**

| 口音描述 |
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

### 中文方言 (Chinese Dialect)

**仅在合成的文本内容为中文时生效。**

| 方言描述 |
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

## 编写指令字符串

使用逗号分隔不同的属性（英文使用半角 `,`，中文使用全角 `，` —— 模型会自动修复匹配错误）。

```
# 英文写法
"female, young adult, high pitch, british accent"

# 中文写法
"女，青年，高音调，四川话"

# 混合写法（会自动标准化）
"female, young adult, 四川话"
```

### 实用技巧

- **自由组合**不同类别的属性：例如 `"male, elderly, low pitch, whisper"`（男，老年，低音调，耳语）。
- **交给模型发挥**：省去您不在意的属性 —— 模型会自动填补其余部分的特征。例如，仅提供 `"female"` 也是完全有效的。
- **大小写不敏感**：`"Male"`、`"MALE"` 和 `"male"` 均可接受，代码会统一转换为小写进行标准化。
- **口音与方言的适用范围**：英文口音仅适用于英文语音，中文方言仅适用于中文语音。
- **属性组合局限**：受训练数据限制，某些特征的组合可能效果不佳 —— 模型可能会忽略该组合中的某些属性。如果输出不符合您的预期，请尝试简化指令字符串。
