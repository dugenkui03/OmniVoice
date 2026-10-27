# `batch_input_ids` 与 `batch_attention_mask` 赋值说明

![Cond、Uncond 与 attention mask 赋值科普图](../images/batch_input_ids_and_attention_mask.png)

本文用同一组数据说明三个操作：

1. 将完整输入写入 Cond。
2. 将 target 段写入 Uncond。
3. 设置 Cond 和 Uncond 的 attention mask。

## 统一示例

假设：

```text
B=2，C=2，max_c_len=9，i=1，c_len=9，u_len=2

位置：       style       text          ref audio     target
Codebook 0：[101,102] [201,202,203]    [11,12]     [1024,1024]
Codebook 1：[101,102] [201,202,203]    [21,22]     [1024,1024]
```

当前输入为：

```python
inp["input_ids"] = [
    [
        [101, 102, 201, 202, 203, 11, 12, 1024, 1024],
        [101, 102, 201, 202, 203, 21, 22, 1024, 1024],
    ]
]  # shape: (1, C, c_len) = (1, 2, 9)
```

同一条输入会在 batch 中形成一对数据：

```text
batch 第 i 行：   [style][text][ref audio][target]  ← Cond
batch 第 B+i 行： [target][padding...]              ← Uncond
```

代入 `B=2`、`i=1`，它们分别位于 batch 第 `1` 行和第 `3` 行。

为了直观看出赋值位置，下面用 `99` 表示尚未写入的空位。实际代码中
`pad_id = audio_mask_id = 1024`，空位实际也填 `1024`，由其他 mask 区分
有效 target 和 padding。

赋值前：

```python
batch_input_ids = [
    [
        [99, 99, 99, 99, 99, 99, 99, 99, 99],
        [99, 99, 99, 99, 99, 99, 99, 99, 99],
    ],  # batch 第 0 行：Cond
    [
        [99, 99, 99, 99, 99, 99, 99, 99, 99],
        [99, 99, 99, 99, 99, 99, 99, 99, 99],
    ],  # batch 第 1 行：Cond，本例写入位置
    [
        [99, 99, 99, 99, 99, 99, 99, 99, 99],
        [99, 99, 99, 99, 99, 99, 99, 99, 99],
    ],  # batch 第 2 行：Uncond
    [
        [99, 99, 99, 99, 99, 99, 99, 99, 99],
        [99, 99, 99, 99, 99, 99, 99, 99, 99],
    ],  # batch 第 3 行：Uncond，本例写入位置
]  # shape: (2B, C, max_c_len) = (4, 2, 9)
```

## 1. 写入 Cond：保留完整序列

```python
batch_input_ids[i, :, :c_len] = inp["input_ids"]
```

左侧各维度的含义：

- `i`：选择 batch 第 `i` 行 Cond 数据。
- `:`：选择全部 `C` 个 codebook。
- `:c_len`：选择序列前 `c_len` 个位置。

代入 `i=1`、`c_len=9`：

```python
batch_input_ids[1, :, :9] = inp["input_ids"]
```

左侧形状为 `(C, c_len) = (2, 9)`，右侧形状为 `(1, 2, 9)`。PyTorch
索引赋值会兼容右侧多余的前导单元素维度；这里可以理解为先通过
`squeeze(0)` 得到 `(2, 9)`，再逐元素覆盖左侧区域。这个过程不会复制
数据，不属于沿 codebook 维展开数据的普通广播。

赋值后，batch 第 `1` 行为：

```python
[
    [101, 102, 201, 202, 203, 11, 12, 1024, 1024],
    [101, 102, 201, 202, 203, 21, 22, 1024, 1024],
]  # [style][text][ref audio][target]
```

## 2. 写入 Uncond：只保留 target

```python
batch_input_ids[B + i, :, :u_len] = inp["input_ids"][..., -u_len:]
```

右侧 `[..., -u_len:]` 保留前面的所有维度，并取得序列末尾的 target 段：

```python
inp["input_ids"][..., -2:] = [
    [
        [1024, 1024],
        [1024, 1024],
    ]
]  # shape: (1, 2, 2)
```

左侧 `B+i = 2+1 = 3`，因此实际赋值为：

```python
batch_input_ids[3, :, :2] = inp["input_ids"][..., -2:]
```

赋值后，batch 第 `3` 行为：

```python
[
    [1024, 1024, 99, 99, 99, 99, 99, 99, 99],
    [1024, 1024, 99, 99, 99, 99, 99, 99, 99],
]  # [target][padding...]
```

Uncond 去掉 style、text 和 ref audio，只保留 target，供 CFG 计算无条件预测。
target 只是移动到这条独立序列的开头，音频帧顺序没有改变。

## 3. 设置 `batch_attention_mask`

`batch_attention_mask` 控制序列位置之间是否可以互相关注：

```text
True：允许关注
False：禁止关注
```

### 3.1 创建

```python
batch_attention_mask = torch.zeros(
    (2 * B, 1, max_c_len, max_c_len),
    dtype=torch.bool,
)
```

形状 `(2B, 1, max_c_len, max_c_len)` 中，
`batch_attention_mask[b, 0, x, y] = True` 表示第 `b` 条样本的 query
位置 `x` 可以关注 key/value 位置 `y`。`x`、`y` 均覆盖补齐到
`max_c_len` 后的所有序列位置；第 `1` 维长度为 `1`，会广播到所有
attention head。不同 batch 样本之间不能通过该矩阵互相关注。

本例形状为 `(4, 1, 9, 9)`，即 `4` 个初始全为 `False` 的 `9×9`
attention 矩阵。

### 3.2 Cond：完整序列互相关注

```python
batch_attention_mask[i, :, :c_len, :c_len] = True
```

代入示例参数：

```python
batch_attention_mask[1, :, :9, :9] = True
```

batch 第 `1` 行对应的整个 `9×9` 区域被设置为 `True`，因此 Cond 的
`[style][text][ref audio][target]` 所有位置都可以互相关注。

### 3.3 Uncond：target 互相关注，padding 只关注自己

首先开放左上角的 `2×2` target 区域：

```python
batch_attention_mask[B + i, :, :u_len, :u_len] = True
# 等价于 batch_attention_mask[3, :, :2, :2] = True
```

然后生成 padding 位置索引，并只开放这些位置的对角线：

```python
pad_diag = torch.arange(u_len, max_c_len)
# pad_diag = [2, 3, 4, 5, 6, 7, 8]

batch_attention_mask[B + i, :, pad_diag, pad_diag] = True
```

最终 batch 第 `3` 行的 attention 矩阵为：

```text
[
  [T, T, F, F, F, F, F, F, F],
  [T, T, F, F, F, F, F, F, F],
  [F, F, T, F, F, F, F, F, F],
  [F, F, F, T, F, F, F, F, F],
  [F, F, F, F, T, F, F, F, F],
  [F, F, F, F, F, T, F, F, F],
  [F, F, F, F, F, F, T, F, F],
  [F, F, F, F, F, F, F, T, F],
  [F, F, F, F, F, F, F, F, T],
]
```

左上角 `2×2` 表示两个 target 位置可以互相关注；其余对角线上的 `T`
表示每个 padding 位置只允许关注自己。

### 3.4 `pad_diag` 的成对高级索引

下面这行使用 PyTorch 的高级索引（advanced indexing），更具体地说是
成对整数索引：

```python
batch_attention_mask[B + i, :, pad_diag, pad_diag] = True
```

固定 batch 和长度为 `1` 的第 1 维后，可以先把 attention mask 简化成
二维矩阵：

```python
matrix = batch_attention_mask[B + i, 0]
```

此时：

```text
matrix.shape = (max_c_len, max_c_len)
```

原赋值可以简化为：

```python
matrix[pad_diag, pad_diag] = True
```

假设：

```python
pad_diag = torch.tensor([2, 3, 4])
```

两个索引张量会按元素位置逐项配对：

```text
第一个 pad_diag：2       3       4
第二个 pad_diag：2       3       4
最终坐标：       (2,2)   (3,3)   (4,4)
```

所以原赋值等价于：

```python
matrix[2, 2] = True
matrix[3, 3] = True
matrix[4, 4] = True
```

它只设置三个对角线点，不会设置 `(2,3)`、`(2,4)` 等位置。二维选择结果
`matrix[pad_diag, pad_diag]` 的形状为 `(3,)`；保留原张量第 1 维的完整
写法，选择结果形状为 `(1, 3)`。

两个高级索引张量不要求形状完全相同，但必须能够广播：

```python
# 形状分别为 (3,) 和 (4,)，无法配对或广播，会报错
matrix[torch.tensor([1, 2, 3]), torch.tensor([4, 5, 6, 7])]

# 形状分别为 (2, 1) 和 (1, 3)，广播为 (2, 3)
row_idx = torch.tensor([2, 3])[:, None]
col_idx = torch.tensor([4, 5, 6])[None, :]
matrix[row_idx, col_idx] = True
```

第二种写法会生成全部六个坐标：

```text
(2,4) (2,5) (2,6)
(3,4) (3,5) (3,6)
```

因此，当前代码重复使用同一个 `pad_diag`，目的就是让 query 和 key
索引一一对应，只开放 padding 区域的对角线。通用的索引、维度消除与
广播规则也可参考
[第二十一章 21.10 节](../chapter21_扩展知识四_注意力机制QKV与MultiHeadAttention.md#2110-代码阅读补充索引赋值单元素维度与广播)。
