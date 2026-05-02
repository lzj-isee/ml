# LLM 解码与采样

大模型生成阶段的解码策略与采样方法，涵盖 Temperature、Penalty、Top-k、Top-p 等。

---

## 目录

- [频率惩罚 & 存在惩罚](#频率惩罚--存在惩罚)

---

## 频率惩罚 & 存在惩罚

**Frequency Penalty** (频率惩罚) 和 **Presence Penalty** (存在惩罚) 是用于在生成质量和多样性之间进行权衡的方法。

### 原理

这两个惩罚都是在 **logits** 上直接**减法**实现的，降低已出现 token 的选择概率。

### Frequency Penalty (频率惩罚)

让 token 每次在文本中出现都受到惩罚，**进行累计**。

```python
# 在生成第 i 个 token 时，对已有 token 累计惩罚
for token_id in used_tokens:
    count = frequency_count[token_id]  # 该token已出现的次数
    logits[token_id] -= frequency_penalty * count
```

**公式**: `logit -= p * count`

### Presence Penalty (存在惩罚)

如果一个 token 已经在文本中出现过，就会受到**固定惩罚**。

```python
# 只要出现过，就固定惩罚
for token_id in set(used_tokens):
    logits[token_id] -= presence_penalty  # 固定值，与次数无关
```

**公式**: `logit -= p` (if appeared)

### 对比

| 特性 | Frequency Penalty | Presence Penalty |
|------|-------------------|------------------|
| **惩罚力度** | 与出现次数成正比 | 固定值 |
| **公式** | `logit -= p * count` | `logit -= p` (if appeared) |
| **效果** | 强烈抑制重复词 | 轻度抑制已用词，保留多样性 |
| **典型值** | 0.0 ~ 2.0 | 0.0 ~ 2.0 |

### 代码实现

```python
from collections import Counter

def apply_penalties(logits, used_token_ids, freq_penalty=0.0, presence_penalty=0.0):
    """
    logits: [vocab_size] 原始输出
    used_token_ids: [seq_len] 已生成的token序列
    """
    counts = Counter(used_token_ids)
    
    # Frequency Penalty: 每个出现的token按次数累计惩罚
    for token_id, count in counts.items():
        logits[token_id] -= freq_penalty * count
    
    # Presence Penalty: 只要出现过就固定惩罚（去重后）
    unique_tokens = set(used_token_ids)
    for token_id in unique_tokens:
        logits[token_id] -= presence_penalty
    
    return logits
```

### 为什么用减法？

- **logits 是 softmax 前的值**，减法等价于除法作用于概率
- `log(p) - c = log(p / e^c)`，即降低该 token 的概率
- 减法比除法计算更简单直接

### 与 Temperature 的区别

| 参数 | 作用方式 | 效果 |
|------|----------|------|
| **Temperature** | 改变整个概率分布的尖锐程度 | 低T更确定，高T更随机 |
| **Penalty** | 直接修改特定token的logits | 抑制重复，增加多样性 |

---

*（待补充：Temperature、Top-k、Top-p、Beam Search 等解码方法）*
