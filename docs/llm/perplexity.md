# 困惑度 Perplexity

> 语言模型评估的核心指标：衡量模型对文本序列的预测能力

---

## 什么是困惑度？

**困惑度（Perplexity, PPL）** 是衡量语言模型性能的经典指标，它表示模型在面对下一个词时的"困惑"程度。

### 直观理解

- **困惑度 = 3**：相当于每次预测下一个词时，面对3个等概率的选择
- **困惑度 = 100**：相当于面对100个等概率选择
- **困惑度越低** → 模型对文本的预测越准确 → 模型性能越好

---

## 计算步骤

### Step 1: 计算联合概率

语言模型对测试数据集的联合概率可以分解为条件概率的乘积：

$$
P(w_1, w_2, \dots, w_N) = \prod_{i=1}^N P(w_i \mid w_1, w_2, \dots, w_{i-1})
$$

其中：
- $P(w_i \mid w_1, w_2, \dots, w_{i-1})$：模型根据上文预测当前词 $w_i$ 的概率
- $N$：序列长度（token 数）

### Step 2: 计算对数概率

为了避免数值下溢（概率乘积可能极小），通常使用对数概率：

$$
\log P(w_1, w_2, \dots, w_N) = \sum_{i=1}^N \log P(w_i \mid w_1, w_2, \dots, w_{i-1})
$$

**为什么用对数？**
- 乘法变加法，计算更稳定
- 避免浮点数下溢（概率连乘可能小于最小正浮点数）

### Step 3: 计算困惑度

将对数概率代入困惑度公式：

$$
\text{Perplexity} = \exp\left(-\frac{1}{N} \sum_{i=1}^N \log P(w_i \mid w_1, w_2, \dots, w_{i-1})\right)
$$

---

## 公式简化与等价形式

### 与交叉熵的关系

困惑度实际上是**交叉熵的指数形式**：

$$
\text{Perplexity} = \exp(H(P, Q)) = \exp\left(-\frac{1}{N} \sum_{i=1}^N \log P(w_i \mid \text{context})\right)
$$

其中 $H(P, Q)$ 是模型分布 $Q$ 与真实分布 $P$ 之间的交叉熵。

### 与平均负对数似然的关系

$$
\text{Perplexity} = \exp\left(\text{Average NLL}\right)
$$

**Average NLL**（平均负对数似然）是语言模型训练的标准损失函数。

---

## 代码实现

### PyTorch 实现

```python
import torch
import torch.nn.functional as F
import math

def calculate_perplexity(logits, targets, mask=None):
    """
    计算困惑度
    
    Args:
        logits: [batch_size, seq_len, vocab_size] 模型输出
        targets: [batch_size, seq_len] 真实标签
        mask: [batch_size, seq_len] 有效位置掩码（可选）
    
    Returns:
        perplexity: 标量
    """
    # 计算 log softmax
    log_probs = F.log_softmax(logits, dim=-1)
    
    # 获取目标词的 log 概率
    batch_size, seq_len, vocab_size = log_probs.shape
    log_probs = log_probs.view(-1, vocab_size)
    targets = targets.view(-1)
    
    # 收集目标词的 log 概率
    nll_loss = F.nll_loss(log_probs, targets, reduction='none')
    nll_loss = nll_loss.view(batch_size, seq_len)
    
    # 应用 mask（忽略 padding）
    if mask is not None:
        nll_loss = nll_loss * mask
        total_tokens = mask.sum()
    else:
        total_tokens = batch_size * seq_len
    
    # 计算平均负对数似然
    avg_nll = nll_loss.sum() / total_tokens
    
    # 计算困惑度
    perplexity = torch.exp(avg_nll)
    
    return perplexity.item()


# 使用示例
batch_size, seq_len, vocab_size = 2, 10, 50000
logits = torch.randn(batch_size, seq_len, vocab_size)
targets = torch.randint(0, vocab_size, (batch_size, seq_len))

ppl = calculate_perplexity(logits, targets)
print(f"Perplexity: {ppl:.2f}")
```

### Hugging Face Transformers

```python
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

model = GPT2LMHeadModel.from_pretrained("gpt2")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

text = "Hello, how are you?"
inputs = tokenizer(text, return_tensors="pt")

with torch.no_grad():
    outputs = model(**inputs, labels=inputs["input_ids"])
    loss = outputs.loss
    perplexity = torch.exp(loss)

print(f"Perplexity: {perplexity.item():.2f}")
```

---

## 困惑度的意义

### 数值解读

| 困惑度 | 含义 | 模型水平 |
|-------|------|---------|
| **~100** | 每次预测面对100个等概率选择 | 较差（随机猜测级别）|
| **~30** | 每次预测面对30个等概率选择 | 一般 |
| **~10** | 每次预测面对10个等概率选择 | 较好 |
| **< 10** | 每次预测面对不到10个选择 | 优秀 |

### 不同任务的典型困惑度

| 任务/模型 | 困惑度 | 说明 |
|----------|-------|------|
| **GPT-2 (Small)** | ~20 | 1.24亿参数 |
| **GPT-2 (Large)** | ~15 | 7.74亿参数 |
| **GPT-3** | ~10 | 1750亿参数 |
| **人类水平** | ~5-8 | 估计值 |

---

## 困惑度的局限性

### 1. 与生成质量不完全正相关

- **低困惑度 ≠ 高质量生成**
- 模型可能过拟合训练数据，困惑度很低但泛化差
- 生成质量还涉及多样性、连贯性、事实准确性等

### 2. 对短文本敏感

- 短句子的困惑度波动大
- 更适合评估较长文本或整个测试集

### 3. 不考虑语义正确性

- 只衡量概率建模能力
- 不判断生成内容是否符合事实或逻辑

### 4. 不同词表无法直接比较

- 不同 tokenizer 的词表大小不同
- 词表越大，理论上困惑度可能越高
- 比较时应确保使用相同词表

---

## 面试要点

### 常见问题

**Q1: 什么是困惑度？**
- A: 衡量语言模型预测能力的指标，表示模型预测下一个词时的"困惑"程度。困惑度越低，模型性能越好。

**Q2: 困惑度的计算公式？**
- A: 
$$
\text{PPL} = \exp\left(-\frac{1}{N} \sum_{i=1}^N \log P(w_i \mid \text{context})\right)
$$

**Q3: 为什么用对数概率计算？**
- A: 避免数值下溢。概率连乘可能导致浮点数溢出为0，取对数后乘法变加法，数值更稳定。

**Q4: 困惑度与交叉熵的关系？**
- A: 困惑度是交叉熵的指数形式：$\text{PPL} = \exp(H(P, Q))$

**Q5: 困惑度越低越好吗？**
- A: 基本是的，但要注意：
  - 过低可能过拟合
  - 不同词表不能直接比较
  - 低困惑度不等于高生成质量

**Q6: 为什么 NLP 常用困惑度而不是准确率？**
- A: 
  - 语言生成是概率分布预测，不是单一类别判断
  - 困惑度能反映模型对整个概率分布的建模能力
  - 准确率只关心 top-1 预测，太粗糙

---

## 总结

```
困惑度 = exp(平均负对数似然)
       = exp(交叉熵)

直观意义: 模型每次预测下一个词时，面对多少等概率选择

越低越好: PPL < 10 优秀，PPL ~20 一般，PPL > 100 较差

计算步骤:
  1. 计算每个位置的条件概率 P(w_i|context)
  2. 取对数并求平均: avg_log_prob = (1/N) * sum(log P)
  3. 取指数: PPL = exp(-avg_log_prob)
```

**核心洞察**：困惑度是语言模型的"标准成绩"，但它只是评估指标之一，完整的模型评估还需要结合人工评测、下游任务表现等。

---

*参考：语言模型评估标准方法*
