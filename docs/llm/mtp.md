# MTP (Multi-Token Prediction) 多Token预测

## 概述

MTP 是一种在大语言模型训练和推理阶段同时生成多个后续Token的技术，旨在解决传统"Next Token Prediction"（NTP）的两大痛点：

1. **推理速度瓶颈**：NTP生成过程是串行的，生成N个Token需要跑N次完整的前向传播，受限于显存带宽
2. **缺乏全局规划**：模型只关注当下token，往往导致局部最优，缺乏对序列的"大局观"

通过解码阶段的优化，将1-token的生成转变为multi-token的生成，MTP在**训练阶段**一次学习多个位置的label提升样本利用效率，在**推理阶段**实现成倍的推理加速。

---

## 顺序型 vs 并行型 MTP

MTP可分为**顺序型MTP**和**并行型MTP**两种实现方式，其中**DeepSeek V3实际采用的是顺序型MTP**。

| 特性 | 顺序型 MTP | 并行型 MTP |
|------|-----------|-----------|
| **架构** | 模块串行连接，第k个模块依赖第k-1个模块的输出 | 独立预测头，所有token基于同一hidden state并行预测 |
| **因果性** | ✅ 保持因果链，预测$t_{i+k}$时利用了$t_{i+k-1}$的信息 | ❌ 打破因果链，预测$t_{i+k}$时未利用中间token信息 |
| **预测质量** | 高，逻辑连贯 | 较低，可能产生不合理序列 |
| **推理速度** | 较快（轻量级模块串联） | 最快（单次投影得到N个token） |
| **典型应用** | DeepSeek-V3, Qwen3-Next | 实验性研究 |

### 顺序型 MTP

#### 训练时 (Teacher Forcing)

```
主模型处理 t_i → 得到隐藏状态 h_i^0
h_i^0 预测 t_{i+1}

MTP模块1: h_i^0 + E(t_{i+1}) 预测 t_{i+2}
MTP模块2: h_i^1 + E(t_{i+2}) 预测 t_{i+3}
...
```

**关键点**：训练时使用**真实的token嵌入** $E(t_{i+k})$，而非模型自身的预测，保证良好的训练信号。

#### 推理时 (Autoregressive)

```
主模型处理 t_i → 得到 h_i^0 → 预测 t_{i+1}
MTP模块1: h_i^0 + E(t_{i+1}) 预测 t_{i+2}
MTP模块2: h_i^1 + E(t_{i+2}) 预测 t_{i+3}
...
```

**本质**：顺序型MTP是"自回归的"，但在预测$t_{i+k}$时**不需要**通过大计算量的Transformer Block计算$t_{i+k-1}$，而是利用轻量级MTP模块快速生成，从而实现加速。

#### 架构特点

```
输入序列: [t_1, t_2, ..., t_i]
         ↓
    [主模型 Main Model] → h_i^0 ──────→ 预测 t_{i+1}
         ↓                              ↓
    [MTP Module 1]  → h_i^1 ──────→ 预测 t_{i+2}
         ↓                              ↓
    [MTP Module 2]  → h_i^2 ──────→ 预测 t_{i+3}
         ↓
        ...
```

- **主模型**：完整Transformer，计算量占比>95%
- **MTP模块**：轻量级Transformer层（通常1层），计算量占比<5%
- **共享输出头**：MTP模块与主模型共享词表投影层

### 并行型 MTP

所有N个预测 $\hat{t}_{i+1}, \hat{t}_{i+2}, ..., \hat{t}_{i+N}$ 都是**独立地**基于$h_i^0$生成的。

```python
# 核心实现：一次线性投影得到N个token的logits
mtp_projection = nn.Linear(D, N * V)  # D: hidden_dim, V: vocab_size
mtp_logits = mtp_projection(hidden_state).view(N, V)  # (N, V)
```

**注意点**：
- 预测$\hat{t}_{i+k}$时**没有利用**$\hat{t}_{i+1}, ..., \hat{t}_{i+k-1}$的信息
- 速度快，但可能生成语义不连贯的序列（如"我 苹果 吃"）

---

## 为何MTP适合做推测解码

MTP最迷人的地方在于它如何解决"推理速度瓶颈"——它实现了一种**自推测解码（Self-Speculative Decoding）**。

### 传统推测解码的痛点

传统推测解码需要：
- 一个额外的**草稿模型（Draft Model）**快速生成候选序列
- 一个大模型作为**目标模型（Target Model）**验证

**问题**：部署复杂度高，需要维护两个模型。

### MTP作为"天然草稿模型"

DeepSeek-V3的MTP模块本身就是完美的Draft Model：

| 优势 | 说明 |
|------|------|
| **无需额外部署** | MTP模块与主模型一体化训练，无需单独的草稿模型 |
| **计算开销极小** | MTP模块非常轻量（通常只有1层），额外计算占比<5% |
| **质量更高** | 顺序MTP保持因果链，生成的草稿质量优于简单小模型 |
| **加速效果显著** | MTP猜对时推理速度直接翻倍 |

### 推测解码流程

```
Step 1: 主模型生成 t_{i+1}
Step 2: MTP模块快速生成草稿 [t_{i+2}, t_{i+3}, ..., t_{i+N}]
Step 3: 下一次主模型前向，同时验证草稿token
        ✅ 接受：继续使用MTP生成下一步草稿
        ❌ 拒绝：从第一个错误位置重新开始
```

**关键洞察**：
- 主模型计算量占95%+，只运行1次产生$h_i^0$
- MTP扩展计算量<5%，就像主模型跑完后瞬间接了一个"小尾巴"
- 相对传统方法（生成每个token都跑主模型），MTP方式：**1次主模型 + N次轻量MTP = 得到N+1个Token**

---

## 代码实现

### 顺序型 MTP

```python
import torch
import torch.nn as nn

class MTPModule(nn.Module):
    """
    DeepSeek-V3风格的顺序MTP模块。
    输入：前一个隐藏状态 + 下一个token的嵌入
    输出：更新后的隐藏状态 + 预测的logits
    """
    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        # 投影层：融合隐藏状态和token嵌入
        self.projection = nn.Linear(d_model, d_model)
        # 轻量级Transformer块
        self.transformer_block = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=8,
            dim_feedforward=d_model * 4,
            batch_first=True
        )
        # 共享输出头
        self.output_head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, hidden_state: torch.Tensor, next_token_emb: torch.Tensor
                ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            hidden_state: [batch, d_model] 来自主模型或上一个MTP模块
            next_token_emb: [batch, d_model] 下一个token的嵌入
        Returns:
            updated_hidden: [batch, d_model]
            logits: [batch, vocab_size]
        """
        # 融合信息
        combined = hidden_state + next_token_emb
        projected = self.projection(combined)
        # 轻量级变换
        updated = self.transformer_block(projected.unsqueeze(0)).squeeze(0)
        # 预测
        logits = self.output_head(updated)
        return updated, logits


class SequentialMTPModel(nn.Module):
    """顺序MTP完整模型示例"""
    def __init__(self, d_model: int, vocab_size: int, n_mtp: int):
        super().__init__()
        self.n_mtp = n_mtp
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.main_output_head = nn.Linear(d_model, vocab_size, bias=False)
        self.mtp_modules = nn.ModuleList([
            MTPModule(d_model, vocab_size) for _ in range(n_mtp)
        ])

    @torch.no_grad()
    def generate_with_mtp(self, main_hidden: torch.Tensor, max_tokens: int) -> list[int]:
        """
        顺序MTP推理加速
        Args:
            main_hidden: 主模型输出的隐藏状态 [batch, d_model]
        Returns:
            drafted_tokens: 预测的token ID列表
        """
        drafted = []
        current_hidden = main_hidden.clone()

        # 主模型预测第一个token
        first_logits = self.main_output_head(current_hidden)
        first_token = torch.argmax(first_logits, dim=-1).item()
        drafted.append(first_token)

        # 顺序执行MTP模块
        for k in range(min(max_tokens - 1, self.n_mtp)):
            # 获取上一步预测token的嵌入
            token_emb = self.embedding(torch.tensor([drafted[-1]], device=main_hidden.device))
            # MTP前向
            current_hidden, logits = self.mtp_modules[k](current_hidden, token_emb)
            next_token = torch.argmax(logits, dim=-1).item()
            drafted.append(next_token)

        return drafted
```

### 并行型 MTP

```python
class ParallelMTPHead(nn.Module):
    """
    并行MTP预测头。
    一次投影，同时预测N个未来token。
    """
    def __init__(self, hidden_size: int, vocab_size: int, num_predict: int):
        super().__init__()
        self.num_predict = num_predict
        self.vocab_size = vocab_size
        # 核心：一个线性层投影到 N * V
        self.mtp_projection = nn.Linear(
            hidden_size,
            num_predict * vocab_size,
            bias=False
        )

    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden_state: [batch, hidden_size]
        Returns:
            logits: [batch, num_predict, vocab_size]
        """
        batch_size = hidden_state.size(0)
        raw_logits = self.mtp_projection(hidden_state)  # [batch, N*V]
        logits = raw_logits.view(batch_size, self.num_predict, self.vocab_size)
        return logits
```

---

## 训练阶段：更丰富的监督信号

MTP的引入为模型带来了更丰富的训练信号：

### 损失函数

$$
\mathcal{L}_{total} = \mathcal{L}_{main} + \sum_{k=1}^{N} \lambda_k \cdot \mathcal{L}_{MTP_k}
$$

### Teacher Forcing机制

```python
# 主模型输出
hidden = main_model(tokens)  # [batch, seq_len, hidden_dim]
main_logits = output_head(hidden)

# MTP模块训练（使用真实token嵌入）
mtp_losses = []
prev_hidden = hidden[:, -1, :]  # 最后一个位置的隐藏状态
for k in range(N_MTP):
    # 使用真实token的嵌入，而非模型预测
    true_token_emb = embedding(tokens[:, -1 + k + 1])
    prev_hidden, logits = mtp_modules[k](prev_hidden, true_token_emb)
    mtp_losses.append(cross_entropy(logits, targets[:, -1 + k + 2]))
```

**关键点**：训练时使用**Ground Truth**的token嵌入，强迫主模型输出的隐向量$h_i^0$必须蕴含足够丰富的信息，不仅要能推导出$t_{i+1}$，还要为推导后续token提供坚实基础。

---

## 总结

DeepSeek-V3的MTP策略实现了"一石二鸟"：

| 维度 | 收益 |
|------|------|
| **训练** | 多步预测目标倒逼模型学习更具全局观的特征表示，提升表征能力和规划能力 |
| **推理** | 无需额外草稿模型，内建推测解码，实现端到端的加速优化 |

这种从训练目标到推理加速的端到端设计，为大模型架构优化提供了优雅范例，也被后续开源模型（如Qwen3-Next）所采用。

---

## 参考

- DeepSeek-V3 Technical Report
- Qwen3 Technical Report
- [DeepSeek技术解读(2)-MTP（Multi-Token Prediction）的前世今生](https://zhuanlan.zhihu.com/p/1975333103573677711)
- [DeepSeek MTP 论文解析](https://zhuanlan.zhihu.com/p/1997072284737897370)

---

*标签: `#MTP` `#Multi-Token-Prediction` `#推测解码` `#DeepSeek` `#推理加速`*
