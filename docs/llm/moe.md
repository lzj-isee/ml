# MoE (Mixture of Experts) 负载均衡

> Sequence 粒度的负载均衡损失计算方法

---

## 背景：MoE 负载均衡问题

在 MoE 模型中，每个 token 通过 Router 选择 $K$ 个专家进行处理。如果路由不均匀，会导致：
- 某些专家过载，成为瓶颈
- 某些专家闲置，浪费参数
- 训练不稳定，模型性能下降

---

## Sequence 粒度负载均衡损失

与 token 级负载均衡不同，**sequence 粒度**将一个完整序列作为统计单元，用序列内所有 token 的路由计数来做均衡。

### 符号定义

| 符号 | 含义 |
|-----|------|
| $N$ | 路由专家总数 |
| $T$ | 当前序列长度（token 数） |
| $K$ | 每个 token 激活的专家数（Top-K） |
| $x_t$ | 第 $t$ 个 token 的表示 |
| $g_{i,t} \in [0,1]$ | 第 $t$ 个 token 对专家 $i$ 的 gate 值（经 sigmoid，尚未归一化） |
| $\mathbb{1}_{i,t}$ | 指示函数：token $t$ 是否被路由到专家 $i$（Top-K 选中为 1，否则 0） |

### 序列级统计量

对**整个序列**做一次汇总：

#### 1. 专家 $i$ 被选中的总次数

$$
f_i = \sum_{t=1}^T \mathbf{1}_{i,t}
$$

**含义**：序列中所有 token 里，专家 $i$ 被选中多少次。

#### 2. 专家 $i$ 的累计 gate 值

$$
P_i = \sum_{t=1}^T g_{i,t}
$$

**含义**：专家 $i$ 的 gate 值在整序列上的累积。

---

## 负载均衡损失公式

### 目标

希望 $f_i$（专家 $i$ 的实际负载）尽可能接近**平均负载** $\frac{TK}{N}$。

**理想平均负载推导**：
- 序列共 $T$ 个 token
- 每个 token 选 $K$ 个专家
- 共 $N$ 个专家
- 平均每个专家应该处理 $\frac{T \cdot K}{N}$ 个 token

### 损失函数

$$
\mathcal{L}_\text{bal} = \alpha \sum_{i=1}^N \left( f_i - \frac{T \cdot K}{N} \right)^2
$$

### 公式解读

| 组件 | 含义 |
|-----|------|
| $f_i$ | 专家 $i$ 实际被选中的 token 次数 |
| $\frac{T \cdot K}{N}$ | 理想平均次数（均匀分布时的目标值） |
| $(\cdot)^2$ | 平方惩罚，偏离越大惩罚越重 |
| $\sum_{i=1}^N$ | 对所有专家的偏离求和 |
| $\alpha$ | 权重系数，通常很小（如 $10^{-2}$） |

---

## 关键特性

### 1. 最小值为 0

当所有专家负载完全相等时：

$$
f_1 = f_2 = \cdots = f_N = \frac{T \cdot K}{N}
$$

损失 $\mathcal{L}_\text{bal} = 0$

### 2. 仅训练阶段使用

| 阶段 | 是否使用 | 说明 |
|-----|---------|------|
| **训练** | ✅ 使用 | 加入总 loss，引导 Router 学习均衡路由 |
| **推理** | ❌ 移除 | 不影响生成质量，仅用于训练正则化 |

### 3. 与 Token-Drop 策略兼容

- 若丢弃部分 token，统计量只在**剩余 token** 上计算
- 公式不变，$T$ 为实际保留的 token 数

---

## 与 Token 级负载均衡对比

| 特性 | Token 级 | **Sequence 级** |
|-----|---------|----------------|
| **统计单元** | 单个 token | 整个序列 |
| **计算时机** | 每个 token 独立计算 | 序列结束时一次性计算 |
| **粒度** | 细粒度 | 粗粒度 |
| **稳定性** | 噪声较大 | 更平滑稳定 |
| **适用场景** | 短序列、在线学习 | 长序列、batch 训练 |

### Token 级损失示例

$$
\mathcal{L}_\text{bal}^\text{token} = \alpha \sum_{t=1}^T \sum_{i=1}^N \mathbf{1}_{i,t} \cdot \log(f_i)
$$

**问题**：单个 token 的路由具有随机性，统计不稳定。

### Sequence 级优势

- **更稳定**：整序列统计，随机性被平均
- **更符合实际**：GPU 并行通常以 sequence/batch 为单位
- **易于实现**：只需在序列末尾计算一次

---

## 代码实现

```python
import torch
import torch.nn.functional as F

def sequence_load_balance_loss(router_probs, expert_indices, alpha=1e-2):
    """
    计算 sequence 粒度的负载均衡损失
    
    Args:
        router_probs: [T, N] 每个 token 对每个专家的 gate 值（sigmoid后）
        expert_indices: [T, K] 每个 token 选中的 K 个专家索引
        alpha: 损失权重系数
    
    Returns:
        loss: 标量负载均衡损失
    """
    T, N = router_probs.shape
    K = expert_indices.shape[1]
    
    # 1. 计算每个专家被选中的次数 f_i
    # 创建 one-hot 指示矩阵 [T, N]
    expert_mask = torch.zeros(T, N, device=router_probs.device)
    for k in range(K):
        expert_mask.scatter_(1, expert_indices[:, k:k+1], 1)
    
    # f_i = sum over T of 1_{i,t}
    f = expert_mask.sum(dim=0)  # [N]
    
    # 2. 计算理想平均负载
    avg_load = (T * K) / N
    
    # 3. 计算负载均衡损失
    loss = alpha * ((f - avg_load) ** 2).sum()
    
    return loss


# 示例
T, N, K = 512, 8, 2  # 512 tokens, 8个专家, Top-2
router_probs = torch.sigmoid(torch.randn(T, N))  # gate 值
_, expert_indices = torch.topk(router_probs, K, dim=-1)  # Top-K 选择

loss = sequence_load_balance_loss(router_probs, expert_indices, alpha=1e-2)
print(f"Load balance loss: {loss.item():.4f}")
```

---

## 超参数调优

| 参数 | 典型值 | 说明 |
|-----|-------|------|
| $\alpha$ | $10^{-2}$ ~ $10^{-3}$ | 权重太小：均衡效果弱；太大：影响主任务学习 |
| $K$ | 1 ~ 4 | 通常 Top-1 或 Top-2，越大负载均衡越难 |

### 调参建议

1. **先固定 $\alpha = 10^{-2}$**，观察专家负载分布
2. 若负载仍不均衡，**适当增大** $\alpha$
3. 若主任务性能下降明显，**适当减小** $\alpha$

---

## 面试要点

### 常见问题

**Q1: 什么是 sequence 粒度的负载均衡？**
- A: 将整个序列作为统计单元，计算序列内每个专家被选中的总次数 $f_i$，使其接近理想平均 $\frac{TK}{N}$

**Q2: 公式中的 $\frac{TK}{N}$ 是怎么来的？**
- A: 序列共 $T$ 个 token，每个选 $K$ 个专家，总共 $TK$ 次选择，平均分给 $N$ 个专家

**Q3: 为什么用平方损失？**
- A: 平方惩罚对偏离更敏感，且数学性质好（可导、凸函数）

**Q4: 推理阶段为什么去掉负载均衡损失？**
- A: 负载均衡是**训练正则化项**，用于引导 Router 学习；推理时只需要训练好的 Router，不需要额外约束

**Q5: 与 token-drop 如何兼容？**
- A: 丢弃 token 后，只在剩余 token 上计算 $f_i$ 和损失，公式不变

**Q6: Sequence 级 vs Token 级的优缺点？**
- A:
  - Sequence 级：更稳定、噪声小、适合 batch 训练
  - Token 级：粒度细、响应快，但噪声大

---

## 总结

```
MoE 负载均衡问题
    ↓
Sequence 级解决方案
    ↓
统计量:
  f_i = sum_t(1_{i,t})      # 专家i被选中的次数
  avg = T*K/N               # 理想平均负载
    ↓
损失函数:
  L_bal = α * sum_i((f_i - avg)^2)
    ↓
特性:
  - 完全均衡时损失为0
  - 仅训练阶段使用
  - 与token-drop兼容
```

**一句话**：Sequence 级负载均衡损失就是统计一整条序列里每个专家被激活的 token 次数与理想平均值的平方差之和，用一个标量引导 MoE 负载尽量均衡。

---

*参考：Switch Transformer, DeepSeek-MoE 等 MoE 架构论文*
