# RLHF 与 PPO

RLHF (Reinforcement Learning from Human Feedback) 相关知识点，涵盖 PPO、DPO、GRPO 等算法。

---

## 目录

- [PPO Advantage 计算](#ppo-advantage-计算)
- [PPO vs GRPO 的 Advantage 区别](#ppo-vs-grpo-的-advantage-区别)
- [DPO (Direct Preference Optimization)](#dpo-direct-preference-optimization)

---

## PPO Advantage 计算

PPO 训练大模型时，**Advantage 通常使用 GAE (Generalized Advantage Estimation)** 计算。

### GAE 公式

$$
\hat{A}_t = \sum_{l=0}^{\infty} (\gamma \lambda)^l \delta_{t+l}
$$

其中 **TD-error** $\delta_t$：
$$
\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)
$$

### 代码实现

```python
def compute_gae(rewards, values, next_values, dones, gamma=0.99, lam=0.95):
    """
    rewards: [T] 每个token的reward
    values: [T] Critic模型预测的当前状态价值
    next_values: [T] 下一个状态的价值
    dones: [T] 是否结束
    """
    advantages = []
    gae = 0
    
    # 从后向前计算
    for t in reversed(range(len(rewards))):
        if dones[t]:
            delta = rewards[t] - values[t]
            gae = delta
        else:
            delta = rewards[t] + gamma * next_values[t] - values[t]
            gae = delta + gamma * lam * gae
            
        advantages.insert(0, gae)
    
    return advantages  # [T]
```

### RLHF 流程

1. **Reward Model 打分**: 对生成的 token/序列计算 reward $r_t$
2. **Critic 预测价值**: value network 输出 $V(s_t)$
3. **计算 Advantage**: GAE 得到每个位置的优势值
4. **PPO 更新**: 用 Advantage 计算策略梯度 + value loss 更新 Critic

### 关键参数

| 参数 | 作用 | 典型值 |
|------|------|--------|
| $\gamma$ | 折扣因子，平衡短期/长期回报 | 0.99 |
| $\lambda$ | GAE权衡参数，$\lambda=0$ 退化为TD，$\lambda=1$ 为蒙特卡洛 | 0.95 |

### 为什么用 GAE？

- **降低方差**: 相比纯蒙特卡洛 (MC) 回报，GAE 方差更小
- **控制偏差**: 通过 $\lambda$ 在偏差和方差之间 trade-off
- **适合长序列**: LLM 生成序列长，MC 方差太大

---

## PPO vs GRPO 的 Advantage 区别

| 特性 | PPO | GRPO |
|------|-----|------|
| **Advantage 粒度** | **Token-level**：每个位置的 advantage 不同 | **Sequence-level**：同一个序列所有 token 共享相同 advantage |
| **计算方式** | GAE 累积 TD-error | Group 内相对奖励归一化 |
| **Critic 依赖** | 需要单独的 Critic 模型预测 $V(s)$ | **不需要 Critic**，用 group 平均作为 baseline |

### PPO: Token-level Advantage

```python
# PPO: 每个 token 有不同的 advantage
advantages = [0.1, 0.3, 0.5, -0.2, 0.0]  # 5个token，5个不同的adv
```

GAE 从后向前累积，每个位置的 $\delta_t$ 不同，导致同一个序列内不同 token 的 advantage 不同。

### GRPO: Sequence-level Advantage

```python
# GRPO: 同一个序列所有 token 共享相同 advantage
group_rewards = [0.8, 0.6, 0.9, 0.5]  # 4个response的reward
mean_reward = 0.7
std_reward = 0.15

# 每个 response 的 advantage（该序列内所有token相同）
adv_0 = (0.8 - 0.7) / 0.15 = 0.67   # response 0 的所有token都是0.67
adv_1 = (0.6 - 0.7) / 0.15 = -0.67  # response 1 的所有token都是-0.67
```

**GRPO 核心公式**：
$$A_i = \frac{r_i - \text{mean}(\{r\})}{\text{std}(\{r\})}$$

这个 sequence-level 的 advantage 会**广播到该序列的每一个 token**上。

### 为什么 GRPO 这样设计？

1. **省去 Critic**：不需要训练单独的 value network，减少内存和计算
2. **相对奖励更稳定**：用 group 内的相对排名，而不是绝对 reward 值
3. **适合推理任务**：对于数学、代码等答案明确的任务，group 内对比更直观

---

## DPO (Direct Preference Optimization)

DPO 是 RLHF 的替代方案，**直接把偏好数据用于优化策略，绕过显式奖励模型和强化学习**。

### DPO 要解决的问题

RLHF 传统流程：
1. 训练奖励模型 (Reward Model)
2. 用 PPO 强化学习优化策略

**问题**：PPO 训练不稳定，奖励模型和策略分离，流程复杂。

**DPO 核心创新**：直接用偏好数据优化策略，无需奖励模型和 RL。

---

### 偏好建模：Bradley-Terry 模型

将人类偏好比较转化为概率：

$$
P(A \succ B) = \frac{\exp(r(A))}{\exp(r(A)) + \exp(r(B))} = \sigma(r(A) - r(B))
$$

**关键直觉**：
- 胜负概率只取决于**奖励差**
- sigmoid 把差值压缩到 [0,1] 区间
- 差值越大，获胜概率越接近 1；差值为 0，概率 0.5

**训练损失**（二元交叉熵）：
$$
\mathcal{L} = -\log(\sigma(r(A) - r(B)))
$$

---

### RLHF 优化目标与解析解

RLHF 目标：最大化期望奖励 + KL 约束

$$
\max_{\pi} \mathbb{E}_{x \sim \pi}[r(x)] - \beta \cdot \text{KL}(\pi \|\| \pi_{\text{ref}})
$$

**β 的作用**：
- β 大 → 强约束，模型保守，不偏离 π_ref
- β 小 → 弱约束，激进拟合偏好，易过拟合

**变分法解析解**：
$$
\pi^*(x) = \frac{1}{Z} \cdot \pi_{\text{ref}}(x) \cdot \exp\left(\frac{r(x)}{\beta}\right)
$$

**直觉**：最优策略 = 参考策略 × 奖励的指数放大因子

---

#### 变分法推导的关键理解

**为什么可以"逐点"求解？**

把分布 $\pi$ 看作**无穷维函数**（变分法的核心视角）：

| 有限维 | 无穷维（函数空间） |
|--------|-------------------|
| 变量 $\mathbf{x} \in \mathbb{R}^n$ | 函数 $\pi \in \mathcal{F}$ |
| 梯度 $\nabla f$ | 变分导数 $\frac{\delta \mathcal{L}}{\delta \pi(x)}$ |
| 条件：$\frac{\partial f}{\partial x_i} = 0, \forall i$ | 条件：$\frac{\delta \mathcal{L}}{\delta \pi(x)} = 0, \forall x$ |

**关键**：目标函数是**可分的**（Separable）——每个 $\pi(x)$ 只影响该位置的贡献，不同 $x$ 之间无耦合。

因此优化问题 = **无穷多个独立的子问题**（每个 $x$ 一个），必须各自满足梯度为 0。

---

### DPO 关键推导：用策略表示奖励

从解析解变形，提取奖励：

$$
r(x) = \beta \cdot \log\left(\frac{\pi^*(x)}{\pi_{\text{ref}}(x)}\right) + \beta \cdot \log Z
$$

对同一问题的两个回答 A、B，常数 Z 抵消：

$$
r(A) - r(B) = \beta \cdot \left[ \log\frac{\pi^*(A)}{\pi_{\text{ref}}(A)} - \log\frac{\pi^*(B)}{\pi_{\text{ref}}(B)} \right]
$$

**核心含义**：**奖励差完全等价于策略对数概率比的差**。不需要单独训练奖励模型，策略网络本身的对数概率就隐含了奖励信息。

---

### DPO 损失函数

代入 Bradley-Terry 模型的 sigmoid：

$$
P(A \succ B) = \sigma\left( \beta \cdot \left[\log\frac{\pi(A)}{\pi_{\text{ref}}(A)} - \log\frac{\pi(B)}{\pi_{\text{ref}}(B)}\right] \right)
$$

**DPO 损失函数**：

$$
\mathcal{L}_{\text{DPO}} = -\log \sigma\left( \beta \cdot \left[\log\frac{\pi(A)}{\pi_{\text{ref}}(A)} - \log\frac{\pi(B)}{\pi_{\text{ref}}(B)}\right] \right)
$$

---

### 直观理解

| 组件 | 作用 |
|------|------|
| **对数比** | 衡量模型让胜者概率相比 π_ref 涨了多少，败者跌了多少 |
| **β 缩放** | 控制偏好信号的放大/缩小程度 |
| **sigmoid** | 把奖励差转为"胜者更好的概率" |
| **-log** | 预测概率低则惩罚大，逼迫模型拉大胜败者的奖励差距 |

**一句话**：DPO 逼迫模型提高对胜者的相对概率，同时用 π_ref 做锚点防止跑偏。

---

### 超参数 β 调参指南

| β 值 | 效果 | 风险 |
|------|------|------|
| **太小 (<0.1)** | 激进拟合偏好 | 过拟合、模式崩溃、遗忘通用能力 |
| **太大 (>0.5)** | 过于保守 | 优化不足，和原始模型几乎无差别 |
| **适中 (~0.1)** | 平衡拟合与保留 | 常用默认值 |

---

### DPO 优势总结

1. **无需显式奖励模型**：直接复用策略网络本身
2. **无需强化学习**：稳定、高效的监督式训练
3. **理论优雅**：奖励-策略的等价关系给出闭式解
4. **实现简单**：只需 log-prob 计算和参考模型对比

---

*（待补充：PPO Loss 完整构成、KL Penalty、DPO 代码实现等）*
