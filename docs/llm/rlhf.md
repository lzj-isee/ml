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

### GRPO 的劣势与问题

#### 1. Token-level 监督信号缺失

PPO 通过 GAE 为**每个 token**计算不同的 advantage，模型能知道序列中哪个位置贡献好/坏。

GRPO 的 advantage 是 **sequence-level**，同一个序列所有 token 共享相同 advantage：

```python
# PPO: 每个 token 知道自己在序列中的具体贡献
advantages = [0.1, -0.2, 0.5, 0.3]  # token 0贡献+0.1, token 1贡献-0.2...

# GRPO: 所有 token 共享同一个"平均分"
advantage = 0.67  # 不知道哪个 token 好，只知道这个 response 整体好
```

**后果**：
- **Reward sparsity**：模型无法定位 response 中具体哪个 token 导致了低分
- 训练信号模糊，不如 PPO 精细

#### 2. Group Variance / Collapse 问题

GRPO 依赖 group 内的 reward 分布计算 advantage：

$$
A_i = \frac{r_i - \text{mean}(\{r\})}{\text{std}(\{r\})}
$$

**问题场景**：

```python
# 如果 group 内所有 sample 的 reward 相同
r_group = [1.0, 1.0, 1.0, 1.0]  # 或全为 0
# std = 0 → 除以零或数值不稳定 → advantage 无法计算
```

**实际训练中的演变**：
- **训练初期**：policy 还没学会，采样多样性高，group variance 正常
- **训练后期**：policy 收敛，多数 sample 趋于相似，group variance 下降 → **更新信号减弱甚至消失**

#### 3. Sample Efficiency 问题

GRPO 必须**生成整个 group** 后才能计算 advantage：

| 方法 | 更新时机 |
|------|----------|
| PPO | Critic 预估 value，采样完立即计算 advantage 更新 |
| GRPO | 必须等 group 内所有 G 个 response 都生成并打分后才能计算相对优势 |

**后果**：
- 延迟更新，样本利用率相对较低
- 如果 group 内 reward 分布方差大，baseline 估计不准

#### 4. 缓解策略（工程实践）

| 方法 | 思路 |
|------|------|
| **Temperature sampling** | 提高采样 temperature，强制增加输出多样性，防止 collapse |
| **KL penalty** | 添加 KL divergence 约束，防止 policy 坍缩到单一模式 |
| **Dynamic group size** | 根据 variance 动态调整 group size（variance 低时增大 G）|
| **Reward shaping** | 引入过程奖励或长度惩罚，打破 binary reward，增加信号丰富度 |
| **Repetition penalty** | DeepSeek 实际采用，防止模型生成重复内容导致 group 同质化 |

---

### PPO vs GRPO 对比总结

| 维度 | PPO | GRPO |
|------|-----|------|
| **Advantage 来源** | Critic 网络估算的 value baseline | Group 内采样输出的相对奖励均值 |
| **Reward 粒度** | Token-level (GAE) | Sequence-level |
| **Critic 依赖** | 需要单独训练 | **不需要**，节省显存 |
| **显存占用** | 高（Policy + Critic + Reward Model）| 低（Policy + Reward Model）|
| **Sample efficiency** | 高，采样完立即更新 | 低，需等整个 group 生成 |
| **更新稳定性** | Critic 估计误差，但不受 group variance 影响 | Group variance 敏感，可能 collapse |
| **适用场景** | 有明确 token-level reward（tool use、code exec 反馈）| Outcome-only reward（数学答案对错）|

---

### 为什么 GRPO 训练开始时 Loss 为 0？

这是一个常见的观察现象，下面从公式推导解释原因。

#### GRPO 损失公式

$$
\mathcal{L}_{\text{GRPO}}(\theta) = -\frac{1}{G} \sum_{i=1}^{G} \frac{1}{|o_i|} \sum_{t=1}^{|o_i|} \left[ \underbrace{\text{clip-比率} \cdot \hat{A}_{i,t}}_{\text{优势部分}} - \underbrace{\text{KL散度}}_{\text{正则部分}} \right]
$$

#### 训练启动时的特殊状态

**关键条件**：训练开始时，待更新参数 $\pi_\theta$ **未发生更新**，与历史参数 $\pi_{\theta^{\text{old}}}$ **完全一致**。

因此：**比率 = 1**

$$
\frac{\pi_\theta(o_{i,t} \mid q, o_{i,<t})}{\pi_{\theta^{\text{old}}}(o_{i,t} \mid q, o_{i,<t})} = 1
$$

#### 1. 优势部分为什么为 0？

GRPO 中优势值在**组内标准化**：

$$
\hat{A}_i = \frac{r_i - \text{mean}(\{r_1, r_2, \cdots, r_G\})}{\text{std}(\{r_1, r_2, \cdots, r_G\})}
$$

**关键性质**：标准化后的优势值在组内**均值为 0**。

因此当参数比率为 1 时：

$$
\frac{1}{G} \sum_{i=1}^{G} \frac{1}{|o_i|} \sum_{t=1}^{|o_i|} \hat{A}_{i,t} = 0
$$

**→ 优势部分 Loss = 0**

#### 2. KL 散度部分为什么为 0？

GRPO 采用 **K3 估计** 减少计算量（而非标准 KL）：

$$
\text{KL}_{\text{K3}} = \frac{\pi_{\text{ref}}}{\pi_\theta} - \log\frac{\pi_{\text{ref}}}{\pi_\theta} - 1
$$

训练开始时，$\pi_{\text{ref}}$ 与 $\pi_\theta$ 完全相同：

$$
\frac{\pi_{\text{ref}}}{\pi_\theta} = 1
$$

代入：

$$
\text{KL}_{\text{K3}} = 1 - \log(1) - 1 = 0
$$

**→ KL 散度部分 Loss = 0**

#### 总结

| 部分 | 训练开始时 | 原因 |
|-----|-----------|------|
| **优势部分** | 0 | 组内标准化优势值均值为 0 |
| **KL 散度部分** | 0 | $\pi_\theta = \pi_{\text{ref}}$，比率为 1 |
| **总 Loss** | **0** | 两部分均为 0 |

#### 重要说明：与 Loss 计算方式的关系

用户指出：这与 loss 的具体计算方式有关。

- **标准 GRPO**：先在 seq 内部算平均，再所有 seq 算平均 → **开始时为 0**
- **全局 token 平均**：如果是按照全局 token 数直接归一化（如 DAPO 的 Token-Level Loss），开始时**可能不为 0**

DAPO 提出的 **Token-Level Loss** 改进正是针对此问题，将归一化方式从 $\frac{1}{G} \cdot \frac{1}{|o_i|}$ 改为 $\frac{1}{\sum |o_i|}$，防止长序列梯度被稀释。

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

## DAPO：GRPO 的改进版

DAPO (Dynamic Sampling Policy Optimization) 针对 GRPO 实际训练中的问题，提出了六项改进。

### 核心动机

GRPO 训练中存在三个主要问题：
1. **clip 范围设置不合理** —— 低概率关键 token 被抑制
2. **采样冗余** —— 得分相同的样本产生 0 优势，浪费梯度
3. **长序列梯度被稀释** —— 长回答的 token 权重过低

### 六项改进

| 改进 | 问题 | 解决方案 |
|------|------|----------|
| **Clip-Higher** | clip 上界过小，低概率关键 token 涨幅受限 | 提高上界 $\epsilon_{\text{high}}$，释放上涨空间 |
| **Dynamic Sampling** | 采样结果得分相同 → 优势为 0 → 梯度浪费 | 约束条件：$0 < \|\{o_i \mid \text{is_equivalent}(a, o_i)\}\| < G$，保证得分多样性 |
| **Token-Level Loss** | GRPO 先对 sample 内 token 平均，再对 batch 平均，长回答 token 权重被稀释 | 改为全局按总 token 数归一化：$\frac{1}{\sum|o_i|}$ 替代 $\frac{1}{G} \cdot \frac{1}{|o_i|}$ |
| **Soft Punishment** | 回答过长问题 | 双阈值：超阈值线性惩罚，突破第二阈值则奖励归零 |
| **移除 KL 散度** | 长文本推理模型分布会显著偏离初始模型，KL 约束不再必要 | **直接删除 KL 项** |
| **规则奖励** | Reward hacking：模型利用奖励模型漏洞 | 编程/数学题用规则判断对错：$R = 1$ (正确) / $-1$ (错误) |

### 关键结论

> **DAPO 在长文本推理场景中移除了 KL 散度约束**
> 
> 传统 RLHF/PPO 使用 KL 惩罚防止策略偏离参考模型太远，但实验发现长文本推理模型的分布**天然会显著偏离**初始模型，此时 KL 约束反而成为负担，因此 DAPO 选择直接移除。

---

*（待补充：PPO Loss 完整构成、KL Penalty、DPO 代码实现等）*
