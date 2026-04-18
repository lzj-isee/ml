# LLM 面试八股整理

> 整理自牛客网等来源，持续更新中...

---

## 一、Transformer 相关

### 1. Transformer 的自注意力机制及相比 RNN 的优势？

**优势：**

1. RNN 难以有效捕捉长距离依赖；自注意力机制直接建模所有 token 之间的关系，无论距离远近，轻松捕捉全局依赖
2. RNN 上下文信息逐时间步累积，前向传播中较早的信息可能丢失；Transformer 每个 token 都能直接访问整个序列的上下文，信息保留更完整
3. RNN 固定内存占用；Transformer 为二阶复杂度，超长 context 占用大量显存

### 2. 自注意力计算过程？Softmax 为什么要缩放？

**计算过程：**

1. **线性投影**：输入 X 通过三个线性层得到 Q, K, V
2. **计算注意力分数**：`Scores = Q @ K^T`
3. **缩放**：`Scores = Scores / sqrt(d_k)`
4. **Masking（可选）**：对不允许关注的位置设为 -inf
5. **Softmax**：`Attention_Weights = Softmax(Scores)`
6. **加权求和**：`Output = Attention_Weights @ V`

**为什么要缩放？**

Q 和 K 的元素通常均值为 0，方差为 1。点积 `q @ k` 是 `d_k` 个独立变量的和，方差为 `d_k`，标准差为 `sqrt(d_k)`。

不缩放会导致：
- **梯度消失**：点积过大时，Softmax 输出接近 one-hot，梯度接近 0
- **数值不稳定**：exp 大数可能溢出

缩放后方差稳定在 1 左右，保证训练稳定。

### 3. 为什么 Transformer 需要多头注意力机制？

Attention 是一种动态的线性加权计算。单头情况下，每次 attention 只能关注一个方面。

保持 dim 不变而增加 head 数量，增加了模型的表示能力，使一个 attention 层能从不同方面处理隐空间向量。

### 4. 为什么 Transformer 需要位置编码？

Transformer 处理序列数据时没有时序计算，token 之间是并行处理的，需要额外融入 token 之间的位置信息。

---

## 二、位置编码

### 1. RoPE（Rotary Position Embedding）

以绝对位置编码的计算方式实现了相对位置编码。要求 QK 的 dim 必须为偶数，两个一组做 theta 旋转。向量旋转后模长不变，求内积的结果受旋转角度影响，体现了相对位置编码信息。

参考：https://zhuanlan.zhihu.com/p/662790439

### 2. 位置编码总览

参考：https://zhuanlan.zhihu.com/p/642846676

### 3. NTK-aware Interpolation

参考：https://zhuanlan.zhihu.com/p/662790439

### 4. YaRN（Yet another RoPE extension）

参考：https://zhuanlan.zhihu.com/p/15311461897

---

## 三、注意力机制变体

### 1. GQA 原理

Hidden-state 分成两路：
- 一路分为 `head_num * head_dim` 的 Q 向量
- 另一路分为 `kv_num * head_dim` 的 KV 向量

例如 Qwen2-0.5B：hidden-state=896，head-num=14，head-dim=64，kv-num=2，则 7 个 Q 共享 1 个 KV。

### 2. MHA / GQA / MQA 区别？

TODO

### 3. Cross Attention 和 Self Attention 区别？

TODO

### 4. Prefix LM 与 Causal LM 区别？

- **Causal LM**：完全单向，第 k 个 token 无法感知 k+1 个 token，attention 是下三角矩阵
- **Prefix LM**：分成两部分，prefix 部分可双向感知，generation 部分只能向前感知，attention 矩阵分两块（完全矩阵 + 下三角）

### 5. GQA的好处是什么？



---

## 四、模型架构

### 1. BERT 和 GPT 的区别？

| | BERT | GPT |
|---|---|---|
| 架构 | Encoder-only | Decoder-only |
| 注意力 | 双向 | 单向（因果） |
| 预训练任务 | MLM + NSP | CLM（预测下一个 token） |
| 擅长任务 | 理解类（分类、NER、QA） | 生成类（续写、对话） |

### 2. RMS Norm 计算公式？

$$\text{RMSNorm}(x) = \frac{x_i}{\sqrt{\frac{1}{N}\sum_{i=0}^{N-1}x_i^2 + \epsilon}} \cdot \text{weight}_i$$

### 3. RMS Norm 相比 Layer Norm 有什么特点？

- 省去了计算均值和减均值的环节，直接除均方根
- 效果相当，计算量更小

参考：https://zhuanlan.zhihu.com/p/9788103003

### 4. FFN 块计算公式？

```python
# SwiGLU 结构（Qwen2）
class MLP(nn.Module):
    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))
```

三个矩阵：
- `gate_proj`：升维 + 门控
- `up_proj`：升维
- `down_proj`：降维

### 5. 激活函数

**SiLU / Swish：**
$$\text{SiLU}(x) = x \cdot \sigma(x) = \frac{x}{1 + e^{-x}}$$

**GeLU：**
$$\text{GeLU}(x) = x \cdot \text{CDF}(x)$$
其中 CDF 是正态分布累积函数

---

## 五、Softmax 相关

### 1. Softmax 数值溢出及解决方法？

Softmax 的 exp 计算容易数值溢出。解决方法：所有 logits 减去最大值（logsumexp trick）。

$$\text{softmax}(x) = \text{softmax}(x - \max(x))$$

分子分母同时除以常数，结果不变。

---

## 六、微调方法

### 1. LLM 微调方式有哪些？

- 全参数微调
- LoRA
- Prefix Tuning / P-Tuning
- Adapter

### 2. Adapter 和 LoRA 的区别？

TODO

### 3. P-Tuning 原理

在输入 token 前面额外加入若干可训练的向量作为额外的 token。
- v1：只在输入层加
- v2：在每一层 hidden-state 都加

参考：https://zhuanlan.zhihu.com/p/635848732

---

## 七、RLHF 相关

### 1. PPO 用了几个模型？

TODO

### 2. GRPO 相比 PPO 节省了哪个模型？

TODO

### 3. RLHF 的 KL 散度作用？

计算 Actor 和 Reference 之间的 KL 散度，作为正则项：
- 防止模型偏离 Reference 太多
- 稳定训练
- 一定程度上保持泛化性

### 4. RLHF 的熵正则？

训练时 Actor 输出的熵可能降低，导致 rollout 高度相似，降低训练效率。

加入熵作为 loss 项，维持一定熵值，丰富 rollout 多样性。

### 5. 什么是 Reward Hacking？

TODO

### 6. Reference 模型在训练中会更新吗？

TODO

### 7. GRPO 的 Training Loop 描述？

TODO

### 8. 有了解过 RLHF 框架？OpenRLHF 和 veRL 的区别？

TODO

---

## 八、训练与显存优化

### 1. Gradient Accumulation 原理？

TODO

### 2. Gradient Checkpointing 原理？

TODO

### 3. 训练时左 padding 还是右 padding？

TODO

### 4. DeepSpeed Zero1/2/3 区别？

TODO

### 5. FSDP 和 DeepSpeed Zero3 区别？

TODO

### 6. HSDP 和 FSDP 区别？

TODO

### 7. All-Reduce 和 All-Gather 区别？

TODO

### 8. Attention Mask 的作用？

TODO

---

## 九、推理优化

### 1. 投机采样（Speculative Sampling）原理？

1. 用小模型（Draft Model）快速生成草稿序列
2. 大模型一次前向验证多个草稿 token
3. 逐个验证：若大模型概率 ≥ 草稿模型概率，接受；否则拒绝并重新采样
4. 不改变输出分布，但大幅减少大模型调用次数

### 2. LLM 复读机问题？原因？

**现象**：模型陷入重复循环，如 "the the the..." 或重复短语。

**原因**：
- 训练目标局部化，不感知全局重复
- 预测时陷入 pattern 会自我强化

### 3. 如何缓解 LLMs 复读机问题？

TODO

---

## 十、Scaling Law

针对某一数据分布，在某计算量限定条件下，存在最优的模型大小和数据大小配比，与计算量的幂呈线性关系。

**经验公式**：`C ≈ 6ND`
- C：计算量（FLOPs）
- N：模型参数量
- D：训练 token 数

参考：https://zhuanlan.zhihu.com/p/667489780

---

## 十一、评估指标

### 1. PPL（Perplexity）公式？

$$\text{PPL}(W) = \left(\prod_{t=1}^{N} P(w_t | w_1, ..., w_{t-1})\right)^{-\frac{1}{N}}$$

等价于交叉熵损失的指数。

### 2. MMLU 如何计算准确率？

参考：https://huggingface.co/blog/zh/open-llm-leaderboard-mmlu

---

## 十二、其他问题

### 1. LLM 输入和输出 Embedding 能否共用？

TODO

### 2. Qwen2 词表大小？

TODO

### 3. 小语言模型，相同参数量是更深还是更宽好？

TODO

### 4. 线性 Attention 了解吗？

TODO

### 5. Base 模型和 Chat 模型的 EOS Token 分别是什么？

TODO

### 6. Function Call 是什么？

TODO

### 7. MoE 原理和实现？

TODO

### 8. Harness方案是什么？


---

## 十三、代码考核题

### 待实现

- [ ] 手撕 Cross Attention
- [ ] 手撕 MHA（PyTorch）
- [ ] 手撕 RoPE
- [ ] 写 ReLU 和 SwiGLU
- [ ] 手撕 Softmax（含数值稳定性）
- [ ] 手撕 RMS Norm
- [ ] 手撕 Layer Norm

---

## 十四、算法题（SeedCoding）

1. [x] [802. 找到最终的安全状态](https://leetcode.cn/problems/find-eventual-safe-states/)
2. [x] [199. 二叉树的右视图](https://leetcode.cn/problems/binary-tree-right-side-view/)
3. [x] [19. 删除链表的倒数第 N 个结点](https://leetcode.cn/problems/remove-nth-node-from-end-of-list/)
4. [x] [88. 合并两个有序数组](https://leetcode.cn/problems/merge-sorted-array/)
5. [x] [3. 无重复字符的最长子串](https://leetcode.cn/problems/longest-substring-without-repeating-characters/)
6. [x] 海量数据求中位数, 询问数据范围等等，可以提出用分桶的方法，基本只需要On，如果未知分布，或者分布非常不均衡，可以用分块排序与作为兜底
7. [x] [79. 单词搜索](https://leetcode.cn/problems/word-search/)
8. [x] [704. 二分查找](https://leetcode.cn/problems/binary-search/) # NOTE: 这题得靠背
9. [x] [面试题 10.03. 搜索旋转数组](https://leetcode.cn/problems/search-rotate-array-lcci/) # NOTE: 这题得靠背
10. [x] [221. 最大正方形](https://leetcode.cn/problems/maximal-square/)
11. [x] [1608. 特殊数组的特征值](https://leetcode.cn/problems/special-array-with-x-elements-greater-than-or-equal-x/)
12. [x] [322. 零钱兑换](https://leetcode.cn/problems/coin-change/)
13. [x] [518. 零钱兑换 II](https://leetcode.cn/problems/coin-change-ii/)

## SUPPLEMENTARY

1. [图像理解与生成统一模型——前沿模型架构理解](https://zhuanlan.zhihu.com/p/1943651171823777079) 

---

*最后更新: 2026-04-09*