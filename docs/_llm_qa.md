## 牛客摘录
### <font style="color:rgb(51, 51, 51);">Transformer的自注意力机制及相比RNN的优势？</font>
优势：

1. RNN难以有效捕捉长距离依赖；自注意力机制直接建模所有 token 之间的关系，无论距离远近，轻松捕捉全局依赖。
2. 上下文信息逐时间步累积，前向传播中较早的信息可能丢失；每个 token 都能直接访问整个序列的上下文，信息保留更完整。
3. RNN固定内存占用；transformer为二阶复杂度，超长context占用大量显存

### ROPE？
以绝对位置编码的计算方式实了相对位置编码，要求qk的dim必须为偶数，两个一组，做theta旋转。向量旋转后，模长不变，求内积的结果收到旋转角度的影响，体现了相对位置编码信息

[https://zhuanlan.zhihu.com/p/662790439](https://zhuanlan.zhihu.com/p/662790439)

### <font style="color:rgb(51, 51, 51);">自注意力计算中为何除以 $$\sqrt{d_k} $$？</font>
各维度独立的高维正态分布的方差随着维度线性增长

### <font style="color:rgb(51, 51, 51);">现在LLM微调的方式有哪些？ 再问Adapter和Lora的区别？</font>
ptuning、全参数微调、lora微调、adapter

lora是调整linear层的参数，adapeter是增加了额外的linear层

### <font style="color:rgb(51, 51, 51);">Prefix LM与Causal LM区别？</font>
causalLM是完全单向的，第k个token无法向后感知到k+1个token，attention是一个下三角矩阵

prefixLM分成两部分，prefix部分可以双向感知，但是后续的generation部分只能向前感知无法向后感知，attention矩阵分成两块，一块完全矩阵，一块下三角

### GQA原理
hidden-state分成两路，其中一路被分为head-num * head-dim的q向量

另一路被分为kv-num * head-dim的kv向量，

比如在qwen2-0.5b中，hidden-state为896，head-num为14，head-dim为64

kv-num为2，那么7个q会共享一个1个kv

### <font style="color:rgb(51, 51, 51);">介绍一下softmax的数值溢出，以及有什么办法解决。</font>
softmax其中的指数计算exp容易造成数值溢出，解决思路是将logits减去一个常数，使logits足够小。

由于是所有logits都减去一个常数，相当于分子分母同时除以一个常数，这时候的softmax的计算结果不会发生改变

### 为什么Transformer的架构需要多头注意力机制？
attention组成了一种动态的线性加权计算更新隐空间的向量，在只有一个head的情况下，每次attention都只能关注一个方面

保持dim不变而增加head数量，增加了模型的表示能力，使一个attention层能从不同方面处理隐空间向量

### 为什么Transformer需要位置编码？
transformer处理序列数据的时候没有时序计算，token之间的处理是并行的，需要额外容易token之间的位置信息

### 为什么会出现LLMs复读机问题？
数据和训练的问题，首先训练目标是局部的，不感知全局的重复，其次如果预测的时候陷入了一个pattern，这个pattern会反过来强化模型继续沿着这个pattern继续生成

### PPL公式是什么
$ PPL(W) = \left( \prod_{t=1}^{N} P(w_t | w_1, ..., w_{t-1}) \right)^{-\frac{1}{N}} $

### 讲讲位置编码
[https://zhuanlan.zhihu.com/p/642846676](https://zhuanlan.zhihu.com/p/642846676)

### RLHF的熵正则
训练时会出现actor输出结果的熵降低的情况，导致rollout结果高度相似，降低训练效率。

加入熵作为loss，使模型训练时维持一定大小的熵，丰富rollout多样性不至于坍缩。

> 但是我在GRPO训练countdown-game的时候试过，也可以不加，能稳在大约0.1
>

### RLHF的kl散度的理解？
计算的是actor和reference之间的KL散度，loss目标是降低actor和reference之间关于当前输出token的probability的KL估计值。

主要是为了使模型训练不偏移reference太多，稳定训练，有说法是一定程度上保持模型的泛化性。

### 讲解下scaling law
[https://zhuanlan.zhihu.com/p/667489780](https://zhuanlan.zhihu.com/p/667489780)

<!-- 这是一张图片，ocr 内容为：1T 6.0 10B 5.5 1012 5B 5.0 2.5B 100B 678 4.5 PARAMETERS 1011 1B TOKENS 10B 500M 250M 1010 1.0B 75M 2.5 109 100M 2.0 1025 1017 1021 1023 1019 1025 1019 1021 1023 1020 1021 1017 1019 1018 1017 FLOPS FLOPS FLOPS -->
![](https://cdn.nlark.com/yuque/0/2025/png/26195591/1746451052414-2f646638-8430-4a86-86d8-e1fcd6dd26f4.png)

针对某一数据分布，在某一个计算量限定条件下，存在最优效率的模型大小和数据大小配比，且与计算量的幂呈线性关系。

可以在小规模的模型下实验loss和计算量的曲线，取下包络，进而计算出最优计算效率的情况下，计算量和模型大小的关系，以此预测大规格模型的size、所需的数据量、以及最后可能达到的loss。

> 计算量和模型参数以及token的关系大约是C=6ND，其中N表示模型的参数（只包含了linear，不包含embedding这些），D表示训练的token数量。
>

### 说一下Ptuning的原理
[https://zhuanlan.zhihu.com/p/635848732](https://zhuanlan.zhihu.com/p/635848732)

在输入token（基于每一层的hidden-state）前面额外加入若干个可以训练的向量作为额外的token补充，v1版本只在训练层加，v2版本在每一层的hidden-state都加。

### BERT和GPT的区别，都是怎么训练的
<font style="color:rgb(26, 28, 30);">BERT 和 GPT 都是非常有影响力的预训练语言模型，都基于 Transformer 架构，但它们在设计哲学和擅长任务上有所不同。主要区别体现在以下几点：</font>

+ **<font style="color:rgb(26, 28, 30);">架构上：</font>**<font style="color:rgb(26, 28, 30);"> </font><font style="color:rgb(26, 28, 30);">BERT 采用的是 Transformer 的</font><font style="color:rgb(26, 28, 30);"> </font>**<font style="color:rgb(26, 28, 30);">Encoder-only</font>**<font style="color:rgb(26, 28, 30);"> </font><font style="color:rgb(26, 28, 30);">结构，而 GPT 采用的是 Transformer 的</font><font style="color:rgb(26, 28, 30);"> </font>**<font style="color:rgb(26, 28, 30);">Decoder-only</font>**<font style="color:rgb(26, 28, 30);"> </font><font style="color:rgb(26, 28, 30);">结构。</font>
+ **<font style="color:rgb(26, 28, 30);">信息处理方向（注意力机制）上：</font>**<font style="color:rgb(26, 28, 30);"> </font><font style="color:rgb(26, 28, 30);">BERT 具有</font><font style="color:rgb(26, 28, 30);"> </font>**<font style="color:rgb(26, 28, 30);">双向</font>**<font style="color:rgb(26, 28, 30);"> </font><font style="color:rgb(26, 28, 30);">的注意力机制，处理一个词时能同时看到它左右两边的上下文，这使得它非常善于理解文本。而 GPT 具有</font><font style="color:rgb(26, 28, 30);"> </font>**<font style="color:rgb(26, 28, 30);">单向（因果）</font>**<font style="color:rgb(26, 28, 30);"> </font><font style="color:rgb(26, 28, 30);">的注意力机制，处理一个词时只能看到它左边的上下文，这使其天然适合进行文本生成。</font>
+ **<font style="color:rgb(26, 28, 30);">预训练任务上：</font>**<font style="color:rgb(26, 28, 30);"> </font><font style="color:rgb(26, 28, 30);">BERT 的核心预训练任务是</font><font style="color:rgb(26, 28, 30);"> </font>**<font style="color:rgb(26, 28, 30);">Masked Language Model (MLM)</font>**<font style="color:rgb(26, 28, 30);">，通过预测被遮盖的词来学习双向上下文；以及</font><font style="color:rgb(26, 28, 30);"> </font>**<font style="color:rgb(26, 28, 30);">Next Sentence Prediction (NSP)</font>**<font style="color:rgb(26, 28, 30);"> </font><font style="color:rgb(26, 28, 30);">来理解句子关系。GPT 的核心预训练任务是</font><font style="color:rgb(26, 28, 30);"> </font>**<font style="color:rgb(26, 28, 30);">Causal Language Modeling (CLM)</font>**<font style="color:rgb(26, 28, 30);">，即预测序列中的下一个 token，这直接服务于其生成能力。</font>
+ **<font style="color:rgb(26, 28, 30);">擅长任务上：</font>**<font style="color:rgb(26, 28, 30);"> 正因为架构和训练方式的不同，BERT 更擅长需要深度理解文本的任务，比如文本分类、序列标注、问答等；而 GPT 更擅长文本生成、续写、对话等任务。</font>

### 投机采样简单说一下
<font style="color:rgb(26, 28, 30);">好的，投机采样（Speculative Sampling）是一种加速大型语言模型推理生成速度的技术。您提到的‘投石采样’是同一个意思。它的核心思想是利用一个比大模型小得多、推理速度更快的模型（称为草稿模型），来预先猜测接下来可能生成的几个 token，形成一个‘草稿序列’。然后，将这个草稿序列和已生成的文本一起输入到原始的大模型（作为验证模型）中进行一次前向推理。大模型会计算草稿序列中每个 token 在当前上下文下的概率。</font>

<font style="color:rgb(26, 28, 30);">接着，我们逐个验证这些草稿 token：如果大模型计算出的概率大于等于草稿模型对同一个 token 预测的概率，我们就接受这个 token。一旦遇到一个 token 不满足这个条件，或者草稿序列结束，我们就停止接受草稿，并从当前位置开始，根据大模型的概率分布进行一次采样，生成最终的下一个 token。整个过程会从这个新生成的 token 开始重复。</font>

<font style="color:rgb(26, 28, 30);">之所以能提速，是因为大模型进行一次前向推理可以同时计算多个草稿 token 的概率，这相比于一个 token 一个 token 地进行大模型推理要高效得多，尤其是在硬件利用率上。通过这种方式，我们可以用一次大模型的计算，‘批量’地验证并接受多个 token，从而显著减少总的大模型调用次数，提高生成速度。这种方法的一个重要优点是，在保证实现正确的情况下，它不会改变大模型原本的输出概率分布，即生成的文本质量理论上与直接用大模型逐个采样是一样的。</font>

### transformer自注意力，计算过程，softmax为什么要进行缩放，不缩放会导致什么？
各维度独立的高维正态分布的方差随着维度线性增长。不缩放容易导致score数值过大溢出，也容易导致softmax的梯度消失，相当于temperature特别低的情况。

```markdown
好的，这是一个非常经典的 Transformer 自注意力机制中的细节问题，也是面试中常考点。你的理解方向是完全正确的！

我们来详细分解一下。

**1. Transformer 自注意力（Self-Attention）的计算过程**

自注意力机制的目的是让序列中的每个 token 都能“关注”到序列中的其他所有 token，并根据它们的关联程度来加权求和，生成一个新的表示向量，这个新的向量包含了来自整个序列的信息。

基本步骤如下：

a.  **线性投影 (Linear Projections):**
    *   对于输入的 token embedding 序列 `X` (维度 `sequence_length x embedding_dim`)，通过三个不同的线性层（全连接层），分别将其投影到 Query (Q)、Key (K)、Value (V) 三个矩阵。
    *   `Q = X * W_Q`
    *   `K = X * W_K`
    *   `V = X * W_V`
    *   这里 `W_Q`, `W_K`, `W_V` 是可学习的权重矩阵，它们的维度通常是 `embedding_dim x d_k`, `embedding_dim x d_k`, `embedding_dim x d_v`。`d_k` 是 Key/Query 向量的维度，`d_v` 是 Value 向量的维度。在标准的 Transformer 中，通常设置 `d_k = d_v`。
    *   现在 Q, K, V 的维度都是 `sequence_length x d_k` 或 `sequence_length x d_v`。

b.  **计算注意力分数 (Compute Attention Scores):**
    *   计算 Query 和 Key 矩阵的乘积 `Q * K^T`。
    *   `Q` 的每一行（一个 query 向量）与 `K^T` 的每一列（即 `K` 的每一行，一个 key 向量）进行点积。
    *   这个点积的结果是一个 `sequence_length x sequence_length` 的矩阵，矩阵中的每个元素 `(i, j)` 表示序列中第 `i` 个 token 的 query 向量与第 `j` 个 token 的 key 向量的点积。这个点积衡量了第 `i` 个 token 应该“关注”第 `j` 个 token 的程度。
    *   维度: (`sequence_length x d_k`) * (`d_k x sequence_length`) = (`sequence_length x sequence_length`)

c.  **缩放 (Scaling):**
    *   将上一步得到的注意力分数矩阵中的每一个元素**除以** `sqrt(d_k)`。
    *   `Scores = (Q * K^T) / sqrt(d_k)`
    *   这是你问题中关注的核心步骤。

d.  **Masking (可选):**
    *   在某些任务（如文本生成中的 Decoder 部分）或训练阶段（如 Padding），需要防止一个 token 关注到它后面的 token 或 Padding token。这时会在这里对分数矩阵应用一个 Mask，将不允许关注的位置的分数设置为一个非常小的负数（接近 `-inf`），这样经过 Softmax 后它们的权重就会接近 0。

e.  **Softmax: (Compute Attention Weights)**
    *   对经过缩放（和 Masking）后的分数矩阵的**每一行**应用 Softmax 函数。
    *   `Attention_Weights = Softmax(Scores)`
    *   Softmax 将分数转化为概率分布，使得每一行的元素之和为 1。这表示当前 token 对序列中其他所有 token 的注意力权重。
    *   维度不变: `sequence_length x sequence_length`

f.  **加权求和 (Compute Output):**
    *   将 Softmax 得到的注意力权重矩阵与 Value 矩阵相乘。
    *   `Output = Attention_Weights * V`
    *   这表示用计算出的注意力权重，对 Value 向量进行加权求和。
    *   `Attention_Weights` 的每一行（一个 query 向量对应的权重分布）与 `V` 的每一行（一个 value 向量）相乘并求和，得到新的表示向量。
    *   维度: (`sequence_length x sequence_length`) * (`sequence_length x d_v`) = (`sequence_length x d_v`)
    *   这个 `Output` 矩阵的每一行就是输入序列中对应 token 的新的、经过注意力机制整合了全局信息的表示。

**2. 为什么 Softmax 之前要进行缩放（除以 `sqrt(d_k)`）？**

这个缩放是为了防止点积的结果过大，从而导致 Softmax 的梯度消失。

*   **点积结果的范围：** `Q` 和 `K` 矩阵中的元素通常是经过线性投影得到的，并且在训练过程中会被归一化（例如 Batch Norm 或 Layer Norm）。假设 Q 和 K 的元素的均值为 0，方差为 1。对于两个向量 `q` 和 `k` (维度都是 `d_k`)，它们的点积 `q * k` 是 `d_k` 个对应元素乘积的和：`sum(q_i * k_i)`。
*   **方差随着维度增长：** 如果 `q_i` 和 `k_i` 是独立的随机变量，均值为 0，方差为 1，那么它们的乘积 `q_i * k_i` 的均值为 0，方差为 1（因为 `Var(XY) = E[X^2 Y^2] - (E[XY])^2`。如果独立且均值为 0，`E[XY]=0`，`Var(XY) = E[X^2]E[Y^2] = Var(X)Var(Y) = 1*1=1`）。点积 `q * k` 是 `d_k` 个这样的独立同分布随机变量的和。根据概率论，独立随机变量和的方差等于方差的和。因此，点积 `q * k` 的方差大约是 `d_k * 1 = d_k`。
*   **点积的幅度：** 方差是 `d_k` 意味着点积结果的标准差是 `sqrt(d_k)`。也就是说，点积结果的典型幅度会随着 `d_k` 的增大而增大，大约正比于 `sqrt(d_k)`。对于大型模型，`d_k` 通常较大（比如 64 或更多），点积的结果会变得相当大。

**3. 不缩放会导致什么问题？**

如果不除以 `sqrt(d_k)`，注意力分数 `Q * K^T` 的数值会很大。当这些很大的数值被输入到 Softmax 函数中时，会发生以下问题：

a.  **Softmax 梯度消失 (Vanishing Gradients):**
    *   Softmax 函数 `softmax(x_i) = exp(x_i) / sum(exp(x_j))`。
    *   当输入 `x_i` 非常大时，`exp(x_i)` 会变得非常大。如果 `x` 向量中存在一些数值差异，例如 `[100, 101, 102]`，那么 `exp(102)` 会比 `exp(100)` 或 `exp(101)` 大得多。Softmax 的结果会非常接近一个 One-hot 向量，例如 `[~0, ~0, ~1]`。
    *   Softmax 函数的梯度在输入值差异很大时会变得非常小。考虑 Softmax 的导数 `d(softmax(x_i))/dx_j`。当 `x_i` 远大于其他 `x_k` 时，`softmax(x_i)` 接近 1，其他 `softmax(x_j)` (j != i) 接近 0。此时导数 `d(softmax(x_i))/dx_i = softmax(x_i)(1 - softmax(x_i))` 接近 `1 * (1 - 1) = 0`。导数 `d(softmax(x_i))/dx_j (j != i) = -softmax(x_i) * softmax(x_j)` 接近 `-1 * 0 = 0`。
    *   这意味着，对于那些很大的输入分数，经过 Softmax 后的梯度会变得非常小，接近于零。
    *   在反向传播时，这些微小的梯度会乘以前一层传来的梯度，导致前面层的梯度进一步缩小，造成**梯度消失**。这使得模型难以学习到有效的注意力权重，特别是那些需要微妙调整注意力的场景。模型会倾向于把注意力权重集中到少数几个 token 上，无法捕捉更丰富的上下文关系。

b.  **数值不稳定 (Numerical Instability):**
    *   直接计算 `exp(x)` 当 `x` 非常大时，可能会导致数值溢出。虽然实际计算中会使用一些数值稳定的技巧（例如将输入减去最大值再进行指数运算），但缩放从根本上降低了输入的数值范围，有助于提升数值稳定性。

**结论：**

*   你的理解是正确的。点积的方差确实会随着维度 `d_k` 线性增长，导致点积的幅度（标准差）随着 `sqrt(d_k)` 增长。
*   除以 `sqrt(d_k)` 的目的就是为了“归一化”点积结果的幅度，使其方差保持在接近 1 的水平，无论 `d_k` 是多少。
*   这样做的好处是：将缩放后的分数输入到 Softmax 时，数值不会过大，避免 Softmax 输出过于“尖锐”，从而防止梯度过早地消失，保证模型在训练过程中能够有效地学习和调整注意力权重。这对于 Transformer 模型能够处理长序列和复杂依赖关系至关重要。
```

### 介绍一下YaRN
[https://zhuanlan.zhihu.com/p/15311461897](https://zhuanlan.zhihu.com/p/15311461897)

### 介绍一下NTK
[https://zhuanlan.zhihu.com/p/662790439](https://zhuanlan.zhihu.com/p/662790439)

### MMLU如何计算准确率的？
[https://huggingface.co/blog/zh/open-llm-leaderboard-mmlu](https://huggingface.co/blog/zh/open-llm-leaderboard-mmlu)

### RMS Norm 的计算公式写一下？
$ RMSNorm(x) = \frac{x_i}{\sqrt{\frac{1}{N}\sum_{i=0}^{N-1}{x^2_i} + \epsilon}} * \text{weight}_i $

### RMS Norm 相比于 Layer Norm 有什么特点？
[https://zhuanlan.zhihu.com/p/9788103003](https://zhuanlan.zhihu.com/p/9788103003)

RMSNorm相比LayerNorm省去了计算均值和减均值的环节，直接除均方根而不是方差。

RMSNorm效果和LayerNorm的效果相当，但是计算量更小。

### 介绍一下 FFN 块计算公式？
[https://zhuanlan.zhihu.com/p/650237644](https://zhuanlan.zhihu.com/p/650237644)

```python
class Qwen2MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return down_proj
```

三个矩阵，一个做门控，第一个矩阵做升维，然后过门控，第三个矩阵做降维。qwen用的silu

$ \text{SiLU}(x) = x * \text{sigmoid}(x)=x*\frac{1}{1 + exp(-x)} $

### 介绍一下 GeLU 计算公式？
$ \text{GeLU}(x)=x*\text{CDF}(x) $,其中CDF是正态分布分布概率累积函数

另外还有一个近似函数，但是系数复杂没记住，torch是有一	个gelu-fast的实现

### <font style="color:rgb(31, 35, 40);">介绍一下 Swish 计算公式？</font>
$ \text{Swish}(x) =x*\frac{1}{1 + exp(-\beta * x)} $

### MHA，GQA，MQA 三种注意力机制是否了解?区别是什么?


### Cross Attention 和 Self Attention 都是基于注意力机制的，有什么不同点？


### 什么是 LLMs 复读机问题？为什么会出现 LLMs 复读机问题？<font style="color:rgb(31, 35, 40);">如何缓解 LLMs 复读机问题？</font>


### 介绍一下 gradient accumulation 显存优化方式？


### 介绍一下 gradient checkpointing 显存优化方式？


### 什么是 函数调用(function call)？


### MOE原理和实现？


---

## 自己想的
### PPO用了几个模型，分别是什么


### GRPO相比PPO，节省了哪个模型


### 训练是左padding还是右padding，推理的时候是左padding还是右padding


### deepspeed的zero1、zero2、zero3之间的区别是什么


### torch的fsdp和deepspeed的zero3之间有什么区别


### 有了解过torch的hsdp吗，和fsdp有什么区别


### 有没有了解过RLHF的框架？OpenRLHF和veRL直接的区别是什么


### LLM的输入embedding和输出embedding可不可以共用一个


### qwen2的词表大小大概是多少


### 小语言模型SLM，相同参数量是深度更深好还是宽度更宽好


### 有没有了解过线性attention，简单讲讲


### base模型和chat模型的eos-token分别是什么


### 描述下GRPO的training-loop


### RLHF的reference模型在训练中会不会更新


### 分布式计算的all-reduce和all-gather之间的区别是什么，哪个通信量大


### 什么是RLHF的reward-hacking？


### 训练时候的attention-mask作用是什么






## 代码考核
### 手撕：cross-attention


### 写Relu和SwiGLU


### 手撕MHA（pytorch）


### 手撕ROPE




---

1. 专业名词太多
2. 稍微有点太细了

---

# SeedCoding
1. [802. 找到最终的安全状态 - 力扣（LeetCode）](https://leetcode.cn/problems/find-eventual-safe-states/description/)
2. [199. 二叉树的右视图 - 力扣（LeetCode）](https://leetcode.cn/problems/binary-tree-right-side-view/)
3. [19. 删除链表的倒数第 N 个结点 - 力扣（LeetCode）](https://leetcode.cn/problems/remove-nth-node-from-end-of-list/)
4. [88. 合并两个有序数组 - 力扣（LeetCode）](https://leetcode.cn/problems/merge-sorted-array/description/)
5. [3. 无重复字符的最长子串 - 力扣（LeetCode）](https://leetcode.cn/problems/longest-substring-without-repeating-characters/description/)
6. 海量数据求中位数
7. [79. 单词搜索 - 力扣（LeetCode）](https://leetcode.cn/problems/word-search/description/)
8. [704. 二分查找 - 力扣（LeetCode）](https://leetcode.cn/problems/binary-search/description/)
9. [面试题 10.03. 搜索旋转数组 - 力扣（LeetCode）](https://leetcode.cn/problems/search-rotate-array-lcci/description/)
10. [221. 最大正方形 - 力扣（LeetCode）](https://leetcode.cn/problems/maximal-square/description/)
11. [1608. 特殊数组的特征值 - 力扣（LeetCode）](https://leetcode.cn/problems/special-array-with-x-elements-greater-than-or-equal-x/description/)
12. [322. 零钱兑换 - 力扣（LeetCode）](https://leetcode.cn/problems/coin-change/description/)
13. [518. 零钱兑换 II - 力扣（LeetCode）](https://leetcode.cn/problems/coin-change-ii/description/)

