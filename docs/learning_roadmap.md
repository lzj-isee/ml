# NumPy 实现深度学习算法 - 面试准备路线图

## 概述

本文档整理了大模型算法工程师面试中常见的 NumPy 手写算法题，按难度和重要性分级，帮助你系统性地准备面试。

---

## 第一阶段：基础层模块 (必考 ⭐⭐⭐)

这些是最基础的模块，几乎所有面试都会涉及。

### 1. 激活函数
- [ ] ReLU / LeakyReLU / PReLU
- [ ] Sigmoid / Tanh
- [ ] GELU (Transformer常用)
- [ ] Swish / SiLU
- [ ] Softmax (重点，手写要注意数值稳定性)

### 2. 归一化层 (Norm Layer)
- [ ] Batch Normalization (训练/推理模式区别)
- [ ] Layer Normalization (Transformer标配)
- [ ] RMS Normalization (LLaMA等大模型常用)
- [ ] Group Normalization
- [ ] Instance Normalization

### 3. 正则化
- [ ] Dropout (训练/推理模式区别)
- [ ] L1/L2 Regularization

### 4. 线性层
- [ ] 全连接层 (Linear/Dense)
- [ ] 前向传播
- [ ] 反向传播 (重点!)

---

## 第二阶段：卷积与池化 (常考 ⭐⭐⭐)

### 1. 卷积操作
- [ ] Conv1d / Conv2d 前向传播
- [ ] im2col 实现
- [ ] 反向传播 (梯度对kernel和input)

### 2. 池化
- [ ] Max Pooling 前向+反向
- [ ] Average Pooling

### 3. 其他
- [ ] Transposed Convolution (转置卷积)

---

## 第三阶段：循环网络 (中等频率 ⭐⭐)

虽然现在大模型主要用Transformer，但RNN基础仍是重要考点。

- [ ] RNN Cell
- [ ] LSTM (遗忘门、输入门、输出门)
- [ ] GRU
- [ ] BPTT (Backpropagation Through Time)

---

## 第四阶段：Attention机制 (核心考点 ⭐⭐⭐⭐⭐)

**这是大模型面试的重中之重！**

### 1. 基础Attention
- [ ] Scaled Dot-Product Attention
- [ ] Self-Attention
- [ ] Cross-Attention
- [ ] Masked Attention (Causal Mask)

### 2. 多头注意力
- [ ] Multi-Head Attention (MHA)

### 3. 高效Attention变体 (进阶)
- [ ] Multi-Query Attention (MQA)
- [ ] Grouped Query Attention (GQA)
- [ ] Flash Attention 思想 (了解即可，面试可能问原理)

### 4. 位置编码
- [ ] Sinusoidal Position Encoding
- [ ] RoPE (Rotary Position Embedding) - 重点!
- [ ] ALiBi Position Encoding

### 5. KV Cache
- [ ] KV Cache 实现 (推理优化必问)

---

## 第五阶段：Transformer完整实现 (核心 ⭐⭐⭐⭐⭐)

- [ ] Transformer Encoder Block
- [ ] Transformer Decoder Block
- [ ] 完整的 Encoder-Decoder Transformer
- [ ] GPT 风格 Decoder-only Transformer

---

## 第六阶段：大模型架构特性 (高阶 ⭐⭐⭐⭐)

### 1. MoE (Mixture of Experts)
- [ ] Expert Network
- [ ] Top-k Gating / Router
- [ ] Load Balancing Loss
- [ ] 完整 MoE Layer

### 2. FFN变体
- [ ] SwiGLU (LLaMA使用)
- [ ] GeGLU

### 3. 其他优化
- [ ] Gradient Checkpointing (思想)
- [ ] Mixed Precision Training (思想)

---

## 第七阶段：优化器与损失函数 (常考 ⭐⭐⭐)

### 1. 优化器
- [ ] SGD (with Momentum)
- [ ] Adam / AdamW
- [ ] Learning Rate Scheduler (Cosine, Linear Warmup)

### 2. 损失函数
- [ ] Cross Entropy Loss
- [ ] Binary Cross Entropy
- [ ] MSE / MAE
- [ ] Focal Loss
- [ ] KL Divergence

---

## 第八阶段：传统机器学习 (可能涉及 ⭐⭐)

虽然是深度学习岗位，但有时会考察ML基础：

### 1. 分类器
- [ ] K-Nearest Neighbors (KNN)
- [ ] Naive Bayes
- [ ] Logistic Regression
- [ ] SVM (思想为主)

### 2. 聚类
- [ ] K-Means

### 3. 其他
- [ ] Decision Tree
- [ ] Random Forest (思想)
- [ ] PCA (降维)

---

## 学习建议

### 顺序安排
1. **先攻克第一阶段和第四阶段** - 这是面试最高频的考点
2. **然后学习第五、六阶段** - Transformer和MoE是大模型核心
3. **再补充第二、三、七阶段** - 卷积、RNN、优化器是基础
4. **最后看第八阶段** - 传统ML作为补充

### 练习方法
1. 每个模块先用 NumPy 实现前向传播
2. 重点实现反向传播（梯度计算）
3. 与 PyTorch 实现对比验证正确性
4. 注意数值稳定性（如 softmax 的 logsumexp trick）

### 面试技巧
- 手写代码时，先写清楚输入输出 shape
- 注意区分训练/推理模式
- 反向传播要能推导梯度公式
- 优化时间复杂度和空间复杂度

---

## 文件结构规划

```
ml/
├── docs/
│   ├── learning_roadmap.md      # 本文档
│   └── notes/                   # 学习笔记
├── implementations/             # NumPy实现代码
│   ├── basics/                  # 基础模块
│   │   ├── activations.py
│   │   ├── normalizations.py
│   │   └── layers.py
│   ├── attention/               # Attention相关
│   │   ├── attention.py
│   │   ├── multi_head.py
│   │   ├── rope.py
│   │   └── kv_cache.py
│   ├── transformer/             # Transformer
│   │   └── transformer.py
│   ├── moe/                     # MoE
│   │   └── moe.py
│   ├── cnn/                     # 卷积
│   │   └── conv.py
│   ├── rnn/                     # RNN
│   │   └── rnn.py
│   └── optimizers/              # 优化器和损失函数
│       ├── optimizers.py
│       └── losses.py
└── tests/                       # 测试文件 (与PyTorch对比)
    ├── test_activations.py
    ├── test_attention.py
    └── ...
```

---

## 进度追踪

| 阶段 | 模块 | 状态 | 完成日期 |
|------|------|------|----------|
| 1 | 激活函数 | ⬜ | |
| 1 | 归一化层 | ⬜ | |
| 1 | 全连接层+BP | ⬜ | |
| 2 | 卷积层 | ⬜ | |
| 4 | Attention | ⬜ | |
| 4 | Multi-Head Attention | ⬜ | |
| 4 | RoPE | ⬜ | |
| 5 | Transformer | ⬜ | |
| 6 | MoE | ⬜ | |
| 7 | 优化器 | ⬜ | |
| 7 | 损失函数 | ⬜ | |

---

## 参考资源

- [The Annotated Transformer](http://nlp.seas.harvard.edu/annotated-transformer/)
- [llm.c](https://github.com/karpathy/llm.c) - Andrej Karpathy的LLM实现
- [nndl.github.io](https://nndl.github.io/) - 神经网络与深度学习
- PyTorch官方文档 - 作为实现参考

---

*最后更新: 2026-04-09*