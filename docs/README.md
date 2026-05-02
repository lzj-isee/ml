# 文档目录

本文档汇总 `docs` 目录下所有学习资料的内容索引，方便快速查找和后续整理。

---

## 📚 LLM 系统知识点

| 文件 | 内容概述 |
|------|----------|
| [`llm/tokenization.md`](llm/tokenization.md) | 分词方法详解：BBPE (Byte-level BPE) 原理、构建步骤、与BPE对比、跨语言通用性、OOV问题处理 |
| [`llm/rlhf.md`](llm/rlhf.md) | RLHF/PPO/GRPO/DPO：GAE、Token/Sequence-level Advantage、Bradley-Terry、DPO推导与损失函数 |
| [`llm/decoding.md`](llm/decoding.md) | 解码与采样：Frequency/Presence Penalty计算原理、代码实现、与Temperature区别 |

---

## 📋 面试资料

### LLM 面试八股
| 文件 | 内容概述 |
|------|----------|
| [`_llm_qa.md`](_llm_qa.md) | 牛客网摘录的LLM面试题，包含Transformer原理、位置编码(RoPE/NTK/YaRN)、微调方法(LoRA/Adapter/P-Tuning)、RLHF(PPO/GRPO)、训练优化等问答 |
| [`llm_interview_qa.md`](llm_interview_qa.md) | 系统整理的LLM面试知识点，涵盖注意力机制(MHA/GQA/MQA)、归一化(LayerNorm/RMSNorm)、激活函数、推理优化(投机采样/KV Cache)、Scaling Law、评估指标等 |

### 学习路线图
| 文件 | 内容概述 |
|------|----------|
| [`learning_roadmap.md`](learning_roadmap.md) | NumPy实现深度学习算法的面试准备路线图，按8个阶段分级：基础模块→卷积→RNN→Attention→Transformer→MoE→优化器→传统ML，含进度追踪和文件结构规划 |

---

## 🔧 基础知识

### 计算机基础
| 文件 | 内容概述 |
|------|----------|
| [`basics/floating_point_formats.md`](basics/floating_point_formats.md) | 浮点数格式详解：FP32/FP16/BF16/FP8(E4M3/E5M2)的结构对比、数值范围、正规数/次正规数计算、Python验证代码 |

### 编程技巧
| 文件 | 内容概述 |
|------|----------|
| [`basics/python_tips.md`](basics/python_tips.md) | Python常用小技巧：自定义排序(cmp_to_key)、组合数计算(math.comb/scipy)等实用代码片段 |

---

## 📂 文件结构

```
docs/
├── README.md                    # 本文档（目录索引）
├── _llm_qa.md                   # LLM面试题摘录（牛客网来源）
├── learning_roadmap.md          # NumPy实现深度学习面试路线图
├── llm_interview_qa.md          # LLM面试八股系统整理
└── basics/                      # 基础知识
    ├── floating_point_formats.md   # 浮点数格式详解
    └── python_tips.md              # Python小技巧
```

---

## 📝 分类标签

| 标签 | 相关文件 |
|------|----------|
| `#Transformer` | `_llm_qa.md`, `llm_interview_qa.md`, `learning_roadmap.md` |
| `#Attention` | `_llm_qa.md`, `llm_interview_qa.md`, `learning_roadmap.md` |
| `#位置编码` | `_llm_qa.md`, `llm_interview_qa.md` |
| `#微调` | `_llm_qa.md`, `llm_interview_qa.md` |
| `#RLHF` | `_llm_qa.md`, `llm_interview_qa.md` |
| `#量化` | `basics/floating_point_formats.md` |
| `#面试题` | `_llm_qa.md`, `llm_interview_qa.md`, `learning_roadmap.md` |
| `#Python` | `basics/python_tips.md` |

---

*最后更新: 2026-05-02*
