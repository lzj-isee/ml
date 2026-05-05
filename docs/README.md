# 文档目录

本文档汇总 `docs` 目录下所有学习资料的内容索引，方便快速查找和后续整理。

---

## 📚 LLM 系统知识点

| 文件 | 内容概述 |
|------|----------|
| [`llm/tokenization.md`](llm/tokenization.md) | 分词方法详解：BBPE (Byte-level BPE) 原理、构建步骤、与BPE对比、跨语言通用性、OOV问题处理 |
| [`llm/rlhf.md`](llm/rlhf.md) | RLHF/PPO/GRPO/DPO/DAPO：GAE、Advantage计算、Bradley-Terry、DPO推导、**DAPO六项改进（含KL散度移除）** |
| [`llm/decoding.md`](llm/decoding.md) | 解码与采样：Frequency/Presence Penalty计算原理、代码实现、与Temperature区别 |
| [`llm/lora.md`](llm/lora.md) | LoRA原理与初始化：**降维层A高斯随机、升维层B初始化为0**、与Adapter对比、超参数 |
| [`llm/entropy_collapse_reward_hacking.md`](llm/entropy_collapse_reward_hacking.md) | 熵坍塌与Reward Hacking：**现象数据化**（200步/800步统计）、摘要生成任务案例、两者关系、DAPO规则奖励解决 |
| [`llm/auto_thinking.md`](llm/auto_thinking.md) | 大模型深度思考自主切换：**AutoThink三阶段RL**（基础奖励→准确率优化→长度精炼）、基于规则的不确定性计算（熵/Top-2） |
| [`llm/muon_optimizer.md`](llm/muon_optimizer.md) | Muon优化器：**正交化梯度动量**、Newton-Schulz迭代、显存节省50%、Distributed Muon（DP Gather） |
| [`llm/moe.md`](llm/moe.md) | MoE负载均衡：**Sequence粒度**负载均衡损失公式、与Token级对比、代码实现、超参数调优 |
| [`llm/qwen_evolution.md`](llm/qwen_evolution.md) | Qwen模型演进：**Qwen1→Qwen3-2507**架构改进、数据规模增长（3T→36T）、后训练流程演进、思考模式分离 |
| [`llm/kl_estimators.md`](llm/kl_estimators.md) | KL散度估计方法：**K1/K2/K3**三种近似、K3推导（二阶泰勒展开）、数值稳定性分析、为什么RL用K3 |
| [`llm/badcase_handling.md`](llm/badcase_handling.md) | 业务实践：**大模型Badcase解决策略**、6级优先级（Prompt→前后置→Agent→SFT→RL→预训练）、成本与效果权衡 |
| [`llm/perplexity.md`](llm/perplexity.md) | 模型评估指标：**困惑度PPL**计算公式、与交叉熵关系、代码实现、数值解读、局限性分析 |

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
| [`basics/distributed_comm.md`](basics/distributed_comm.md) | 分布式训练通信原语：**Scatter/Gather/Reduce/AllReduce/Broadcast**、通信量对比、Ring-AllReduce算法 |

---

## 📂 文件结构

```
docs/
├── README.md                    # 本文档（目录索引）
├── _llm_qa.md                   # LLM面试题摘录（牛客网来源）
├── learning_roadmap.md          # NumPy实现深度学习面试路线图
├── llm_interview_qa.md          # LLM面试八股系统整理
├── basics/                      # 基础知识
│   ├── floating_point_formats.md   # 浮点数格式详解
│   ├── python_tips.md              # Python小技巧
│   └── distributed_comm.md         # 分布式通信原语
└── llm/                         # LLM 系统知识
    ├── tokenization.md             # BBPE分词
    ├── rlhf.md                     # RLHF/PPO/GRPO/DPO/DAPO
    ├── decoding.md                 # 解码与采样
    ├── lora.md                     # LoRA原理与初始化
    ├── entropy_collapse_reward_hacking.md  # 熵坍塌与Reward Hacking
    ├── auto_thinking.md            # 深度思考自主切换(AutoThink)
    ├── muon_optimizer.md           # Muon优化器
    ├── moe.md                      # MoE负载均衡
    ├── qwen_evolution.md           # Qwen模型演进
    ├── kl_estimators.md            # KL散度估计方法（K1/K2/K3）
    ├── badcase_handling.md         # 大模型Badcase业务实践
    └── perplexity.md               # 困惑度PPL
```

---

## 📝 分类标签

| 标签 | 相关文件 |
|------|----------|
| `#Transformer` | `_llm_qa.md`, `llm_interview_qa.md`, `learning_roadmap.md` |
| `#Attention` | `_llm_qa.md`, `llm_interview_qa.md`, `learning_roadmap.md` |
| `#位置编码` | `_llm_qa.md`, `llm_interview_qa.md` |
| `#微调` | `_llm_qa.md`, `llm_interview_qa.md`, `llm/lora.md` |
| `#LoRA` | `llm/lora.md` |
| `#RLHF` | `_llm_qa.md`, `llm_interview_qa.md`, `llm/rlhf.md` |
| `#DAPO` | `llm/rlhf.md`, `llm/entropy_collapse_reward_hacking.md` |
| `#GRPO` | `llm/rlhf.md`, `llm/auto_thinking.md` |
| `#熵坍塌` | `llm/entropy_collapse_reward_hacking.md` |
| `#RewardHacking` | `llm/entropy_collapse_reward_hacking.md` |
| `#优化器` | `llm/muon_optimizer.md` |
| `#Muon` | `llm/muon_optimizer.md` |
| `#MoE` | `llm/moe.md` |
| `#负载均衡` | `llm/moe.md` |
| `#Qwen` | `llm/qwen_evolution.md` |
| `#KL散度` | `llm/kl_estimators.md` |
| `#业务实践` | `llm/badcase_handling.md` |
| `#困惑度` | `llm/perplexity.md` |
| `#评估指标` | `llm/perplexity.md` |
| `#量化` | `basics/floating_point_formats.md` |
| `#面试题` | `_llm_qa.md`, `llm_interview_qa.md`, `learning_roadmap.md` |
| `#Python` | `basics/python_tips.md` |
| `#分布式` | `basics/distributed_comm.md` |
| `#通信原语` | `basics/distributed_comm.md` |

---

*最后更新: 2026-05-05*
