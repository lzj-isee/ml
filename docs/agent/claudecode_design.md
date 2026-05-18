# Claude Code 独特设计分析

本文档汇总 Claude Code 在 Agent 设计上的独特之处，分析其效果好的原因。

---

## 1. 上下文窗口管理：五层渐进压缩

### 核心设计理念

大模型有上下文窗口限制。业界常见做法：
- **简单截断**：保留最近 N 条，旧的扔掉 → 编程 Agent 灾难（20 轮前读的关键配置可能已丢）
- **全量摘要**：整段对话总结成一段 → 贵（额外 API 调用）、有信息损失

Claude Code 的核心理念：**压缩一定有信息损失，所以能不压就不压，必须压的时候从最轻的手段开始**。

设计了**五个从轻到重的压缩手段**，像医院分诊：先试最温和的，不行再上猛药。

---

### 五层压缩概览

| 层级 | 手段 | 信息损失 | API 开销 | 触发条件 |
|------|------|----------|----------|----------|
| 第 1 层 | 大结果存磁盘 | 几乎为零 | 零 | 工具结果 > 50KB |
| 第 2 层 | 砍掉远古消息 | 低 | 零 | 消息过时 |
| 第 3 层 | 清理老工具输出 | 中低 | 零 | 缓存过期/数量超限 |
| 第 4 层 | 读时投影压缩 | 中 | 低 | 上下文达 90% |
| 第 5 层 | 全量摘要 | 高 | 高（一次 API 调用） | 上下文达 ~93% |

**越往下代价越高，但效果也越强。大部分场景下前三层就够用了**。

---

### 第 1 层：大结果存磁盘

**问题**：Agent 读 10MB 日志文件，Read 工具返回全部内容，吃掉几万 Token。同时读 3 个大文件，一条消息占掉大半个窗口。

**解决**：工具结果进入消息列表前先做"体检"：

```typescript
async function maybePersistLargeToolResult(
  toolResultBlock: ToolResultBlockParam,
  toolName: string,
): Promise<ToolResultBlockParam> {
  const size = contentSize(content)
  // 单个工具结果超过阈值（默认约 50KB）？
  if (size <= threshold) {
    return toolResultBlock  // 没超，原样通过
  }
  // 超了！把完整内容存到磁盘文件
  const result = await persistToolResult(content, toolUseId)
  // 用一个 2KB 的预览替换原内容
  const preview = buildLargeToolResultMessage(result)
  return { ...toolResultBlock, content: preview }
}
```

**双阈值控制**：
- 单个工具结果 > 50KB → 存磁盘，消息里只留 2KB 预览
- 同一条消息所有工具结果总和 > 200KB → 挑最大的几个存磁盘

**精妙之处**：完整内容还在磁盘上。如果模型后面需要那个大文件的某个片段，可以再次调用 Read 工具读取特定行范围。

---

### 第 2 层：砍掉远古消息（HISTORY_SNIP）

**问题**：长对话上百轮，开头那几轮（探索性提问、试探性回答）到后面几乎完全没用，但仍然占着上下文空间。

**解决**：最"粗暴"但也最高效的一层，**直接砍掉**对话开头的一批老消息，插入边界标记告诉模型"这之前的内容已被清理"。

```typescript
if (feature('HISTORY_SNIP')) {
  const snipResult = snipModule.snipCompactIfNeeded(messagesForQuery)
  messagesForQuery = snipResult.messages
  snipTokensFreed = snipResult.tokensFreed
  if (snipResult.boundaryMessage) {
    yield snipResult.boundaryMessage  // 插入边界标记
  }
}
```

**不做任何摘要**，零 API 开销。

**重要细节**：Snip 会把 `snipTokensFreed`（释放了多少 Token）传给第 5 层 Auto-Compact。如果 Snip 已释放足够空间，Auto-Compact 就不需要触发，避免两层同时压缩。

---

### 第 3 层：裁剪老的工具输出（Micro-Compact）

**问题**：经过前两层，剩下"不太老但也不太新"的消息。里面大量工具输出已过时（如 30 分钟前读的文件可能已被改过）。

**解决**：**时间衰减**策略——越老的工具结果越不重要，可以被裁剪。

**可裁剪工具**（可重新获取的）：
```typescript
const COMPACTABLE_TOOLS = new Set([
  FILE_READ_TOOL_NAME,    // 读文件 → 可以重新读
  ...SHELL_TOOL_NAMES,    // 执行命令 → 可以重新执行
  GREP_TOOL_NAME,         // 搜索 → 可以重新搜
  GLOB_TOOL_NAME,         // 查找文件 → 可以重新查
  FILE_EDIT_TOOL_NAME,    // 编辑文件 → 结果可裁剪
  // ...
])
```

**不可裁剪**（不可重新获取的）：AgentTool（子 Agent 输出）、TaskTool（任务状态）——子 Agent 的推理过程砍掉就真的丢了。

**裁剪逻辑**：
```typescript
// 收集所有可裁剪工具的结果 ID
const compactableIds = collectCompactableToolIds(messages)
// 保留最近 5 个，其余全部清理
const keepSet = new Set(compactableIds.slice(-keepRecent))
const clearSet = compactableIds.filter(id => !keepSet.has(id))
```

被裁剪的工具结果替换为标记：`[Old tool result content cleared]`

模型看到标记就知道"这里原来有内容但被清理了"，如果需要可以自己决定重新读文件或重新执行命令。

**为什么叫"时间衰减"**：触发条件跟时间有关，当距离上一次 API 调用超过一定时间（默认约 60 分钟），说明大模型 API 端的 Prompt Cache 大概率已过期。既然缓存已没，清理旧的也不会浪费之前的缓存投入。

---

### 第 4 层：读时投影（Context Collapse）

**问题**：经过前三层，上下文还是太大。全量摘要代价高、信息损失大。有没有"中间态"？

**解决**：**读时投影（Read-Time Projection）**——不修改原始消息，只在调用 API 的那一刻，**动态计算一个"压缩视图"**给模型看。

```typescript
// query.ts 中的调用
// 注意：这是一个"读时投影"——不修改 REPL 的完整历史，
// 只在发送给 API 时计算压缩视图
if (feature('CONTEXT_COLLAPSE') && contextCollapse) {
  const collapseResult = await contextCollapse.applyCollapsesIfNeeded(
    messagesForQuery,
    toolUseContext,
    querySource,
  )
  messagesForQuery = collapseResult.messages
}
```

**两级阈值**：
- **90% 上下文窗口**：主动开始分段压缩旧消息（预留缓冲区）
- **95% 上下文窗口**：紧急压缩更多内容（留足 API 响应空间）

**与第 5 层的配合**：Context Collapse 运行在 Auto-Compact 之前。如果已通过"读时投影"把上下文压到阈值以下，Auto-Compact 就完全不需要触发。模型保留更多细节上下文，而不是被粗糙的全量摘要替代。

---

### 第 5 层：全量摘要（Auto-Compact）

**问题**：前面四层都不够用，上下文实在太大，必须彻底压缩。

**触发阈值**：
```typescript
function getAutoCompactThreshold(model: string): number {
  const effectiveContextWindow = getEffectiveContextWindowSize(model)
  // 有效窗口 - 13K 缓冲区 = 触发阈值
  return effectiveContextWindow - 13_000
}
// 200K Token 模型：180K - 13K = 167K 时触发
```

**三步走**：

**Step 1：生成摘要**
调用大模型，按多维度总结：
- 用户的主要请求和意图
- 关键技术概念
- 涉及的文件和代码片段
- 遇到的错误和修复方案
- 问题解决过程
- 用户的所有消息（不能遗漏任何一条）
- 待完成的任务、当前工作状态、建议的下一步

**Step 2：替换旧消息**
把压缩边界之前的所有消息删掉，替换为生成的摘要。插入边界标记，记录压缩前的 Token 数。

**Step 3：Post-Compact Restoration（压缩后恢复）**
```typescript
export const POST_COMPACT_MAX_FILES_TO_RESTORE = 5
export const POST_COMPACT_TOKEN_BUDGET = 50_000
export const POST_COMPACT_SKILLS_TOKEN_BUDGET = 25_000
```
从文件状态缓存中找出最近访问的文件，按时间排序，挑选最多 5 个、总共不超过 50K Token 的文件内容**重新注入**。同时恢复活跃的 Skill（不超过 25K Token），如有进行中的 Plan 也恢复 Plan 文件。

**为什么要恢复？** 压缩后模型"失忆"了，不记得刚才读过的文件内容。如果不恢复，模型的第一反应就是"让我重新读一下文件"，白白浪费一轮工具调用。主动恢复让模型**无缝继续工作**，体验上几乎感觉不到压缩发生过。

**熔断器机制**：如果全量摘要连续失败 3 次（如 API 超时），系统自动放弃，不会无限重试。

---

### 各层协调设计

| 协调点 | 说明 |
|--------|------|
| Snip → Auto-Compact | Snip 告知 Auto-Compact "已释放多少 Token"，避免重复压缩 |
| Context Collapse → Auto-Compact | Context Collapse 先运行，如果够用，Auto-Compact 不触发 |
| 读时投影 | 不修改原始消息，只在 API 调用时动态计算视图 |

**每一层都在为下一层"减负"**。

---

### 设计启示

1. **能轻则轻，逐步加码**：先用代价最小的手段，实在不行再升级
2. **前三层零 API 开销**：只是"搬运"和"裁剪"数据，大部分场景够用了
3. **分层降级**：不是上来就用大锤（模型调用），而是先用小刀（规则删除、缓存层操作）
4. **成本意识**：每一层都考虑了 token 成本和模型调用成本
5. **压缩后恢复**：主动恢复最近文件，让模型无缝继续，用户体验几乎无感知

---

## 2. 工具系统：40+ 工具，零继承

### 反传统设计

传统 Agent 框架习惯写一个 `BaseTool` 基类然后继承。Claude Code **完全没有继承**，40 多个工具全是纯函数式的 `buildTool()` 工厂函数。

### ToolDef 接口设计

```typescript
type ToolDef<T> = {
  name: string
  description: string
  inputSchema: ZodSchema<T>           // Zod v4 做校验 + 自动生成 JSON Schema
  call(input: T, ctx: ToolUseContext): AsyncGenerator<...>
  isReadOnly(): boolean
  getPermissions(): ToolPermission[]
  renderToolUse?(input: T): ReactNode  // 直接渲染到终端
  getToolUseSummary?(input, result): string  // 压缩上下文时的摘要
}
```

### 设计亮点

| 字段 | 作用 | 设计价值 |
|------|------|----------|
| `inputSchema` | Zod v4 校验 + 自动生成 JSON Schema | 类型安全 + 模型理解 |
| `isReadOnly()` | 声明工具是否只读 | 权限控制前置 |
| `getPermissions()` | 返回所需权限列表 | 细粒度权限管理 |
| `renderToolUse` | ReactNode 直接渲染到终端 | 工具执行过程可视化 |
| `getToolUseSummary` | 生成上下文压缩用的摘要 | **与上下文压缩联动** |

### 完全自包含

每个工具文件包含：
- ✅ Schema 定义
- ✅ 权限声明
- ✅ 执行逻辑
- ✅ UI 渲染
- ✅ 压缩摘要

**没有全局注册表**，每个 session 动态组装工具池：
- 静态工具
- MCP 工具
- Agent 定义的工具

混在一起用，无需继承层级。

### 为什么不用继承？

1. **组合优于继承**：工具能力通过接口组合，而非类继承
2. **动态组装**：每个 session 的工具池都不一样，静态继承无法满足
3. **权限隔离**：工具自声明权限，不依赖外部注册表
4. **上下文压缩联动**：`getToolUseSummary` 直接对接四层压缩机制

---

## 3. Plan Mode：先规划，再执行

### 核心思想

复杂任务应该先**规划再执行**，避免方向跑偏、浪费精力。

这不是一个独立的框架，而是在同一个 Tool-Use Loop 中通过 `EnterPlanMode` 和 `ExitPlanMode` **两个工具**实现的。

### 三步流程

```
┌─────────────────────────────────────────────────────────┐
│  Step 1: 触发 Plan Mode                                  │
│  ├── 模型自主判断（复杂任务）→ 调用 EnterPlanMode         │
│  ├── 简单任务（修 typo、加 log）→ 不进入                  │
│  └── 用户手动触发（Shift+Tab）                           │
│                         ↓                               │
│  Step 2: 只读探索 + 设计方案                              │
│  ├── 权限降为只读（Read/Grep/Glob ✅，Write/Bash ❌）      │
│  ├── 探索代码库                                          │
│  ├── 把计划写入 .claude/plans/ 目录                       │
│  └── 防走神机制：每 5 轮系统提醒"还在 Plan Mode"          │
│                         ↓                               │
│  Step 3: 用户审批后实施                                   │
│  ├── 模型调用 ExitPlanMode                               │
│  ├── 用户确认                                            │
│  └── 权限恢复，按计划实施                                 │
└─────────────────────────────────────────────────────────┘
```

### 关键设计：工具即能力

Plan Mode 不是一种特殊的"模式切换"，而只是调用了两个工具。

```typescript
// 进入 Plan Mode
EnterPlanMode({ reason: "这是一个复杂重构任务" })

// 退出 Plan Mode
ExitPlanMode({ plan: "...", needsApproval: true })
```

**设计价值**：
- 对模型来说，和调用 `Read` 一样自然
- 引擎层无需特殊处理，`query()` 仍然是简单的 `while(true)` 循环
- 权限控制通过工具层面实现（只读工具的 `isReadOnly()`）

### 防走神机制

**问题**：长对话中模型可能"忘记"自己还在 Plan Mode，手痒去改代码。

**解决**：每 5 轮对话，系统自动插入系统消息：

> "你现在还在 Plan Mode，只能使用只读工具。不要执行写入操作。"

**本质**：用**被动提醒**代替**强制限制**，保持灵活性。

### 为什么效果好？

1. **延迟执行**：先想后做，避免"边想边做"导致的思路混乱
2. **用户可控**：关键节点（进入/退出）都要用户确认
3. **权限分级**：Plan Mode 自动降级为只读，防止误操作
4. **极简实现**：没有复杂的模式状态机，两个工具搞定

---

## 4. Prompt 设计：专用工具优于 Bash

### 核心规则

> "当有专用工具可用时，不要用 Bash 来执行命令。"

| 操作 | ✅ 专用工具 | ❌ Bash 命令 |
|------|-----------|-------------|
| 读文件 | `Read` | `cat`、`head`、`tail` |
| 编辑文件 | `Edit` | `sed`、`awk` |
| 创建文件 | `Write` | `echo > ` 重定向 |
| 搜索文件 | `Glob` | `find`、`ls` |
| 搜索内容 | `Grep` | `grep`、`rg` |

### 为什么？可审查性

**对比体验**：

```
专用工具 Read 调用:          Bash 命令:
┌─────────────────────┐     ┌─────────────────────┐
│ 🤖 读取 src/index.ts │     │ $ cat src/index.ts  │
│                     │     │                     │
│ (清晰展示操作意图)    │     │ (一大坨输出)         │
└─────────────────────┘     └─────────────────────┘
     ↑ 明确知道 Agent 在做什么
```

### 为什么？安全性

- **专用工具**：有专用权限检查（如 `Read` 检查文件路径是否在允许范围）
- **Bash 命令**：无保护，模型可以执行任意命令

### 设计动机

这条规则的设计动机值得深思。技术上完全可以直接让模型用 `cat`、`sed`，但 Claude Code 选择了**约束**。

**核心洞察**：

1. **可见性**：专用工具让 Agent 行为对用户透明
2. **可审查**：每个操作都有明确的意图和范围
3. **安全性**：权限控制前置到工具层，而非依赖事后审计
4. **用户体验**：清晰的 UI 反馈，用户知道 Agent 在做什么

### 不只是体验问题

> "所以「用专用工具而不是 Bash」不仅是体验问题，更是**安全问题**。

这条规则把**安全性**内嵌到 Prompt 设计中，通过约束模型行为来降低风险。

---

## 5. Prompt 拼装：环境感知与三级缓存

### 环境信息注入

每次对话开始时，Claude Code 把当前环境信息注入 System Prompt：

```markdown
# 环境信息
- 主工作目录：/Users/you/my-project
- 是否为 Git 仓库：是
- 操作系统平台：darwin (macOS)
- Shell 类型：zsh
- 当前模型：Claude Opus 4.6 (1M context)
- 知识截止日期：2025 年 5 月
```

**为什么重要？**

没有这些信息，模型可能会：
- 在 macOS 上执行 `apt-get install`
- 在 zsh 环境里用 bash 语法
- 在 `/tmp` 下操作而不知道用户工作目录

**本质**：让模型知道自己"在哪里"。

---

### 分割线与三级缓存

System Prompt 组装后的结构：

```
┌─────────────────────────────────────────────────┐
│  [角色定义] 你是一个交互式代理...               │  ← 所有用户完全一样
│  [安全红线] 重要：允许协助已授权的安全测试...   │  ← 所有用户完全一样
│  [行为准则] 一般来说，不要...                   │  ← 所有用户完全一样
│  [操作安全] 仔细考虑操作的可逆性...             │  ← 所有用户完全一样
│  [工具使用] 当有专用工具可用时...               │  ← 所有用户完全一样
├────── __SYSTEM_PROMPT_DYNAMIC_BOUNDARY__ ────────┤  ← 分割线
│  [环境信息] 主工作目录: /Users/you/my-project  │  ← 每个用户不一样
│  [CLAUDE.md] 本项目使用 TypeScript + Jest...   │  ← 每个项目不一样
│  [记忆指令] 你有一个持久记忆系统...             │  ← 每次对话可能不一样
│  [MCP 指令] 你已连接 GitHub MCP server...      │  ← 每个用户不一样
└─────────────────────────────────────────────────┘
```

**分割线的作用**：

| 层级 | 内容 | 缓存策略 |
|------|------|----------|
| 分割线上 | 角色定义、安全红线、行为准则... | **全局缓存**：跨组织跨用户共享 |
| 分割线下 | 环境信息、CLAUDE.md、记忆、MCP | **动态生成**：因人而异 |

**为什么分？** —— 成本优化

Claude API 的 **Prompt Cache 机制**：
- 如果两次请求的 Prompt 前缀完全相同 → 复用上次计算结果
- **费用降低 90%**

对于几万 Token 的 System Prompt，缓存命中与否意味着每次请求**几美分 vs 几美元**的差距。

---

### 三级缓存体系

```
┌────────────────────────────────────────────┐
│  全局缓存（Global Cache）                   │
│  ├── 分割线之上内容                          │
│  └── 跨组织跨用户共享                        │
├────────────────────────────────────────────┤
│  组织缓存（Org Cache）                        │
│  ├── 同一组织内跨会话共享                    │
│  └── 如企业级 MCP 配置                       │
├────────────────────────────────────────────┤
│  会话缓存（Session Cache）                    │
│  ├── 同一个 Section 在一次会话内只计算一次   │
│  └── 如动态生成的记忆内容                    │
└────────────────────────────────────────────┘
```

**每一级都在帮 API 省钱。**

---

### 设计启示

1. **静态 vs 动态分离**：把不变的部分放在分割线之上，最大化缓存命中率
2. **顺序很重要**：前缀匹配，所以分割线位置决定了缓存范围
3. **成本敏感设计**：将成本优化内嵌到架构设计中
4. **环境感知**：让模型知道"在哪里"，避免跨平台错误

---

## 6. 记忆系统：四类型 + 索引架构

### 为什么不用向量数据库？

业界常见方案：用向量数据库存 embedding，做相似度检索。

Claude Code **没有这么做**。为什么？

| 场景 | 向量检索效果 | 实际需要 |
|------|-----------|---------|
| "不要 mock 数据库" | 差：会匹配一堆含"数据库"的无关内容 | 精确匹配行为规则 |
| 用户偏好 | 差：难以用相似度衡量 | 结构化指令 |

**核心洞察**：Agent 需要记的是**结构化行为指令**，不是相似文档片段。

---

### 四类型封闭集合

```typescript
const MEMORY_TYPES = [
  'user',      // 用户画像：角色、偏好、知识水平
  'feedback',  // 行为反馈：该做什么、不该做什么
  'project',   // 项目动态：在做什么、截止日期
  'reference', // 外部指针：哪里能找到什么信息
] as const
```

**为什么只有四种？**

无约束的记忆会膨胀成垃圾堆。限定四种类型，逼 Agent 做**分类决策**——每存一条必须想清楚"这属于哪一类"。

| 类型 | 存储内容 | 关键要求 |
|------|----------|----------|
| `user` | 用户是谁、擅长什么、知识水平 | 因人而异 |
| `feedback` | 用户说过的"不要"和"继续保持" | **必须记 Why + How to apply** |
| `project` | 谁在做什么、截止日期、重要决策 | **相对日期转绝对日期** |
| `reference` | 去哪找信息（Bug 追踪、Grafana 地址） | 不需要内容，只需要位置 |

**Feedback 类型的特殊要求**：

```markdown
**Why:** 上季度 mock 测试全通过但生产迁移失败
**How to apply:** 写测试时始终连接真实数据库
```

光记规则不够。遇到边缘情况（如纯单元测试），Agent 需要根据 **Why** 判断规则是否适用。

---

### 不记什么（排除清单）

Claude Code **明确规定了什么不该存**：

| 不存 | 原因 |
|------|------|
| 代码模式、架构、文件结构 | `grep`/`git`/`CLAUDE.md` 能查到 |
| Git 历史、最近改动 | `git log`/`git blame` 才是权威 |
| 调试方案、修复方法 | 修复已在代码里，commit 消息有上下文 |
| CLAUDE.md 已有内容 | 避免重复 |
| 临时任务、当前对话上下文 | 会话级信息，无需跨会话保持 |

**核心原则**：

> 可以从当前代码推导出来的信息，**一律不存**。

代码是"活的"，记忆是"死的"。如果记忆说"AuthService 在 src/auth.ts:42"，但代码已重构，这条记忆就变成了**权威的错误**。

---

### 存储架构：索引 + 独立文件

**每条记忆的独立文件**：

```markdown
---
name: no-mock-database
description: 集成测试必须用真实数据库
type: feedback
---

集成测试必须使用真实数据库。

**Why:** 上季度 mock 测试通过但生产迁移失败
**How to apply:** 写测试时始终连接真实数据库
```

**MEMORY.md 索引文件**（轻量目录）：

```markdown
- [No Mock Database](feedback_no_mock.md) — tests must use real DB
- [User Preferences](user_preferences.md) — prefers terse responses
- [Auth Rewrite](project_auth.md) — driven by compliance
```

**硬性上限**（双重检查）：

```typescript
const MAX_ENTRYPOINT_LINES = 200
const MAX_ENTRYPOINT_BYTES = 25_000  // 25KB
```

同时检查行数和字节数——有人可能写 199 行，每行 500 字，字节数爆了但行数没超。

**架构关键设计**：

| 组件 | 加载策略 | 目的 |
|------|---------|------|
| MEMORY.md 索引 | **始终加载** | Agent 知道有哪些记忆可用 |
| 独立记忆文件 | **按需加载** | 不撑爆上下文 |

解决了经典矛盾：全塞会占满上下文，完全不塞 Agent 不知道有哪些。

---

### 召回机制：Sonnet 当秘书

**三步流程**：

```
Step 1: 扫描（只读前 30 行）
  ↓
Step 2: Sonnet 选择（≤5 条）
  ↓
Step 3: 加载完整内容注入上下文
```

**Step 1：扫描头部信息**

```typescript
// 只读每个文件的前 30 行（frontmatter 区域）
const { content } = await readFileInRange(filePath, 0, 30)
const { frontmatter } = parseFrontmatter(content)
// 返回：filename, description, type, mtimeMs
```

200 个文件也只需读 6000 行，开销很小。

**Step 2：Sonnet 做选择**

把清单发给 Sonnet：

```typescript
const result = await sideQuery({
  model: getDefaultSonnetModel(),
  system: '从列表中选出最多 5 条最相关的记忆...',
  messages: [{ role: 'user', content: `用户问题: ${query}\n\n可用记忆:\n${manifest}` }],
  max_tokens: 256,  // 只需返回文件名列表
})
```

Sonnet 返回文件名列表（如 `["feedback_no_mock.md"]`），不是内容本身。

**Step 3：加载完整内容**

读取选中记忆的完整内容，作为 `<system-reminder>` 注入。

---

### 关键优化

**1. 记忆陈旧度检测**

```typescript
function memoryFreshnessText(mtimeMs: number): string {
  const d = memoryAgeDays(mtimeMs)
  if (d <= 1) return ''  // 今天/昨天不加警告
  return (
    `这条记忆已有 ${d} 天。记忆是某个时间点的观察，` +
    `关于代码行为或 file:line 的断言可能已过时。` +
    `在当作事实引用前，请先对照当前代码验证。`
  )
}
```

30 天前的记忆说"AuthService 在 src/auth.ts:42"，但代码早改了。陈旧度警告提醒模型**先验证再引用**。

**2. 并行预取**

```typescript
// query.ts — 用户提交后立刻启动，与主模型并行
using pendingMemoryPrefetch = startRelevantMemoryPrefetch(
  state.messages,
  state.toolUseContext,
)
```

Sonnet 延迟通常几百毫秒，等 Opus 响应回来时，记忆选择已完成。**零额外延迟**。

**3. 上下文过滤**

如果用户正在用某个 MCP 工具：
- ❌ 该工具的**使用文档**被过滤（噪声）
- ✅ 该工具的**bug/注意事项**仍被选中（信号）

**正在用的时候，恰恰最需要知道坑在哪里**。

---

### 设计哲学三句话

1. **记该记的，不记能推导的** —— 四类型封闭集合 + 排除清单，防止记忆膨胀成垃圾堆

2. **存索引，按需加载详情** —— MEMORY.md 常驻 System Prompt，独立文件用到才加载，既让 Agent 知道有哪些，又不撑爆上下文

3. **用小模型做秘书，大模型做决策** —— Sonnet 并行预取 + 选择记忆，Opus 只管决策，实现零延迟、低成本、高可靠
