# CLAUDE.md 编写指南

> 来源：[笙囧同学 - 知乎](https://www.zhihu.com/question/1979609139266213083/answer/2009919949922133046)，两个月实战经验总结。

## 核心认知

CLAUDE.md 不是命令手册，是**给高级工程师的入职须知**。不是给机器人编程，而是给一个能力强、但对你项目一无所知的工程师写上下文。

## 第一原则：短

- 3000字 → 规则被大面积无视
- 800字左右 → 最佳平衡点
- **别超过1000字**，信息越多，每条规则的注意力越稀释

砍不动就区分"必须知道的"和"最好知道的"，只留前者。

## 有效结构（四区块）

| 区块 | 内容 |
|------|------|
| **项目概况** | 技术栈、包结构、一句话讲清 |
| **核心约定** | 项目特有的、Claude 推断不出来的关键约定 |
| **开发命令** | 写死具体命令，不用模糊描述 |
| **常见错误** | 最有价值的区块，持续迭代 |

## 最大发现："不要做什么" 比 "要做什么" 更有效

Claude 倾向于用训练数据里最常见的模式（如 try/catch）。在"常见错误"区块里明确写 **"不要用X → 用Y"**，遵守率显著提高。

- "用 Result 模式" → 有时遵守有时忘
- "不要用 try/catch → 用 Result" → 遵守率明显提高

"不要做X"是一条更具体、更容易执行的指令，在 Claude 脑子里建了一道明确的红线。

这个区块应**持续迭代**：每次发现 Claude 反复犯同一个错就加一条，不超过10条。

## 五个具体技巧

1. **路径和命令写死** — `pnpm -F @app/api test` 而非"请运行测试"
2. **设计模式指向参考文件** — `参考：src/shared/result.ts`，代码本身就是最好的规范
3. **写清包依赖关系** — 让 Claude 知道改 A 会影响 B
4. **别写它本来就会的事** — 只写"不明确说就一定会搞错"的东西（不需要说"取有意义的变量名"）
5. **边用边迭代** — 活文档，每次发现问题就改一条，不是一次写完

## 最值钱的经验

> **代码库本身就是最大的 CLAUDE.md。**

- 代码风格一致 → Claude 自动模仿，不需要多写规则
- 代码风格混乱 → 写再多规则也没用，因为代码里信号是矛盾的
- **一个写得好的参考文件，胜过十条文字规则** — Claude 更擅长"照葫芦画瓢"而非"理解抽象规则后推导"

与其在 CLAUDE.md 里用文字描述规范，不如在代码库里维护几个"标杆文件"，然后在 CLAUDE.md 里指向它们。

## 示例模板

```markdown
# 项目概况

TypeScript monorepo，pnpm workspace，Node 20。
三个包：
- @app/core — 业务核心逻辑，零外部依赖
- @app/api — Fastify REST API，依赖 core
- @app/web — Next.js 14 App Router，依赖 core

# 核心约定

- TypeScript 严格模式，禁止 @ts-ignore
- 错误处理统一用 Result<T, E> 模式
  参考实现：packages/core/src/result.ts
  使用示例：packages/api/src/services/user-service.ts
- 数据库操作走 Repository 模式，禁止在 repository 层之外写 SQL

# 开发命令

pnpm dev              # 启动所有包的开发环境
pnpm -F @app/api test # 仅跑 api 测试
pnpm typecheck        # 全量类型检查

# 测试

框架：Vitest
规则：
- 测试文件与源码同目录：foo.ts → foo.test.ts
- 禁止调用真实外部服务，一律 mock

# 常见错误（重要）

- 不要用 try/catch 处理业务错误 → 用 Result<T, E>
- 不要用 any → 用 unknown + 类型守卫
- 不要用 console.log → 用 packages/core/src/logger.ts
- 不要在 @app/core 里引入任何外部包
- 不要改 core 而不检查 api 和 web 是否受影响
- 不要写 eslint-disable 绕过规则，修掉问题本身
```
