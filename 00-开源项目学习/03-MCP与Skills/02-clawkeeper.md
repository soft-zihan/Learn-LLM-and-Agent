# ClawKeeper：OpenClaw 全面安全防护框架

> 📂 **项目**：`clawkeeper`（SafeAI-Lab-X/ClawKeeper）  
> 🎯 **学习价值**：掌握 AI Agent 安全防护的三层架构设计  
> ⏱️ **预计学习时间**：2-3 小时  
> ⭐ **Stars**: 1000+

---

## 目录

- [一、项目概述](#一项目概述)
- [二、三层防护架构](#二三层防护架构)
- [三、Skill 层防护](#三skill-层防护)
- [四、Plugin 层防护](#四plugin-层防护)
- [五、Watcher 层防护](#五watcher-层防护)
- [六、核心安全功能](#六核心安全功能)
- [七、快速上手](#七快速上手)
- [八、安全对比分析](#八安全对比分析)
- [九、学习建议](#九学习建议)

---

## 一、项目概述

### 1.1 什么是 ClawKeeper？

ClawKeeper 是为 **OpenClaw** 等自主 Agent 系统设计的**综合实时安全框架**，号称"The Norton for OpenClaw"（OpenClaw 的诺顿杀毒）。

**核心定位**：
- 全面保护 Agent 运行安全
- 实时威胁防护
- 三层互补架构

### 1.2 为什么需要 ClawKeeper？

自主 Agent 系统面临的安全威胁：

```
安全威胁类型
├── Prompt 注入攻击
├── 凭证泄露
├── 代码注入
├── 危险命令执行（rm -rf / 等）
├── 目标漂移
├── 不安全循环
└── 第三方插件恶意行为
```

ClawKeeper 提供统一的防护机制，覆盖从指令层到系统层的全方位安全。

---

## 二、三层防护架构

ClawKeeper 的核心创新在于**三层互补架构**：

```
ClawKeeper 三层防护
│
├── 1. Skill 层（指令层）
│   ├── 注入安全策略到 Agent 上下文
│   ├── 平台特定安全指南（Windows/飞书等）
│   └── 结构化 Markdown 文档 + 脚本
│
├── 2. Plugin 层（运行时层）
│   ├── 配置审计与加固
│   ├── 威胁检测
│   └── 行为监控
│
└── 3. Watcher 层（系统层）
    ├── 独立监控 Agent（解耦）
    ├── 实时执行干预
    ├── 本地/云端部署
    └── 人与 AI 的监管分离
```

**关键设计**：Watcher 层是**系统无关**的，可以集成到不同的 Agent 平台，实现任务执行与安全强制的**监管分离**。

---

## 三、Skill 层防护

### 3.1 工作原理

通过结构化的 Markdown 文档和脚本，将安全策略直接注入 Agent 上下文。

### 3.2 示例：Windows 安全指南

```bash
# 安装
cd clawkeeper-skill/skills/windows-safety-guide
./scripts/install.ps1

# 使用
# 指示 OpenClaw：
# "Please use the windows-safety-guide skill to enforce 
#  behavior security policies, configuration protection, 
#  and enable nightly security audits."
```

### 3.3 示例：飞书安全指南

```bash
# 安装
cd clawkeeper-skill/skills/feishu-safety-guide
bash scripts/install.sh

# 使用
# "Please use the feishu-safety-guide skill to enforce 
#  message protection, credential security, and enable 
#  periodic security reporting in Feishu (Lark)."
```

---

## 四、Plugin 层防护

### 4.1 功能

运行时强制执行的插件，提供：
- 配置审计
- 威胁检测
- 行为监控

### 4.2 安装与使用

```bash
# Linux/macOS
cd clawkeeper-plugin
bash install.sh

# Windows
cd clawkeeper-plugin
./install.ps1

# 验证安装
npx openclaw clawkeeper audit
```

### 4.3 核心命令

```bash
# 安全审计
npx openclaw clawkeeper audit

# 权限管理
openclaw clawkeeper permission list
openclaw clawkeeper permission allow <tool> <fingerprint>
openclaw clawkeeper permission deny <tool> <fingerprint>
```

---

## 五、Watcher 层防护

### 5.1 创新设计

Watcher 是**独立的、解耦的治理层**，提供：
- 运行时监控
- 执行控制
- 不与 Agent 内部逻辑耦合

### 5.2 安装与部署

```bash
# 1. 安装依赖
pnpm install

# 2. 构建
cd clawkeeper
npm install
npm run build
npm link

# 3. 初始化模式
clawkeeper init remote  # 云端模式
clawkeeper init local   # 本地模式

# 4. 启动 Watcher
clawkeeper remote gateway run  # 远程治理
clawkeeper local gateway run   # 本地治理
```

### 5.3 两种部署模式

| 模式 | 适用场景 | 特点 |
|------|---------|------|
| **Local** | 个人部署 | 本地运行，低延迟 |
| **Remote** | 企业/内网 | 云端运行，集中管理 |

---

## 六、核心安全功能

### 6.1 全面安全扫描

定期扫描运行时环境、依赖和工作区，在威胁发生前提供清晰的风险警报。

### 6.2 实时威胁防护

实时评估 AI 行为，阻止高风险行为：
- Prompt 注入
- 凭证泄露
- 代码注入

### 6.3 执行门控（v1.1 新增）

```javascript
// exec-gate：基于正则的危险命令检测
// 阻止破坏性 shell 命令
const dangerousPatterns = [
  'rm -rf /',           // 删除根目录
  'fork bombs',         // Fork 炸弹
  'curl | sh',          // 管道执行
  'disk wipes'          // 磁盘擦除
];
```

### 6.4 路径保护（v1.1 新增）

保护敏感路径，防止 Agent 读取、写入或删除：
- `~/.ssh/**`
- `~/.aws/credentials`
- `/etc/shadow`

### 6.5 输入验证器（v1.1 新增）

轻量级 JSON Schema 子集验证器，拒绝格式错误的工具输入：
- 缺少字段
- 类型错误
- 超大字符串
- NUL 字节

### 6.6 预算保护（v1.1 新增）

滚动窗口 token 预算控制，超过限制时停止 Agent 执行。

### 6.7 权限存储（v1.1 新增）

持久的允许/拒绝决策，支持会话级和永久级权限。

---

## 七、快速上手

### 7.1 最简安装

```bash
# 克隆仓库
git clone https://github.com/SafeAI-Lab-X/ClawKeeper.git
cd ClawKeeper

# 安装 Skill
cd clawkeeper-skill/skills/windows-safety-guide
./scripts/install.ps1

# 安装 Plugin
cd ../clawkeeper-plugin
bash install.sh

# 验证
npx openclaw clawkeeper audit
```

### 7.2 完整部署

```bash
# 1. 安装所有三层
bash install-all.sh

# 2. 初始化 Watcher
clawkeeper init local

# 3. 启动
clawkeeper local gateway run
```

---

## 八、安全对比分析

ClawKeeper 对比其他 OpenClaw 安全仓库：

| 安全机制 | ClawKeeper | 其他方案 |
|---------|-----------|---------|
| Skill 层防护 | ✅ | ❌ |
| Plugin 层防护 | ✅ | 部分 |
| Watcher 层防护 | ✅ | ❌ |
| 实时威胁防护 | ✅ | 部分 |
| 执行门控 | ✅ | ❌ |
| 路径保护 | ✅ | ❌ |
| 输入验证 | ✅ | ❌ |
| 预算控制 | ✅ | ❌ |
| 权限管理 | ✅ | 部分 |

**结论**：ClawKeeper 提供了最全面的安全防护，三层架构互补覆盖全生命周期。

---

## 九、学习建议

### 9.1 学习路径

```
1. 理解三层架构设计
   ↓
2. 学习 Skill 层安全策略编写
   ↓
3. 掌握 Plugin 层配置与命令
   ↓
4. 部署 Watcher 层监控系统
   ↓
5. 实战：集成到自己的 OpenClaw 系统
```

### 9.2 关键概念

- **监管分离**：任务执行与安全强制分开，提高安全性
- **防御纵深**：三层防护，即使一层被突破还有其他层
- **实时监控**：不是事后审计，而是实时拦截

### 9.3 实战项目

建议实战：
1. 为你的 OpenClaw 系统安装 ClawKeeper
2. 编写自定义 Skill 安全策略
3. 配置 Watcher 监控规则
4. 模拟攻击测试防护效果

---

## 参考链接

1. GitHub: https://github.com/SafeAI-Lab-X/ClawKeeper
2. Paper: https://arxiv.org/abs/2603.24414
3. OpenClaw: https://github.com/openclaw/openclaw
4. Skill 文档: clawkeeper-skill/README.md
5. Plugin 文档: clawkeeper-plugin/README.md
6. Watcher 文档: clawkeeper-watcher/README.md

---

> 💡 **学习建议**：先理解三层架构的设计理念，再逐层学习实现细节。重点掌握 Watcher 层的监管分离思想，这是 AI Agent 安全的核心创新。
