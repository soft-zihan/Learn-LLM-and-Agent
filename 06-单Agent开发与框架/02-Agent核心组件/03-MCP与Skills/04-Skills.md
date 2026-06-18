# Skills 开发实战

> 📅 **更新时间**: 2026-06-17  

---

## 目录

- [1. Skills 开发最佳实践](#1-skills-开发最佳实践)
  - [2.1 什么是 Skills](#21-什么是-skills)
  - [2.2 Skills 的两种范式](#22-skills-的两种范式)
  - [2.3 Skills 设计原则](#23-skills-设计原则)
  - [2.4 Skills 测试](#24-skills-测试)
- [2. 安全与权限控制](#2-安全与权限控制)
- [3. 生产部署](#3-生产部署)

---

## 1. Skills 开发最佳实践

### 2.1 什么是 Skills

#### Skills 定义

**Skills** 是在 MCP 之上的**更高层次的抽象**，它将多个 Tools、Resources、Prompts 组合成可复用的功能模块。

```
Skill = Tools + Resources + Prompts + 配置 + 文档

示例：
代码审查 Skill：
- Tools: analyze_code, suggest_fixes, apply_changes
- Resources: coding_standards, best_practices
- Prompts: review_template, feedback_template
- 配置: 语言特定规则、团队规范
- 文档: 使用说明、示例
```

#### Skills vs Tools

| 特性 | Tools | Skills |
|------|-------|--------|
| **粒度** | 单一功能 | 功能集合 |
| **复用性** | 低 | 高 |
| **组合性** | 需手动 | 内置 |
| **配置** | 参数 | 配置文件 |
| **文档** | 描述 | 完整文档 |
| **测试** | 单元测试 | 集成测试 |
| **版本控制** | 无 | 语义化版本 |
| **分发** | 代码 | 包管理器 |

**类比**：
```
Tools = 函数
Skills = 库/模块

就像：
- requests.get() 是函数
- requests 库是 Skill
```

#### Skills 组合

```python
# Skill 组合示例
class CodeQualitySkill:
    """代码质量 Skill"""
    
    def __init__(self):
        self.tools = {
            "lint": LintTool(),
            "format": FormatTool(),
            "test": TestTool(),
            "review": ReviewTool()
        }
        self.resources = {
            "standards": CodingStandardsResource(),
            "metrics": MetricsResource()
        }
        self.prompts = {
            "review_template": ReviewPrompt(),
            "fix_template": FixPrompt()
        }
    
    async def analyze(self, code: str, language: str) -> dict:
        """完整的代码分析流程"""
        # 1. Lint
        lint_result = await self.tools["lint"].run(code, language)
        
        # 2. Format check
        format_result = await self.tools["format"].check(code, language)
        
        # 3. Review
        review_result = await self.tools["review"].run(
            code,
            language,
            lint_result,
            format_result
        )
        
        return {
            "lint": lint_result,
            "format": format_result,
            "review": review_result
        }
```

#### 复用机制

```python
# Skill 复用
from mcp_skills import SkillManager

# 安装 Skill
# pip install code-review-skill
# pip install testing-skill

manager = SkillManager()

# 加载 Skills
manager.load("code-review-skill")
manager.load("testing-skill")
manager.load("documentation-skill")

# 使用
review_skill = manager.get("code-review-skill")
result = await review_skill.review(code, language="python")

# 组合使用
results = await manager.execute_pipeline([
    ("code-review-skill", "review"),
    ("testing-skill", "generate_tests"),
    ("documentation-skill", "update_docs")
], context={"code": code, "language": "python"})
```

### 2.2 Skills 的两种范式

Skills 有两种主流实现范式，适用于不同场景：

#### 范式一：代码模块型 Skill

即上面描述的 `Skill = Tools + Resources + Prompts + 配置`，以 Python/TS 代码实现，通过 MCP 协议暴露。适合需要稳定接口、团队协作、版本管理的场景。

#### 范式二：程序性知识型 Skill

Skill 本质是**教 AI 如何做事的知识文档**，不依赖特定 MCP 工具。AI 读取 SKILL.md 后，利用自身已有的工具（Bash、文件读写、搜索等）独立完成任务。

**核心区别**：

| 特性 | 代码模块型 | 程序性知识型 |
|------|-----------|-------------|
| **本质** | 可执行代码 | Markdown 知识文档 |
| **运行方式** | 通过 MCP 协议调用 | AI 读取后自主执行 |
| **依赖** | 依赖 MCP Server 运行 | 独立于 MCP，只要有基础工具即可 |
| **维护成本** | 需维护代码 + 环境 | 只需编辑 Markdown |
| **适用场景** | 标准化接口、团队协作 | 个人工作流、知识封装 |
| **示例** | MCP Server 的 Tool 集合 | Qoder Skills、LLMInternSkill |

**程序性知识型 Skill 的文件结构**：

```text
my-skill/
├── SKILL.md                      # 主文件：工作流、决策树、规则
├── skill-references/             # 参考文档：按需加载，减少 token
│   ├── domain-knowledge.md       # 领域知识
│   ├── strategy.md               # 策略规则
│   └── error-handling.md         # 错误处理
├── templates/                    # 输出模板
│   └── report.md
└── examples/                     # 使用示例（可选）
```

**设计原则**：

1. **SKILL.md 保持轻量**（< 150 行）：只写决策树和核心规则，详细内容放 references
2. **渐进式披露**：SKILL.md 通过 `see skill-references/xxx.md` 引用，AI 按需读取，不一次性加载全部知识
3. **独立性**：Skill 不绑定特定 MCP 工具，描述"做什么、怎么做"，不限定"用什么工具做"
4. **Non-Negotiables**：明确列出不可违反的规则（如"不虚构内容"、"必须验证输出"）

**SKILL.md 基本结构**：

```markdown
---
name: my-skill
description: 一句话描述何时使用此 Skill
---

# MySkill

Use this Skill when...（触发条件）

Core rule: ...（核心规则）

## Inputs
（描述输入格式）

## Decision Tree
（模式判断：根据不同输入走不同流程）

## Main Workflow
1. Step one — reference `skill-references/xxx.md`
2. Step two
3. ...

## Output Files
（描述输出结构）

## Non-Negotiables
- 规则 1
- 规则 2
```

**典型案例 — LLMInternSkill**：

```text
LLMInternSkill/
├── SKILL.md                        # 139 行，简历优化工作流
├── skill-references/               # 10 个参考文件
│   ├── jd-analysis.md              # JD 分析规则
│   ├── truth-boundary.md           # 真实性边界（不虚构）
│   ├── evidence-contract.md        # 证据契约
│   ├── resume-polish.md            # 润色规则
│   └── ...
└── templates/
    └── resume-latex/               # LaTeX 模板
```

SKILL.md 只有 139 行，但通过引用 10 个 reference 文件，覆盖了完整的简历优化工作流。AI 读取 SKILL.md 后，根据用户需求按需加载对应的 reference，而不是一次加载所有内容。

**关键：reference 文件是"子程序"，不是"知识库"**：

reference 文件容易被误写成百科式的知识文档。正确的写法是**可执行的子程序**——AI 读取后能直接按步骤操作。

| 错误写法（知识百科） | 正确写法（可执行子程序） |
|---|---|
| "知乎的赞数字段是 liked_count" | "1. 读取 liked_count 字段 2. 转为 int 3. 如果 > 100 则 +20 分" |
| "去重可以用 Jaccard 相似度" | "1. 对每对记录计算 Jaccard 2. > 0.8 视为重复 3. 保留质量分更高的" |
| "常见错误有登录超时、网络异常" | "1. 读取退出码 2. 如果 = 0 转步骤 3 3. 如果 = 1 读取 stderr 转步骤 4" |

每个 reference 文件应包含：

```text
目标：一句话说清楚这个文件做什么
前提条件：执行前必须确认什么
步骤：
  1. 第一步做什么 → 判断条件 → 转哪个步骤
  2. 第二步做什么
  ...
输出：完成后向用户报告什么
```

完整的 Skill 还应包含：

- **`templates/intake.md`**：信息不足时的追问模板（只问缺失的，不重复问）
- **`examples/`**：完整的输入 → 输出示例，校准 AI 的输出质量和格式
- **`evals/`**：行为测试用例，验证 Skill 在边界情况下是否正确（空结果、无效输入、环境异常等）

### 2.3 Skills 设计原则

#### 单一职责

```python
# ❌ 错误示例：一个 Skill 做太多事情
class MegaSkill:
    """什么都做的 Skill"""
    
    async def handle(self, request: str) -> str:
        # 代码审查
        # 测试生成
        # 文档更新
        # 部署
        # 监控
        # ... 太多职责
        pass

# ✅ 正确示例：单一职责
class CodeReviewSkill:
    """只做代码审查"""
    
    async def review(self, code: str, language: str) -> dict:
        """审查代码质量"""
        return analyze_quality(code, language)
    
    async def suggest_fixes(self, issues: list) -> list:
        """建议修复方案"""
        return generate_fixes(issues)

class TestGenerationSkill:
    """只做测试生成"""
    
    async def generate_tests(self, code: str) -> str:
        """生成测试用例"""
        return create_tests(code)
```

#### 明确接口

```python
# ✅ 清晰的接口
from typing import Protocol, runtime_checkable

@runtime_checkable
class CodeReviewSkillProtocol(Protocol):
    """代码审查 Skill 接口"""
    
    async def review(
        self,
        code: str,
        language: str,
        options: dict | None = None
    ) -> dict:
        """
        审查代码
        
        Args:
            code: 源代码
            language: 编程语言
            options: 审查选项
        
        Returns:
            审查结果，包含：
            - issues: 问题列表
            - score: 质量评分
            - suggestions: 改进建议
        """
        ...

# 实现接口
class PythonCodeReviewSkill:
    """Python 代码审查 Skill"""
    
    async def review(
        self,
        code: str,
        language: str,
        options: dict | None = None
    ) -> dict:
        assert language == "python", "只支持 Python"
        
        # 实现审查逻辑
        issues = analyze_python(code)
        score = calculate_score(issues)
        suggestions = generate_suggestions(issues)
        
        return {
            "issues": issues,
            "score": score,
            "suggestions": suggestions
        }
```

#### 错误处理

```python
# ✅ 完善的错误处理
class CodeReviewSkill:
    """代码审查 Skill"""
    
    async def review(self, code: str, language: str) -> dict:
        # 1. 参数验证
        if not code:
            raise ValueError("代码不能为空")
        
        if not language:
            raise ValueError("语言不能为空")
        
        if language not in SUPPORTED_LANGUAGES:
            raise UnsupportedLanguageError(
                f"不支持的语言: {language}，"
                f"支持: {SUPPORTED_LANGUAGES}"
            )
        
        try:
            # 2. 执行审查
            result = await self._analyze(code, language)
            
            # 3. 返回结果
            return result
        
        except SyntaxError as e:
            # 语法错误 - 用户问题
            raise CodeReviewError(f"代码语法错误: {e}") from e
        
        except TimeoutError as e:
            # 超时 - 系统问题
            raise CodeReviewError("审查超时，请稍后重试") from e
        
        except Exception as e:
            # 未知错误
            raise CodeReviewError(f"审查失败: {e}") from e

# 自定义异常
class CodeReviewError(Exception):
    """代码审查异常"""
    pass

class UnsupportedLanguageError(CodeReviewError):
    """不支持的语言异常"""
    pass
```

#### 文档完善

```python
# ✅ 完善的文档
class CodeReviewSkill:
    """
    代码审查 Skill
    
    提供代码质量分析、最佳实践检查、改进建议生成等功能。
    
    支持的编程语言：
    - Python
    - TypeScript
    - JavaScript
    - Go
    
    功能：
    - 代码质量评分
    - 问题检测（语法、逻辑、性能、安全）
    - 最佳实践检查
    - 改进建议生成
    
    使用示例：
    
    ```python
    from code_review_skill import CodeReviewSkill
    
    skill = CodeReviewSkill()
    
    # 审查代码
    result = await skill.review(
        code="def hello(): print('world')",
        language="python"
    )
    
    print(f"评分: {result['score']}")
    print(f"问题: {result['issues']}")
    print(f"建议: {result['suggestions']}")
    ```
    
    配置：
    
    ```yaml
    code_review:
      language: python
      strict_mode: true
      ignore_rules:
        - C0114  # 缺少模块文档
        - C0115  # 缺少类文档
      custom_rules:
        - path: ./rules/custom.py
    ```
    
    参考：
    - [PEP 8 - Python 风格指南](https://pep8.org/)
    - [Clean Code](https://www.amazon.com/Clean-Code-Handbook-Software-Craftsmanship/dp/0132350882)
    """
    
    async def review(self, code: str, language: str) -> dict:
        """
        审查代码
        
        Args:
            code: 源代码字符串
            language: 编程语言名称（python/typescript/javascript/go）
        
        Returns:
            dict: 审查结果，包含：
                - score (float): 质量评分 (0-100)
                - issues (list[dict]): 问题列表，每个问题包含：
                    - line (int): 行号
                    - column (int): 列号
                    - severity (str): 严重程度 (error/warning/info)
                    - message (str): 问题描述
                    - rule (str): 规则 ID
                - suggestions (list[str]): 改进建议列表
                - metrics (dict): 代码度量，包含：
                    - lines (int): 总行数
                    - complexity (int): 圈复杂度
                    - maintainability (float): 可维护性指数
        
        Raises:
            ValueError: 参数无效
            UnsupportedLanguageError: 不支持的编程语言
            CodeReviewError: 审查过程出错
        
        示例：
        
        >>> result = await skill.review("def hello(): pass", "python")
        >>> result['score']
        75.0
        >>> len(result['issues'])
        1
        """
        # 实现...
```

### 2.4 Skills 测试

#### 单元测试

```python
"""
Skill 单元测试
"""
import pytest
from code_review_skill import CodeReviewSkill

@pytest.fixture
def skill():
    """创建 Skill 实例"""
    return CodeReviewSkill()

class TestCodeReviewSkill:
    """CodeReviewSkill 测试"""
    
    @pytest.mark.asyncio
    async def test_valid_python_code(self, skill):
        """测试有效的 Python 代码"""
        code = """
def add(a, b):
    return a + b
"""
        result = await skill.review(code, "python")
        
        assert "score" in result
        assert "issues" in result
        assert "suggestions" in result
        assert isinstance(result["score"], (int, float))
        assert isinstance(result["issues"], list)
        assert 0 <= result["score"] <= 100
    
    @pytest.mark.asyncio
    async def test_code_with_issues(self, skill):
        """测试有问题的代码"""
        code = """
def bad_function(x):
    if x==1:
        return True
    if x==2:
        return True
    if x==3:
        return True
"""
        result = await skill.review(code, "python")
        
        assert len(result["issues"]) > 0
        assert any(issue["severity"] == "warning" for issue in result["issues"])
    
    @pytest.mark.asyncio
    async def test_unsupported_language(self, skill):
        """测试不支持的语言"""
        code = "console.log('hello')"
        
        with pytest.raises(UnsupportedLanguageError):
            await skill.review(code, "ruby")
    
    @pytest.mark.asyncio
    async def test_empty_code(self, skill):
        """测试空代码"""
        with pytest.raises(ValueError, match="代码不能为空"):
            await skill.review("", "python")
    
    @pytest.mark.asyncio
    async def test_performance(self, skill):
        """测试性能"""
        code = "def hello():\n" + "    pass\n" * 1000
        
        import time
        start = time.time()
        await skill.review(code, "python")
        duration = time.time() - start
        
        assert duration < 5.0, "审查时间超过 5 秒"
```

#### 集成测试

```python
"""
Skill 集成测试
"""
import pytest
from mcp.client.session import ClientSession
from code_review_skill import CodeReviewSkill

@pytest.mark.asyncio
async def test_skill_with_mcp_server():
    """测试 Skill 与 MCP Server 集成"""
    # 启动 MCP Server
    server_process = await start_mcp_server()
    
    try:
        # 连接 Client
        async with ClientSession(...) as session:
            await session.initialize()
            
            # 使用 Skill
            skill = CodeReviewSkill()
            result = await skill.review("def hello(): pass", "python")
            
            # 验证结果
            assert result["score"] > 0
    
    finally:
        # 清理
        await server_process.terminate()

@pytest.mark.asyncio
async def test_skill_chaining():
    """测试 Skill 链式调用"""
    code = """
def calculate(items):
    total = 0
    for item in items:
        total += item
    return total
"""
    
    # 1. 代码审查
    review_skill = CodeReviewSkill()
    review_result = await review_skill.review(code, "python")
    
    # 2. 测试生成
    test_skill = TestGenerationSkill()
    test_result = await test_skill.generate(code, "python")
    
    # 3. 文档生成
    doc_skill = DocumentationSkill()
    doc_result = await doc_skill.generate(code, "python")
    
    # 验证
    assert review_result["score"] > 0
    assert len(test_result["tests"]) > 0
    assert len(doc_result["docs"]) > 0
```

#### 性能测试

```python
"""
Skill 性能测试
"""
import pytest
import asyncio
from code_review_skill import CodeReviewSkill

@pytest.mark.asyncio
async def test_concurrent_reviews():
    """测试并发审查"""
    skill = CodeReviewSkill()
    
    codes = [f"def func{i}(): pass" for i in range(100)]
    
    # 并发执行
    tasks = [skill.review(code, "python") for code in codes]
    results = await asyncio.gather(*tasks)
    
    assert len(results) == 100
    assert all("score" in r for r in results)

@pytest.mark.asyncio
async def test_large_codebase():
    """测试大型代码库"""
    skill = CodeReviewSkill()
    
    # 生成大型代码
    large_code = "\n".join([f"def func{i}(): pass" for i in range(1000)])
    
    import time
    start = time.time()
    result = await skill.review(large_code, "python")
    duration = time.time() - start
    
    assert result["score"] >= 0
    assert duration < 10.0, "处理时间过长"
```

#### 安全测试

```python
"""
Skill 安全测试
"""
import pytest
from code_review_skill import CodeReviewSkill

@pytest.mark.asyncio
async def test_malicious_code():
    """测试恶意代码"""
    skill = CodeReviewSkill()
    
    # 注入尝试
    malicious_code = """
import os
os.system('rm -rf /')
"""
    
    # 应该安全地分析，不执行代码
    result = await skill.review(malicious_code, "python")
    
    # 应该检测到安全问题
    assert any(
        "security" in str(issue).lower() or
        "os.system" in str(issue)
        for issue in result["issues"]
    )

@pytest.mark.asyncio
async def test_path_traversal():
    """测试路径遍历攻击"""
    skill = CodeReviewSkill()
    
    # 尝试访问敏感文件
    code_with_path = """
with open('/etc/passwd') as f:
    content = f.read()
"""
    
    result = await skill.review(code_with_path, "python")
    
    # 应该警告安全风险
    assert any(
        "security" in str(issue).lower() or
        "/etc/passwd" in str(issue)
        for issue in result["issues"]
    )
```

---

## 3. 安全与权限控制

### 3.1 认证机制

#### API Key 认证

```python
"""
API Key 认证
"""
import os
from mcp.server.fastmcp import FastMCP
from functools import wraps

mcp = FastMCP("secure-server")

# 验证 API Key
def require_api_key(func):
    """API Key 认证装饰器"""
    @wraps(func)
    async def wrapper(*args, **kwargs):
        api_key = os.getenv("API_KEY")
        if not api_key:
            raise ValueError("API Key 未配置")
        
        # 从上下文获取 API Key
        # 实际实现取决于传输层
        provided_key = kwargs.get("api_key")
        if provided_key != api_key:
            raise ValueError("API Key 无效")
        
        return await func(*args, **kwargs)
    return wrapper

@mcp.tool()
@require_api_key
async def secure_operation(api_key: str, data: str) -> str:
    """需要 API Key 的操作"""
    return f"操作成功：{data}"
```

```typescript
// TypeScript - API Key 认证
import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import express from "express";

const API_KEY = process.env.API_KEY;

const app = express();

// 中间件：验证 API Key
app.use("/sse", (req, res, next) => {
  const providedKey = req.headers["x-api-key"];
  
  if (!providedKey || providedKey !== API_KEY) {
    res.status(401).json({ error: "Invalid API Key" });
    return;
  }
  
  next();
});

const server = new McpServer({
  name: "secure-server",
  version: "1.0.0"
});

// ... 定义工具 ...

app.listen(3000);
```

#### OAuth 认证

```python
"""
OAuth 2.0 认证
"""
from fastapi import FastAPI, Depends, HTTPException
from fastapi.security import OAuth2PasswordBearer
import httpx

app = FastAPI()

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

async def verify_token(token: str = Depends(oauth2_scheme)):
    """验证 OAuth Token"""
    # 调用认证服务验证 token
    async with httpx.AsyncClient() as client:
        response = await client.get(
            "https://auth.example.com/verify",
            headers={"Authorization": f"Bearer {token}"}
        )
        
        if response.status_code != 200:
            raise HTTPException(status_code=401, detail="Invalid token")
        
        return response.json()

@app.post("/mcp/sse")
async def sse_endpoint(token: str = Depends(verify_token)):
    """需要 OAuth 认证的 SSE 端点"""
    # 建立 MCP 连接
    # ...
    pass
```

#### JWT 认证

```python
"""
JWT 认证
"""
import jwt
import os
from datetime import datetime, timedelta
from functools import wraps

SECRET_KEY = os.getenv("JWT_SECRET")
ALGORITHM = "HS256"
TOKEN_EXPIRE_MINUTES = 60

def create_token(user_id: str, role: str = "user") -> str:
    """创建 JWT Token"""
    expire = datetime.utcnow() + timedelta(minutes=TOKEN_EXPIRE_MINUTES)
    
    payload = {
        "sub": user_id,
        "role": role,
        "exp": expire,
        "iat": datetime.utcnow()
    }
    
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)

def verify_jwt(func):
    """JWT 验证装饰器"""
    @wraps(func)
    async def wrapper(*args, **kwargs):
        token = kwargs.pop("token", None)
        if not token:
            raise ValueError("Token 不能为空")
        
        try:
            payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
            kwargs["user"] = payload
        except jwt.ExpiredSignatureError:
            raise ValueError("Token 已过期")
        except jwt.InvalidTokenError:
            raise ValueError("Token 无效")
        
        return await func(*args, **kwargs)
    return wrapper

@mcp.tool()
@verify_jwt
async def authenticated_action(token: str, user: dict, data: str) -> str:
    """需要认证的操作"""
    user_id = user["sub"]
    return f"用户 {user_id} 执行操作：{data}"
```

### 3.2 授权控制

#### 角色权限

```python
"""
基于角色的权限控制 (RBAC)
"""
from enum import Enum
from functools import wraps

class Role(Enum):
    ADMIN = "admin"
    DEVELOPER = "developer"
    VIEWER = "viewer"

class Permission(Enum):
    READ = "read"
    WRITE = "write"
    DELETE = "delete"
    ADMIN = "admin"

# 角色权限映射
ROLE_PERMISSIONS = {
    Role.ADMIN: {Permission.READ, Permission.WRITE, Permission.DELETE, Permission.ADMIN},
    Role.DEVELOPER: {Permission.READ, Permission.WRITE},
    Role.VIEWER: {Permission.READ}
}

def require_permission(permission: Permission):
    """权限检查装饰器"""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            user = kwargs.get("user")
            if not user:
                raise ValueError("用户未认证")
            
            user_role = Role(user.get("role", "viewer"))
            user_permissions = ROLE_PERMISSIONS.get(user_role, set())
            
            if permission not in user_permissions:
                raise PermissionError(f"缺少权限: {permission.value}")
            
            return await func(*args, **kwargs)
        return wrapper
    return decorator

@mcp.tool()
@require_permission(Permission.READ)
async def read_data(user: dict, resource_id: str) -> str:
    """读取数据（需要 READ 权限）"""
    return f"读取资源: {resource_id}"

@mcp.tool()
@require_permission(Permission.WRITE)
async def write_data(user: dict, resource_id: str, data: str) -> str:
    """写入数据（需要 WRITE 权限）"""
    return f"写入资源: {resource_id}"

@mcp.tool()
@require_permission(Permission.DELETE)
async def delete_data(user: dict, resource_id: str) -> str:
    """删除数据（需要 DELETE 权限）"""
    return f"删除资源: {resource_id}"

@mcp.tool()
@require_permission(Permission.ADMIN)
async def admin_operation(user: dict, action: str) -> str:
    """管理操作（需要 ADMIN 权限）"""
    return f"执行管理操作: {action}"
```

#### 资源访问控制

```python
"""
资源级访问控制
"""
from typing import Optional

class ResourceACL:
    """资源访问控制列表"""
    
    def __init__(self):
        self.permissions = {}
    
    def grant(
        self,
        resource_id: str,
        user_id: str,
        permission: Permission
    ):
        """授予权限"""
        key = f"{resource_id}:{user_id}"
        self.permissions[key] = permission
    
    def check(
        self,
        resource_id: str,
        user_id: str,
        permission: Permission
    ) -> bool:
        """检查权限"""
        key = f"{resource_id}:{user_id}"
        return self.permissions.get(key) == permission

# 全局 ACL
acl = ResourceACL()

def require_resource_permission(permission: Permission):
    """资源权限检查装饰器"""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            user = kwargs.get("user")
            resource_id = kwargs.get("resource_id")
            
            if not user or not resource_id:
                raise ValueError("参数缺失")
            
            user_id = user["sub"]
            
            if not acl.check(resource_id, user_id, permission):
                raise PermissionError(
                    f"无权访问资源 {resource_id} ({permission.value})"
                )
            
            return await func(*args, **kwargs)
        return wrapper
    return decorator

@mcp.tool()
@require_resource_permission(Permission.READ)
async def read_resource(
    user: dict,
    resource_id: str
) -> str:
    """读取特定资源"""
    return f"资源内容: {resource_id}"

@mcp.tool()
@require_resource_permission(Permission.WRITE)
async def write_resource(
    user: dict,
    resource_id: str,
    data: str
) -> str:
    """写入特定资源"""
    return f"已更新资源: {resource_id}"
```

#### 工具调用限制

```python
"""
工具调用限制
"""
import time
from collections import defaultdict

class RateLimiter:
    """速率限制器"""
    
    def __init__(self, max_calls: int = 100, window_seconds: int = 3600):
        self.max_calls = max_calls
        self.window_seconds = window_seconds
        self.call_history = defaultdict(list)
    
    def check_limit(self, user_id: str) -> bool:
        """检查是否超过限制"""
        now = time.time()
        window_start = now - self.window_seconds
        
        # 清理过期记录
        self.call_history[user_id] = [
            t for t in self.call_history[user_id]
            if t > window_start
        ]
        
        # 检查限制
        return len(self.call_history[user_id]) < self.max_calls
    
    def record_call(self, user_id: str):
        """记录调用"""
        self.call_history[user_id].append(time.time())

# 全局速率限制器
rate_limiter = RateLimiter(max_calls=100, window_seconds=3600)

def rate_limit(max_calls: int = 10, window_seconds: int = 60):
    """速率限制装饰器"""
    limiter = RateLimiter(max_calls, window_seconds)
    
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):            user = kwargs.get("user")
            if not user:
                raise ValueError("用户未认证")
            
            user_id = user["sub"]
            
            if not limiter.check_limit(user_id):
                raise RateLimitError(
                    f"超过调用限制 ({max_calls}/{window_seconds}s)"
                )
            
            limiter.record_call(user_id)
            return await func(*args, **kwargs)
        return wrapper
    return decorator

@mcp.tool()
@rate_limit(max_calls=10, window_seconds=60)
async def rate_limited_tool(user: dict, query: str) -> str:
    """限制调用频率的工具"""
    return f"处理查询: {query}"

# 自定义异常
class RateLimitError(Exception):
    """速率限制异常"""
    pass
```

### 3.3 安全防护

#### 输入验证

```python
"""
输入验证
"""
import re
from typing import Annotated
from pydantic import BaseModel, Field, validator

class SearchInput(BaseModel):
    """搜索输入验证"""
    query: str = Field(
        ...,
        min_length=1,
        max_length=100,
        description="搜索关键词"
    )
    
    limit: int = Field(
        default=10,
        ge=1,
        le=50,
        description="结果数量"
    )
    
    source: str = Field(
        default="web",
        pattern="^(web|docs|code)$",
        description="数据源"
    )
    
    @validator("query")
    def validate_query(cls, v):
        """验证查询内容"""
        # 防止 SQL 注入
        if re.search(r"[';\"\\]", v):
            raise ValueError("查询包含非法字符")
        
        # 防止 XSS
        if re.search(r"<script>|javascript:", v, re.IGNORECASE):
            raise ValueError("查询包含潜在 XSS 攻击")
        
        return v.strip()

@mcp.tool()
async def safe_search(query: str, limit: int = 10) -> str:
    """安全的搜索工具"""
    try:
        # 使用 Pydantic 验证
        input_data = SearchInput(query=query, limit=limit)
        
        # 执行搜索
        results = perform_search(input_data.query, input_data.limit)
        
        return format_results(results)
    
    except ValueError as e:
        return f"输入验证失败: {e}"
    except Exception as e:
        return f"搜索失败: {e}"
```

#### 输出过滤

```python
"""
输出过滤
"""
import re
from html import escape

class OutputFilter:
    """输出过滤器"""
    
    def __init__(self):
        # 敏感信息模式
        self.sensitive_patterns = [
            re.compile(r"\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b"),  # 信用卡
            re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),  # SSN
            re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"),  # 邮箱
            re.compile(r"password\s*=\s*['\"]\w+['\"]"),  # 密码
        ]
    
    def filter_sensitive(self, text: str) -> str:
        """过滤敏感信息"""
        for pattern in self.sensitive_patterns:
            text = pattern.sub("[REDACTED]", text)
        return text
    
    def escape_html(self, text: str) -> str:
        """转义 HTML"""
        return escape(text)
    
    def sanitize(self, text: str, escape_html: bool = True) -> str:
        """清理输出"""
        # 过滤敏感信息
        text = self.filter_sensitive(text)
        
        # 转义 HTML
        if escape_html:
            text = self.escape_html(text)
        
        return text

# 全局过滤器
output_filter = OutputFilter()

@mcp.tool()
async def safe_query(user_id: str) -> str:
    """安全的查询工具"""
    try:
        # 执行查询
        result = query_database(user_id)
        
        # 过滤输出
        filtered_result = output_filter.sanitize(result)
        
        return filtered_result
    
    except Exception as e:
        return f"查询失败: {e}"
```

#### 注入防护

```python
"""
注入防护
"""
import sqlite3
import subprocess
from typing import Optional

class InjectionProtection:
    """注入防护工具类"""
    
    @staticmethod
    def protect_sql(query: str, params: tuple = None) -> tuple[str, tuple]:
        """
        防止 SQL 注入
        
        使用参数化查询
        """
        # ❌ 错误：字符串拼接
        # sql = f"SELECT * FROM users WHERE id = {user_id}"
        
        # ✅ 正确：参数化查询
        sql = "SELECT * FROM users WHERE id = ?"
        params = (user_id,)
        
        return sql, params
    
    @staticmethod
    def protect_command(command: str, args: list[str]) -> list[str]:
        """
        防止命令注入
        
        使用参数列表，不使用 shell=True
        """
        # ❌ 错误：shell 执行
        # subprocess.run(f"ls {directory}", shell=True)
        
        # ✅ 正确：参数列表
        return [command] + args
    
    @staticmethod
    def sanitize_path(path: str, allowed_base: str) -> str:
        """
        防止路径遍历
        
        限制在允许的目录内
        """
        import os
        
        # 解析绝对路径
        abs_path = os.path.abspath(path)
        abs_base = os.path.abspath(allowed_base)
        
        # 检查是否在允许的目录内
        if not abs_path.startswith(abs_base):
            raise ValueError(f"路径超出允许范围: {allowed_base}")
        
        return abs_path

# 使用示例
@mcp.tool()
async def safe_database_query(user_id: str) -> str:
    """安全的数据库查询"""
    try:
        # 使用参数化查询
        sql, params = InjectionProtection.protect_sql(
            "SELECT * FROM users WHERE id = ?",
            (user_id,)
        )
        
        conn = sqlite3.connect("database.db")
        cursor = conn.cursor()
        cursor.execute(sql, params)
        results = cursor.fetchall()
        conn.close()
        
        return str(results)
    
    except Exception as e:
        return f"查询失败: {e}"

@mcp.tool()
async def safe_file_read(file_path: str) -> str:
    """安全的文件读取"""
    try:
        # 限制访问目录
        allowed_base = "/app/data"
        safe_path = InjectionProtection.sanitize_path(file_path, allowed_base)
        
        with open(safe_path, 'r') as f:
            return f.read()
    
    except ValueError as e:
        return f"安全错误: {e}"
    except Exception as e:
        return f"读取失败: {e}"
```

#### 审计日志

```python
"""
审计日志
"""
import logging
import json
from datetime import datetime
from functools import wraps

# 配置审计日志
audit_logger = logging.getLogger("audit")
audit_logger.setLevel(logging.INFO)

# 文件处理器
file_handler = logging.FileHandler("audit.log")
file_handler.setFormatter(
    logging.Formatter("%(asctime)s - %(message)s")
)
audit_logger.addHandler(file_handler)

# 控制台处理器
console_handler = logging.StreamHandler()
console_handler.setFormatter(
    logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
)
audit_logger.addHandler(console_handler)

def audit_log(func):
    """审计日志装饰器"""
    @wraps(func)
    async def wrapper(*args, **kwargs):
        # 提取用户信息
        user = kwargs.get("user", {})
        user_id = user.get("sub", "anonymous")
        
        # 记录开始
        audit_logger.info(
            f"TOOL_CALL_START | "
            f"user={user_id} | "
            f"tool={func.__name__} | "
            f"args={json.dumps(kwargs, default=str)}"
        )
        
        try:
            # 执行
            result = await func(*args, **kwargs)
            
            # 记录成功
            audit_logger.info(
                f"TOOL_CALL_SUCCESS | "
                f"user={user_id} | "
                f"tool={func.__name__} | "
                f"result_length={len(str(result))}"
            )
            
            return result
        
        except Exception as e:
            # 记录失败
            audit_logger.error(
                f"TOOL_CALL_ERROR | "
                f"user={user_id} | "
                f"tool={func.__name__} | "
                f"error={str(e)}"
            )
            raise
    
    return wrapper

@mcp.tool()
@audit_log
async def audited_operation(user: dict, action: str) -> str:
    """带审计日志的操作"""
    return f"执行操作: {action}"

# 审计日志示例：
# 2025-06-12 10:30:00 - TOOL_CALL_START | user=user123 | tool=audited_operation | args={"action": "delete", "resource": "data.json"}
# 2025-06-12 10:30:01 - TOOL_CALL_SUCCESS | user=user123 | tool=audited_operation | result_length=20
```

---

## 4. 生产部署

### 4.1 部署架构

#### 单体部署

```
┌─────────────────────────────────┐
│         Docker Container         │
│                                  │
│  ┌──────────────────────────┐   │
│  │   MCP Server (Python)    │   │
│  │                          │   │
│  │  - Tools                 │   │
│  │  - Resources             │   │
│  │  - Prompts               │   │
│  └──────────────────────────┘   │
│            ↕                     │
│  ┌──────────────────────────┐   │
│  │    Stdio Transport       │   │
│  └──────────────────────────┘   │
└─────────────────────────────────┘
         ↑ Stdio
         │
┌─────────────────────────────────┐
│      Claude Desktop / Cursor     │
└─────────────────────────────────┘
```

**docker-compose.yml**：

```yaml
version: '3.8'

services:
  mcp-server:
    build: .
    container_name: mcp-server
    restart: unless-stopped
    environment:
      - API_KEY=${API_KEY}
      - DATABASE_URL=${DATABASE_URL}
    volumes:
      - ./data:/app/data
    stdin_open: true
    tty: true
```

**Dockerfile**：

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# 安装依赖
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 复制代码
COPY . .

# 运行
CMD ["python", "server.py"]
```

#### 微服务部署

```
┌──────────────────────────────────────────────────────┐
│                  Kubernetes Cluster                    │
│                                                        │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  │
│  │  Search      │  │  Database   │  │  GitHub     │  │
│  │  MCP Server  │  │  MCP Server │  │  MCP Server │  │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘  │
│         │                │                │           │
│         └────────────────┼────────────────┘           │
│                          │                            │
│              ┌───────────┴───────────┐               │
│              │    API Gateway         │               │
│              │    (HTTP/SSE)          │               │
│              └───────────┬───────────┘               │
└──────────────────────────┼───────────────────────────┘
                           │
              ┌────────────┴────────────┐
              │   MCP Client (Remote)    │
              └─────────────────────────┘
```

**Kubernetes Deployment**：

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: mcp-server
spec:
  replicas: 3
  selector:
    matchLabels:
      app: mcp-server
  template:
    metadata:
      labels:
        app: mcp-server
    spec:
      containers:
      - name: mcp-server
        image: my-registry/mcp-server:latest
        ports:
        - containerPort: 3000
        env:
        - name: API_KEY
          valueFrom:
            secretKeyRef:
              name: mcp-secrets
              key: api-key
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: mcp-secrets
              key: database-url
        resources:
          requests:
            memory: "256Mi"
            cpu: "250m"
          limits:
            memory: "512Mi"
            cpu: "500m"
        livenessProbe:
          httpGet:
            path: /health
            port: 3000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /health
            port: 3000
          initialDelaySeconds: 5
          periodSeconds: 5
---
apiVersion: v1
kind: Service
metadata:
  name: mcp-server
spec:
  selector:
    app: mcp-server
  ports:
  - port: 80
    targetPort: 3000
  type: ClusterIP
```

#### 容器化部署

```yaml
# docker-compose.prod.yml
version: '3.8'

services:
  # MCP Server
  mcp-server:
    build: ./mcp-server
    restart: unless-stopped
    environment:
      - NODE_ENV=production
      - API_KEY=${API_KEY}
    networks:
      - mcp-network
    depends_on:
      - redis
      - postgres

  # Redis 缓存
  redis:
    image: redis:7-alpine
    restart: unless-stopped
    volumes:
      - redis-data:/data
    networks:
      - mcp-network

  # PostgreSQL
  postgres:
    image: postgres:15-alpine
    restart: unless-stopped
    environment:
      - POSTGRES_DB=mcp
      - POSTGRES_USER=${DB_USER}
      - POSTGRES_PASSWORD=${DB_PASSWORD}
    volumes:
      - postgres-data:/var/lib/postgresql/data
    networks:
      - mcp-network

  # Nginx 反向代理
  nginx:
    image: nginx:alpine
    restart: unless-stopped
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
      - ./ssl:/etc/nginx/ssl
    networks:
      - mcp-network
    depends_on:
      - mcp-server

networks:
  mcp-network:
    driver: bridge

volumes:
  redis-data:
  postgres-data:
```

#### 云原生部署

```yaml
# AWS ECS Task Definition
{
  "family": "mcp-server",
  "networkMode": "awsvpc",
  "requiresCompatibilities": ["FARGATE"],
  "cpu": "512",
  "memory": "1024",
  "containerDefinitions": [
    {
      "name": "mcp-server",
      "image": "my-registry/mcp-server:latest",
      "portMappings": [
        {
          "containerPort": 3000,
          "protocol": "tcp"
        }
      ],
      "environment": [
        {
          "name": "NODE_ENV",
          "value": "production"
        }
      ],
      "secrets": [
        {
          "name": "API_KEY",
          "valueFrom": "arn:aws:secretsmanager:region:account:secret:MySecret"
        }
      ],
      "logConfiguration": {
        "logDriver": "awslogs",
        "options": {
          "awslogs-group": "/ecs/mcp-server",
          "awslogs-region": "us-east-1",
          "awslogs-stream-prefix": "ecs"
        }
      }
    }
  ]
}
```

### 4.2 性能优化

#### 连接池

```python
"""
数据库连接池
"""
from databases import Database
from contextlib import asynccontextmanager

# 创建连接池
database = Database(
    "postgresql://user:password@localhost/mydb",
    min_size=5,   # 最小连接数
    max_size=20   # 最大连接数
)

@asynccontextmanager
async def get_db():
    """获取数据库连接"""
    async with database.transaction() as transaction:
        yield transaction

@mcp.tool()
async def query_with_pool(user_id: str) -> str:
    """使用连接池查询"""
    async with get_db() as db:
        result = await db.fetch_all(
            "SELECT * FROM users WHERE id = $1",
            user_id
        )
        return str(result)
```

```typescript
/**
 * HTTP 连接池 - TypeScript
 */
import http from "http";
import https from "https";

// 创建连接池
const httpAgent = new http.Agent({
  keepAlive: true,
  maxSockets: 50,
  maxFreeSockets: 10,
  timeout: 30000,
  freeSocketTimeout: 5000
});

const httpsAgent = new https.Agent({
  keepAlive: true,
  maxSockets: 50,
  maxFreeSockets: 10,
  timeout: 30000,
  freeSocketTimeout: 5000
});

// 使用连接池
async function makeRequest(url: string) {
  const agent = url.startsWith("https") ? httpsAgent : httpAgent;
  
  const response = await fetch(url, {
    agent,
    headers: { "Connection": "keep-alive" }
  });
  
  return response.json();
}
```

#### 缓存策略

```python
"""
多级缓存策略
"""
import asyncio
from functools import wraps
from typing import Optional
import hashlib

class CacheLayer:
    """缓存层"""
    
    def __init__(self):
        # L1: 内存缓存（快速访问）
        self.memory_cache = {}
        self.memory_ttl = 60  # 1 分钟
        
        # L2: Redis 缓存（分布式）
        self.redis_client = None  # 初始化 Redis
    
    async def get(self, key: str) -> Optional[str]:
        """获取缓存"""
        # L1 缓存
        if key in self.memory_cache:
            item = self.memory_cache[key]
            if asyncio.get_event_loop().time() - item["time"] < self.memory_ttl:
                return item["value"]
            else:
                del self.memory_cache[key]
        
        # L2 缓存
        if self.redis_client:
            value = await self.redis_client.get(key)
            if value:
                # 回填 L1
                self.memory_cache[key] = {
                    "value": value,
                    "time": asyncio.get_event_loop().time()
                }
                return value
        
        return None
    
    async def set(self, key: str, value: str, ttl: int = 3600):
        """设置缓存"""
        # 设置 L1
        self.memory_cache[key] = {
            "value": value,
            "time": asyncio.get_event_loop().time()
        }
        
        # 设置 L2
        if self.redis_client:
            await self.redis_client.setex(key, ttl, value)
    
    def generate_key(self, func_name: str, *args, **kwargs) -> str:
        """生成缓存键"""
        params = f"{func_name}:{args}:{sorted(kwargs.items())}"
        return hashlib.md5(params.encode()).hexdigest()

# 全局缓存
cache = CacheLayer()

def cached(ttl: int = 3600):
    """缓存装饰器"""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # 生成缓存键
            cache_key = cache.generate_key(func.__name__, *args, **kwargs)
            
            # 尝试获取缓存
            result = await cache.get(cache_key)
            if result:
                return result
            
            # 执行函数
            result = await func(*args, **kwargs)
            
            # 设置缓存
            await cache.set(cache_key, result, ttl)
            
            return result
        return wrapper
    return decorator

@mcp.tool()
@cached(ttl=300)  # 缓存 5 分钟
async def cached_search(query: str) -> str:
    """带缓存的搜索"""
    results = perform_search(query)
    return format_results(results)
```

#### 并发控制

```python
"""
并发控制
"""
import asyncio
from asyncio import Semaphore

# 限制并发数
semaphore = Semaphore(10)  # 最多 10 个并发

@mcp.tool()
async def concurrent_safe_operation(user_id: str) -> str:
    """并发安全的操作"""
    async with semaphore:
        # 执行操作
        result = await expensive_operation(user_id)
        return result

# 批量处理
async def process_batch(items: list, batch_size: int = 5) -> list:
    """批量处理"""
    results = []
    
    for i in range(0, len(items), batch_size):
        batch = items[i:i + batch_size]
        
        # 并发处理批次
        tasks = [process_item(item) for item in batch]
        batch_results = await asyncio.gather(*tasks)
        
        results.extend(batch_results)
        
        # 批次间延迟
        await asyncio.sleep(0.1)
    
    return results
```

#### 负载均衡

```yaml
# Nginx 负载均衡配置
upstream mcp_servers {
    least_conn;  # 最少连接算法
    
    server mcp-server-1:3000 weight=3;
    server mcp-server-2:3000 weight=2;
    server mcp-server-3:3000 weight=1;
    
    # 健康检查
    keepalive 32;
}

server {
    listen 80;
    server_name mcp.example.com;
    
    location /sse {
        proxy_pass http://mcp_servers;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        
        # 超时设置
        proxy_connect_timeout 10s;
        proxy_send_timeout 300s;
        proxy_read_timeout 300s;
    }
    
    location /messages {
        proxy_pass http://mcp_servers;
        proxy_http_version 1.1;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

### 4.3 监控告警

#### 健康检查

```python
"""
健康检查端点
"""
from fastapi import FastAPI
import psutil
import asyncio

app = FastAPI()

@app.get("/health")
async def health_check():
    """健康检查"""
    # 系统资源
    cpu_percent = psutil.cpu_percent(interval=1)
    memory = psutil.virtual_memory()
    
    # 检查项
    checks = {
        "cpu": {
            "status": "healthy" if cpu_percent < 90 else "warning",
            "value": cpu_percent
        },
        "memory": {
            "status": "healthy" if memory.percent < 90 else "warning",
            "value": memory.percent
        },
        "uptime": {
            "status": "healthy",
            "value": psutil.boot_time()
        }
    }
    
    # 总体状态
    overall_status = all(
        check["status"] == "healthy"
        for check in checks.values()
    )
    
    return {
        "status": "healthy" if overall_status else "unhealthy",
        "checks": checks,
        "timestamp": asyncio.get_event_loop().time()
    }

@app.get("/ready")
async def readiness_check():
    """就绪检查"""
    # 检查依赖服务
    checks = {
        "database": await check_database(),
        "redis": await check_redis(),
        "external_api": await check_external_api()
    }
    
    all_ready = all(checks.values())
    
    return {
        "ready": all_ready,
        "checks": checks
    }

async def check_database() -> bool:
    """检查数据库连接"""
    try:
        # 执行简单查询
        await database.fetch_one("SELECT 1")
        return True
    except:
        return False

async def check_redis() -> bool:
    """检查 Redis 连接"""
    try:
        await redis_client.ping()
        return True
    except:
        return False

async def check_external_api() -> bool:
    """检查外部 API"""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get("https://api.example.com/health")
            return response.status_code == 200
    except:
        return False
```

#### 性能指标

```python
"""
性能指标收集
"""
import time
from prometheus_client import Counter, Histogram, Gauge
from functools import wraps

# 定义指标
REQUEST_COUNT = Counter(
    'mcp_tool_calls_total',
    'Total tool calls',
    ['tool_name', 'status']
)

REQUEST_DURATION = Histogram(
    'mcp_tool_duration_seconds',
    'Tool execution duration',
    ['tool_name']
)

ACTIVE_CONNECTIONS = Gauge(
    'mcp_active_connections',
    'Number of active connections'
)

def track_metrics(func):
    """性能指标装饰器"""
    @wraps(func)
    async def wrapper(*args, **kwargs):
        tool_name = func.__name__
        start_time = time.time()
        
        try:
            result = await func(*args, **kwargs)
            
            # 记录成功
            REQUEST_COUNT.labels(tool_name=tool_name, status="success").inc()
            
            return result
        
        except Exception as e:
            # 记录失败
            REQUEST_COUNT.labels(tool_name=tool_name, status="error").inc()
            raise
        
        finally:
            # 记录耗时
            duration = time.time() - start_time
            REQUEST_DURATION.labels(tool_name=tool_name).observe(duration)
    
    return wrapper

@mcp.tool()
@track_metrics
async def monitored_tool(query: str) -> str:
    """带监控的工具"""
    return process_query(query)
```

**Grafana Dashboard 配置**：

```json
{
  "dashboard": {
    "title": "MCP Server Metrics",
    "panels": [
      {
        "title": "Tool Call Rate",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(mcp_tool_calls_total[5m])",
            "legendFormat": "{{tool_name}} - {{status}}"
          }
        ]
      },
      {
        "title": "Tool Latency",
        "type": "graph",
        "targets": [
          {
            "expr": "histogram_quantile(0.95, rate(mcp_tool_duration_seconds_bucket[5m]))",
            "legendFormat": "{{tool_name}} p95"
          }
        ]
      },
      {
        "title": "Error Rate",
        "type": "singlestat",
        "targets": [
          {
            "expr": "sum(rate(mcp_tool_calls_total{status=\"error\"}[5m])) / sum(rate(mcp_tool_calls_total[5m]))"
          }
        ]
      }
    ]
  }
}
```

#### 错误追踪

```python
"""
错误追踪 - Sentry 集成
"""
import sentry_sdk
from sentry_sdk.integrations.asyncio import AsyncioIntegration

# 初始化 Sentry
sentry_sdk.init(
    dsn="https://your-dsn@sentry.io/project-id",
    integrations=[AsyncioIntegration()],
    traces_sample_rate=1.0,
    environment="production"
)

@mcp.tool()
async def tracked_tool(query: str) -> str:
    """带错误追踪的工具"""
    # 添加上下文
    sentry_sdk.set_context("query", {
        "length": len(query),
        "preview": query[:50]
    })
    
    sentry_sdk.set_tag("tool", "tracked_tool")
    
    try:
        result = process_query(query)
        return result
    
    except Exception as e:
        # 自动发送到 Sentry
        sentry_sdk.capture_exception(e)
        raise
```

#### 日志聚合

```python
"""
结构化日志 - JSON 格式
"""
import json
import logging
from datetime import datetime

class JSONFormatter(logging.Formatter):
    """JSON 日志格式化器"""
    
    def format(self, record):
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno
        }
        
        # 添加异常信息
        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)
        
        # 添加额外字段
        if hasattr(record, "user_id"):
            log_entry["user_id"] = record.user_id
        if hasattr(record, "tool_name"):
            log_entry["tool_name"] = record.tool_name
        
        return json.dumps(log_entry)

# 配置日志
logger = logging.getLogger("mcp")
logger.setLevel(logging.INFO)

handler = logging.StreamHandler()
handler.setFormatter(JSONFormatter())
logger.addHandler(handler)

# 使用
logger.info(
    "Tool executed",
    extra={
        "user_id": "user123",
        "tool_name": "search",
        "duration": 1.5
    }
)

# 输出：
# {"timestamp": "2025-06-12T10:30:00", "level": "INFO", "message": "Tool executed", "module": "server", "function": "search", "line": 42, "user_id": "user123", "tool_name": "search", "duration": 1.5}
```

---

