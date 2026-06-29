"""
Skills 系统集成 - 对应教程 04-Skills系统集成

Skills（技能）是预定义的能力模块，每个 Skill 包含：
- SKILL.md：技能定义文件（YAML frontmatter + Markdown 描述）
- 可选的脚本/资源文件

本模块实现了 Skills 的加载、注册、查询：

核心类：
1. SkillDefinition - 技能定义数据类
2. SkillLoader - 从文件系统加载 SKILL.md
3. SkillRegistry - 技能注册表（单例）

SKILL.md 格式示例：
    ---
    name: skill-creator
    description: 创建新的 Agent Skill
    ---
    # Skill Creator
    这是一个用于创建新技能的技能...

教程对应：
- 04-Skills系统集成：技能加载、注册、查询
- 01-工具调用与集成实战：Skills 作为工具的一种
"""

import os
from pathlib import Path
from typing import Optional
import yaml


class SkillDefinition:
    """
    技能定义数据类
    
    每个技能包含：
    - name: 技能名称（唯一标识）
    - description: 技能描述（帮助 LLM 理解何时使用）
    - content: 技能详细内容（Markdown 格式）
    - metadata: 元数据（来自 YAML frontmatter）
    """
    
    def __init__(self, name: str, description: str, content: str, metadata: dict = None):
        self.name = name
        self.description = description
        self.content = content
        self.metadata = metadata or {}
    
    def to_dict(self) -> dict:
        """序列化为字典（用于 API 返回）"""
        return {
            "name": self.name,
            "description": self.description,
            "metadata": self.metadata
        }


class SkillLoader:
    """
    技能加载器：从文件系统加载 SKILL.md
    
    扫描指定目录下的子目录，查找包含 SKILL.md 的目录，
    解析 SKILL.md 的 YAML frontmatter 和 Markdown 内容。
    
    SKILL.md 格式：
        ---
        name: skill-name
        description: 技能描述
        ---
        # 技能详细内容（Markdown）
    """
    
    def __init__(self, skills_dir: str):
        """
        Args:
            skills_dir: 技能根目录（包含多个技能子目录）
        """
        self.skills_dir = Path(skills_dir)
    
    def load_all(self) -> list[SkillDefinition]:
        """
        加载目录下所有技能
        
        遍历 skills_dir 下的子目录，查找包含 SKILL.md 的目录。
        
        Returns:
            加载成功的技能列表
        """
        skills = []
        if not self.skills_dir.exists():
            return skills
        
        for item in self.skills_dir.iterdir():
            # 只处理包含 SKILL.md 的子目录
            if item.is_dir() and (item / "SKILL.md").exists():
                skill = self._load_skill(item / "SKILL.md")
                if skill:
                    skills.append(skill)
        
        return skills
    
    def _load_skill(self, skill_file: Path) -> Optional[SkillDefinition]:
        """
        加载单个 SKILL.md 文件
        
        解析格式：
        1. 以 "---" 开头的 YAML frontmatter
        2. 第二个 "---" 之后的 Markdown 内容
        
        Args:
            skill_file: SKILL.md 文件路径
        
        Returns:
            SkillDefinition 对象，解析失败返回 None
        """
        content = skill_file.read_text(encoding="utf-8")
        
        # 检查是否有 YAML frontmatter（以 --- 开头）
        if content.startswith("---"):
            parts = content.split("---", 2)
            if len(parts) >= 3:
                # 解析 YAML frontmatter
                frontmatter = yaml.safe_load(parts[1])
                return SkillDefinition(
                    name=frontmatter.get("name", skill_file.parent.name),
                    description=frontmatter.get("description", ""),
                    content=parts[2].strip(),
                    metadata=frontmatter
                )
        
        return None


class SkillRegistry:
    """
    技能注册表：管理所有已加载的技能
    
    功能：
    - 注册单个技能
    - 从目录批量加载
    - 按名称查询
    - 按关键词搜索相关技能
    - 构建系统提示词（将技能信息注入 LLM）
    """
    
    def __init__(self):
        self._skills: dict[str, SkillDefinition] = {}
    
    def register(self, skill: SkillDefinition):
        """注册一个技能"""
        self._skills[skill.name] = skill
    
    def register_directory(self, skills_dir: str):
        """从目录加载并注册所有技能"""
        loader = SkillLoader(skills_dir)
        for skill in loader.load_all():
            self.register(skill)
    
    def get(self, name: str) -> Optional[SkillDefinition]:
        """按名称获取技能"""
        return self._skills.get(name)
    
    def list_all(self) -> list[SkillDefinition]:
        """列出所有已注册的技能"""
        return list(self._skills.values())
    
    def find_relevant(self, query: str) -> list[SkillDefinition]:
        """
        根据查询关键词搜索相关技能
        
        搜索范围：技能名称 + 技能描述
        匹配方式：子串匹配（不区分大小写）
        
        Args:
            query: 搜索关键词
        
        Returns:
            匹配的技能列表
        """
        query_lower = query.lower()
        return [
            s for s in self._skills.values()
            if query_lower in s.name.lower() or query_lower in s.description.lower()
        ]
    
    def build_system_prompt(self, query: str = "") -> str:
        """
        构建包含技能信息的系统提示词
        
        将可用技能列表格式化为 Markdown，注入到 LLM 的系统提示中，
        让 LLM 知道有哪些技能可以使用。
        
        Args:
            query: 可选的查询关键词，用于过滤相关技能
        
        Returns:
            格式化的技能列表字符串
        """
        if not self._skills:
            return ""
        
        relevant = self.find_relevant(query) if query else list(self._skills.values())
        if not relevant:
            return ""
        
        lines = ["## 可用技能", ""]
        for skill in relevant[:5]:  # 最多展示5个
            lines.append(f"- **{skill.name}**: {skill.description}")
        
        return "\n".join(lines)


# ── 全局技能注册表（单例）────────────────────────────────────

skill_registry = SkillRegistry()


def init_skills(skills_dir: str = None):
    """
    初始化技能系统（应用启动时调用）
    
    从默认目录或指定目录加载所有技能。
    默认目录：02-Agent工具与协议/SKILLS/
    
    Args:
        skills_dir: 技能目录路径，None 使用默认路径
    """
    if skills_dir is None:
        # 相对路径：从当前文件向上找到项目根目录
        skills_dir = os.path.join(
            os.path.dirname(__file__),
            "../../../02-Agent工具与协议/SKILLS"
        )
    
    skill_registry.register_directory(skills_dir)
    
    print(f"[Skills] 已加载 {len(skill_registry.list_all())} 个技能")
    for skill in skill_registry.list_all():
        desc = skill.description[:50] + "..." if len(skill.description) > 50 else skill.description
        print(f"  - {skill.name}: {desc}")
