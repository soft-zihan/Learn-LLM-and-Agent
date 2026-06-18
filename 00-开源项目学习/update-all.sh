#!/bin/bash

# ===========================================
# 99-开源项目学习 更新脚本
# 遍历所有项目执行 git pull，保持与上游同步
# 用法：cd 99-开源项目学习 && bash update-all.sh
# ===========================================

set -e

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 统计
SUCCESS=0
SKIPPED=0
FAILED=0
FAILED_LIST=""

# 更新单个项目的函数
update_project() {
    local dir="$1"
    local name=$(basename "$dir")
    
    if [ ! -d "$dir/.git" ]; then
        echo -e "${YELLOW}⚠  ${name}${NC}: 不是 Git 仓库，跳过"
        ((SKIPPED++))
        return
    fi
    
    echo -n -e "${BLUE}🔄 ${name}${NC}: 正在更新... "
    
    # 进入项目目录
    cd "$dir"
    
    # 获取当前分支名
    local branch=$(git rev-parse --abbrev-ref HEAD 2>/dev/null)
    
    # 尝试拉取更新
    if git pull --rebase origin "$branch" > /tmp/update_${name}.log 2>&1; then
        # 检查是否有实际更新
        if grep -q "Already up to date" /tmp/update_${name}.log; then
            echo -e "${GREEN}已是最新 ✓${NC}"
        else
            local commits=$(git log --oneline -5 2>/dev/null | head -3 | wc -l)
            echo -e "${GREEN}已更新 ✓${NC} (最近 ${commits} 次提交)"
        fi
        ((SUCCESS++))
    else
        echo -e "${RED}更新失败 ✗${NC}"
        echo -e "   ${RED}错误详情${NC}: $(tail -1 /tmp/update_${name}.log)"
        ((FAILED++))
        FAILED_LIST="${FAILED_LIST}  - ${name}\n"
    fi
    
    rm -f /tmp/update_${name}.log
    cd ..
}

echo "==========================================="
echo "  99-开源项目学习 批量更新"
echo "  $(date '+%Y-%m-%d %H:%M:%S')"
echo "==========================================="
echo ""

# ---- 01-大模型训练基础 ----
echo -e "${BLUE}📂 01-大模型训练基础${NC}"
if [ -d "01-大模型训练基础" ]; then
    cd 01-大模型训练基础
    for project in self-llm llm-action minimind; do
        [ -d "$project" ] && update_project "$project"
    done
    cd ..
else
    echo -e "  ${YELLOW}目录不存在，跳过${NC}"
fi
echo ""

# ---- 02-RAG实战 ----
echo -e "${BLUE}📂 02-RAG实战${NC}"
if [ -d "02-RAG实战" ]; then
    cd 02-RAG实战
    for project in rag-from-scratch rag-tutorial agentic-rag-for-dummies production-agentic-rag-course; do
        [ -d "$project" ] && update_project "$project"
    done
    cd ..
else
    echo -e "  ${YELLOW}目录不存在，跳过${NC}"
fi
echo ""

# ---- 03-MCP与Skills ----
echo -e "${BLUE}📂 03-MCP与Skills${NC}"
if [ -d "03-MCP与Skills" ]; then
    cd 03-MCP与Skills
    for project in mcp-python-sdk mcp-typescript-sdk mcp-servers LLMInternSkill clawkeeper; do
        [ -d "$project" ] && update_project "$project"
    done
    cd ..
else
    echo -e "  ${YELLOW}目录不存在，跳过${NC}"
fi
echo ""

# ---- 04-Agent实战 ----
echo -e "${BLUE}📂 04-Agent实战${NC}"
if [ -d "04-Agent实战" ]; then
    cd 04-Agent实战
    for project in GenAI_Agents Hello-agents Deer-flow Smolagents Ragent openai-agents-sdk crewai learn-claude-code; do
        [ -d "$project" ] && update_project "$project"
    done
    cd ..
else
    echo -e "  ${YELLOW}目录不存在，跳过${NC}"
fi
echo ""

# ---- 汇总报告 ----
echo "==========================================="
echo "  更新完成汇总"
echo "==========================================="
echo -e "  ${GREEN}✓ 成功${NC}: ${SUCCESS} 个项目"
echo -e "  ${YELLOW}⚠ 跳过${NC}: ${SKIPPED} 个（非 Git 仓库）"

if [ $FAILED -gt 0 ]; then
    echo -e "  ${RED}✗ 失败${NC}: ${FAILED} 个项目"
    echo -e "\n${RED}失败项目列表${NC}:"
    echo -e "$FAILED_LIST"
    echo -e "${YELLOW}建议${NC}: 检查网络连接，或对失败项目手动执行 git pull"
fi

TOTAL=$((SUCCESS + SKIPPED + FAILED))
echo ""
echo "共处理 ${TOTAL} 个项目"
echo "==========================================="

# 如果有失败，退出码为 1
[ $FAILED -eq 0 ] && exit 0 || exit 1
