# md-to-pdf-pandoc 安装指南

> 📖 使用指南请查看 [SKILL.md](../.qoder/skills/md-to-pdf-pandoc/SKILL.md)

## 系统要求

- **操作系统**: macOS / Linux / Windows (WSL)
- **Python**: 3.8+
- **磁盘空间**: ~2GB (LaTeX 完整安装)

---

## macOS 安装

### 1. 安装 Pandoc

```bash
brew install pandoc
```

### 2. 安装 LaTeX (MacTeX)

**完整安装** (推荐，约 4GB):
```bash
brew install --cask mactex
```

**精简安装** (仅 XeLaTeX，约 500MB):
```bash
# 下载 BasicTeX
# https://www.tug.org/mactex/morepackages.html
sudo tlmgr update --self
sudo tlmgr install xetex
sudo tlmgr install xecjk  # 中文支持
sudo tlmgr install ctex    # 中文宏包
```

### 3. 安装 Python 依赖

```bash
pip3 install PyPDF2 markdown
```

### 4. 验证安装

```bash
pandoc --version
xelatex --version
python3 -c "import PyPDF2; print('✅ PyPDF2 已安装')"
```

---

## Linux 安装 (Ubuntu/Debian)

### 1. 安装 Pandoc

```bash
sudo apt update
sudo apt install pandoc
```

### 2. 安装 LaTeX

```bash
# 完整安装
sudo apt install texlive-full

# 或精简安装 (推荐)
sudo apt install texlive-xetex texlive-lang-chinese
```

### 3. 安装 Python 依赖

```bash
pip3 install PyPDF2 markdown
```

### 4. 验证安装

```bash
pandoc --version
xelatex --version
python3 -c "import PyPDF2; print('✅ PyPDF2 已安装')"
```

---

## Windows 安装

### 方案 A: WSL (推荐)

1. 安装 WSL2: `wsl --install`
2. 进入 WSL: `wsl`
3. 按照上面的 **Linux 安装** 步骤操作

### 方案 B: 原生 Windows

#### 1. 安装 Pandoc

下载并安装: https://github.com/jgm/pandoc/releases

#### 2. 安装 MiKTeX

下载: https://miktex.org/download

安装后运行 MiKTeX Console，安装以下包:
- `xetex`
- `xecjk`
- `ctex`

#### 3. 安装 Python 依赖

```bash
pip install PyPDF2 markdown
```

#### 4. 验证安装

```bash
pandoc --version
xelatex --version
python -c "import PyPDF2; print('✅ PyPDF2 已安装')"
```

---

## 常见问题

### Q1: XeLaTeX 找不到中文字体？

**macOS**: STSong 是系统自带字体，应该自动可用。

**Linux**: 需要安装中文字体:
```bash
# Ubuntu/Debian
sudo apt install fonts-arphic-uming fonts-wqy-zenhei

# 更新字体缓存
fc-cache -fv
```

**Windows**: 使用系统自带的宋体 (SimSun) 或微软雅黑。

### Q2: Pandoc 转换失败，提示 "Error producing PDF"？

检查 LaTeX 日志:
```bash
pandoc test.md -o test.pdf --pdf-engine=xelatex -V verbosity
```

常见原因:
- LaTeX 包缺失: `sudo tlmgr install <package>`
- 字体缺失: 检查中文字体安装
- 编码问题: 确保文件是 UTF-8 编码

### Q3: PyPDF2 合并 PDF 时报错？

确保 PDF 文件没有被其他程序占用:
```bash
# 检查文件是否被占用
lsof | grep <pdf-file>
```

### Q4: 想使用其他中文字体？

修改 `convert.py` 中的字体配置:
```python
'-V', 'CJKmainfont=Your-Font-Name',  # 替换为你的字体名
```

查看可用字体:
```bash
# macOS
fc-list :lang=zh

# Linux
fc-list :lang=zh

# Windows
# 在 C:\Windows\Fonts 中查看
```

---

## 卸载

### macOS
```bash
brew uninstall pandoc
# LaTeX 卸载（如果需要）
sudo rm -rf /usr/local/texlive
brew uninstall --cask mactex
pip3 uninstall PyPDF2 markdown
```

### Linux
```bash
sudo apt remove pandoc texlive-xetex texlive-lang-chinese
pip3 uninstall PyPDF2 markdown
```

---

## 更新

```bash
# 更新 Pandoc
brew upgrade pandoc  # macOS
sudo apt upgrade pandoc  # Linux

# 更新 LaTeX 包
sudo tlmgr update --self --all

# 更新 Python 包
pip3 install --upgrade PyPDF2 markdown
```

---

## 性能优化建议

### 首次编译较慢？

LaTeX 首次运行会生成格式文件，后续会快很多。可以预热:
```bash
# 预编译常用字体
fmtutil-sys --all
```

### 加速转换

1. **使用预编译格式**:
   ```bash
   pandoc input.md -o output.pdf --pdf-engine=xelatex \
     -V mainfont="STSong" \
     --resource-path=.
   ```

2. **批量转换时并行**:
   ```bash
   # 使用 GNU parallel
   ls *.md | parallel python3 convert.py {}
   ```

---

## 技术支持

- **Pandoc 文档**: https://pandoc.org/MANUAL.html
- **LaTeX 文档**: https://www.latex-project.org/help/
- **PyPDF2 文档**: https://pypdf2.readthedocs.io/
- **问题反馈**: 提交 Issue 到项目仓库
