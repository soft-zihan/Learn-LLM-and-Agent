#!/usr/bin/env python3
"""
批量 Markdown -> PDF (Pandoc + XeLaTeX)
学术级排版
"""

import sys
import os
from pathlib import Path
import subprocess

# 添加 LaTeX 到 PATH
os.environ['PATH'] = '/Library/TeX/texbin:' + os.environ['PATH']

def convert(md_file, pdf_file):
    """转换单个文件"""
    print(f"转换: {md_file.name}")
    
    md_dir = md_file.parent
    
    # Pandoc 命令
    cmd = [
        'pandoc',
        str(md_file),
        '-o', str(pdf_file),
        '--pdf-engine=xelatex',
        '-V', 'geometry:margin=2cm',
        '-V', 'CJKmainfont=STSong',
        '-V', 'monofont=Monaco',
        '--resource-path', f'.:{md_dir}/../comics',
    ]
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=120
        )
        
        if result.returncode == 0:
            print(f"✓ {pdf_file.name}")
            return True
        else:
            print(f" {md_file.name}")
            if result.stderr:
                print(f"  错误: {result.stderr[:200]}")
            return False
            
    except Exception as e:
        print(f"✗ {md_file.name}: {e}")
        return False

def main():
    if len(sys.argv) < 2:
        print("用法: python convert.py [目录] [输出目录]")
        print("示例: python convert.py . ./my-pdfs")
        sys.exit(1)
    
    dir_path = Path(sys.argv[1])
    
    # 支持自定义输出目录
    if len(sys.argv) >= 3:
        output = Path(sys.argv[2])
    else:
        output = dir_path / 'pdf'
    
    output.mkdir(exist_ok=True)
    
    files = sorted(dir_path.glob('*.md'))
    
    if not files:
        print("未找到 Markdown 文件!")
        sys.exit(0)
    
    print(f"找到 {len(files)} 个文件\n")
    print("=" * 60)
    
    success = failed = 0
    
    for f in files:
        pdf_file = output / f"{f.stem}.pdf"
        if convert(f, pdf_file):
            success += 1
        else:
            failed += 1
        print()
    
    print("=" * 60)
    print(f"完成! 成功: {success} | 失败: {failed}")
    print(f"PDF 保存在: {output}")

if __name__ == '__main__':
    main()
