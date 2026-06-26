#!/usr/bin/env python3
"""
合并所有 PDF 为一个文件（保留原有书签）
使用 PyPDF2 简单合并，保留每个 PDF 的原有书签结构
"""

import sys
import re
from pathlib import Path
from PyPDF2 import PdfMerger

def merge_pdfs(pdf_dir, output_file):
    """合并所有 PDF，保留原有书签"""
    print(f"合并 {pdf_dir} 中的所有 PDF...")
    
    # 获取所有 PDF 文件
    pdfs = list(pdf_dir.glob('*.pdf'))
    
    if not pdfs:
        print("未找到 PDF 文件!")
        return False
    
    # 使用自然排序（按数字前缀排序）
    def natural_sort_key(pdf_file):
        name = pdf_file.stem
        match = re.match(r'^(\d+)', name)
        if match:
            return int(match.group(1))
        return 999  # 没有数字前缀的排最后
    
    pdfs.sort(key=natural_sort_key)
    
    print(f"找到 {len(pdfs)} 个 PDF 文件")
    print()
    
    merger = PdfMerger()
    
    for pdf_file in pdfs:
        print(f"  合并: {pdf_file.name}")
        # 简单合并，保留原有书签，不添加新书签
        merger.append(str(pdf_file))
    
    # 写入输出文件
    merger.write(str(output_file))
    merger.close()
    
    print(f"\n✓ 合并完成: {output_file.name}")
    print(f"  文档数: {len(pdfs)} 个")
    print(f"  已保留原有书签结构")
    return True

def main():
    if len(sys.argv) < 2:
        print("用法: python merge.py [PDF目录] [输出文件]")
        print("示例: python merge.py ./pdf ./my-merged.pdf")
        sys.exit(1)
    
    pdf_dir = Path(sys.argv[1])
    
    if not pdf_dir.exists():
        print(f"目录不存在: {pdf_dir}")
        sys.exit(1)
    
    # 支持自定义输出文件
    if len(sys.argv) >= 3:
        output_file = Path(sys.argv[2])
    else:
        output_file = pdf_dir.parent / 'merged.pdf'
    
    if merge_pdfs(pdf_dir, output_file):
        print(f"\n合并文件: {output_file}")
    else:
        sys.exit(1)

if __name__ == '__main__':
    main()
