#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试临时文件清理功能
"""

from pathlib import Path
import time
from datetime import datetime

def create_test_files():
    """创建测试文件"""
    test_dir = Path("./test_cleanup")
    test_dir.mkdir(exist_ok=True)
    
    # 创建不同类型的临时文件
    test_files = [
        test_dir / "~$正常文件.pptx",      # Office临时文件
        test_dir / "temp_file.tmp",         # .tmp文件
        test_dir / "backup~file.pptx",      # 包含~的文件
        test_dir / "document.bak",          # 备份文件
        test_dir / ".DS_Store",             # macOS文件
        test_dir / "normal_file.pptx",      # 正常文件
        test_dir / "subdir/~$subfile.pptx", # 子目录中的临时文件
    ]
    
    # 创建子目录
    (test_dir / "subdir").mkdir(exist_ok=True)
    
    for file_path in test_files:
        file_path.touch()
        print(f"创建: {file_path}")
    
    print(f"\n测试目录: {test_dir}")
    print(f"包含 {len(test_files)} 个文件（其中5个是临时文件）")
    print("\n运行以下命令测试:")
    print("1. 试运行模式: python file_mover.py --source ./test_cleanup --dry-run")
    print("2. 实际清理: python file_mover.py --source ./test_cleanup --cleanup")
    print("3. 移动模式: python file_mover.py --source ./test_cleanup --cleanup --cleanup-action move")

if __name__ == "__main__":
    create_test_files()