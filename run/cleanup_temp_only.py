#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
独立的临时文件清理工具
可以单独运行清理临时文件，不进行分类
"""

import argparse
from pathlib import Path
import yaml
from file_mover import FileClassifierMover

def main():
    parser = argparse.ArgumentParser(description='临时文件清理工具')
    parser.add_argument('--directory', '-d', default='./input',
                       help='要清理的目录 (默认: ./input)')
    parser.add_argument('--action', '-a', choices=['delete', 'move'], 
                       default='delete', help='清理动作')
    parser.add_argument('--dry-run', action='store_true',
                       help='试运行模式')
    parser.add_argument('--age', type=int, default=0,
                       help='最小文件年龄(分钟)')
    
    args = parser.parse_args()
    
    # 创建临时配置
    temp_config = {
        'paths': {'source_dir': args.directory, 'target_base_dir': './output'},
        'temp_file_cleanup': {
            'enabled': True,
            'action': args.action,
            'target_dir': './temp_cleanup',
            'patterns': ['~$*', '*.tmp', '*~', '*.bak', '*.swp', '*.DS_Store'],
            'min_file_age_minutes': args.age,
            'clean_on_start': True,
            'clean_on_end': False,
            'dry_run': args.dry_run,
            'keep_empty_dirs': False
        },
        'advanced': {'skip_temp_files': True},
        'logging': {'enabled': True, 'log_level': 'INFO'}
    }
    
    # 保存临时配置
    temp_config_file = Path("temp_cleanup_config.yaml")
    with open(temp_config_file, 'w', encoding='utf-8') as f:
        yaml.dump(temp_config, f)
    
    # 运行清理
    mover = FileClassifierMover(str(temp_config_file))
    mover.cleanup_temp_files(is_startup=True)
    
    # 清理临时配置文件
    temp_config_file.unlink()

if __name__ == "__main__":
    main()