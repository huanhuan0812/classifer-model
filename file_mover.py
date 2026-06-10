#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
文件分类移动程序
根据分类结果将PPTX、DOCX文件移动到对应的类别文件夹
支持置信度阈值过滤
"""

import os
import sys
import shutil
import logging
import csv
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import yaml

# 导入分类器模块
from predict_onnx import (
    predict_file, load_models, _config as model_config,
    _text_tokenizer, _filename_tokenizer, _categories
)

# 设置日志
def setup_logging(config: dict):
    """配置日志系统"""
    log_config = config.get('logging', {})
    
    if not log_config.get('enabled', True):
        # 禁用日志
        logging.basicConfig(handlers=[logging.NullHandler()])
        return
    
    log_file = log_config.get('log_file', 'file_mover.log')
    log_level = getattr(logging, log_config.get('log_level', 'INFO').upper())
    
    # 创建日志目录
    Path(log_file).parent.mkdir(parents=True, exist_ok=True)
    
    # 配置日志
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )


class FileClassifierMover:
    """文件分类移动器"""
    
    def __init__(self, config_path: str = "file_classifier_config.yaml"):
        """初始化"""
        self.config = self.load_config(config_path)
        self.setup_paths()
        
        # 记录处理统计
        self.stats = {
            'total': 0,
            'moved': 0,
            'skipped_low_confidence': 0,
            'skipped_size_limit': 0,
            'skipped_empty_text': 0,
            'failed': 0,
            'already_exists': 0,
            'category_stats': {}
        }
        
        # 移动记录列表
        self.moved_records = []
        
        # 初始化模型
        print("正在加载分类模型...")
        load_models()
        print("模型加载完成！")
    
    def load_config(self, config_path: str) -> dict:
        """加载配置文件"""
        config_path = Path(config_path)
        
        if not config_path.exists():
            print(f"配置文件不存在: {config_path}")
            print("使用默认配置...")
            return self.get_default_config()
        
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            print(f"配置文件加载成功: {config_path}")
            return config
        except Exception as e:
            print(f"配置文件加载失败: {e}")
            print("使用默认配置...")
            return self.get_default_config()
    
    def get_default_config(self) -> dict:
        """获取默认配置"""
        return {
            'paths': {
                'source_dir': './input',
                'target_base_dir': './output'
            },
            'prediction': {
                'threshold': 0.7,
                'verbose': True,
                'supported_formats': ['.pptx', '.ppt', '.docx']
            },
            'file_handling': {
                'move_files': True,
                'create_category_dirs': True,
                'overwrite': False,
                'keep_original_name': True,
                'conflict_resolution': 'rename'
            },
            'categories': {
                'target_categories': [],
                'exclude_categories': [],
                'low_confidence_action': 'move_to_uncertain',
                'uncertain_folder_name': '_uncertain_low_confidence'
            },
            'logging': {
                'enabled': True,
                'log_file': 'file_mover.log',
                'log_level': 'INFO',
                'save_moved_list': True,
                'moved_list_file': 'moved_files.csv'
            },
            'advanced': {
                'max_file_size_mb': 100,
                'skip_empty_text': False,
                'recursive_scan': True
            }
        }
    
    def setup_paths(self):
        """设置路径"""
        paths = self.config['paths']
        self.source_dir = Path(paths['source_dir'])
        self.target_base_dir = Path(paths['target_base_dir'])
        
        # 检查源目录
        if not self.source_dir.exists():
            raise FileNotFoundError(f"源目录不存在: {self.source_dir}")
        
        # 创建目标基础目录
        if self.config['file_handling']['create_category_dirs']:
            self.target_base_dir.mkdir(parents=True, exist_ok=True)
        
        self.supported_formats = self.config['prediction']['supported_formats']
        self.threshold = self.config['prediction']['threshold']
        
        # 分类过滤
        self.target_categories = set(self.config['categories'].get('target_categories', []))
        self.exclude_categories = set(self.config['categories'].get('exclude_categories', []))
        
        # 文件处理选项
        self.move_files = self.config['file_handling']['move_files']
        self.overwrite = self.config['file_handling']['overwrite']
        self.keep_original_name = self.config['file_handling']['keep_original_name']
        self.conflict_resolution = self.config['file_handling']['conflict_resolution']
        
        # 其他选项
        self.max_file_size = self.config['advanced']['max_file_size_mb'] * 1024 * 1024
        self.skip_empty_text = self.config['advanced']['skip_empty_text']
        self.low_confidence_action = self.config['categories']['low_confidence_action']
        self.uncertain_folder = self.config['categories']['uncertain_folder_name']
        
        # 日志
        setup_logging(self.config)
        self.logger = logging.getLogger(__name__)
    
    def get_target_path(self, file_path: Path, category: str, confidence: float) -> Optional[Path]:
        """获取目标文件路径"""
        
        # 检查置信度阈值
        if confidence < self.threshold:
            if self.low_confidence_action == "move_to_uncertain":
                category = self.uncertain_folder
                self.stats['skipped_low_confidence'] += 1
                self.logger.info(f"  置信度不足 {confidence:.2%} < {self.threshold:.2%}，移至不确定文件夹")
            else:
                self.logger.info(f"  置信度不足 {confidence:.2%} < {self.threshold:.2%}，跳过")
                return None
        
        # 检查类别过滤
        if self.target_categories and category not in self.target_categories:
            self.logger.info(f"  类别 {category} 不在目标类别列表中，跳过")
            return None
        
        if category in self.exclude_categories:
            self.logger.info(f"  类别 {category} 在排除列表中，跳过")
            return None
        
        # 构建目标路径
        target_dir = self.target_base_dir / category
        target_dir.mkdir(parents=True, exist_ok=True)
        
        if self.keep_original_name:
            target_path = target_dir / file_path.name
        else:
            # 使用时间戳重命名
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            new_name = f"{category}_{timestamp}_{file_path.name}"
            target_path = target_dir / new_name
        
        # 处理文件名冲突
        if target_path.exists():
            if self.overwrite:
                self.logger.info(f"  覆盖已存在文件: {target_path.name}")
                return target_path
            elif self.conflict_resolution == "rename":
                # 重命名文件
                counter = 1
                stem = target_path.stem
                suffix = target_path.suffix
                while target_path.exists():
                    new_name = f"{stem}_{counter}{suffix}"
                    target_path = target_dir / new_name
                    counter += 1
                self.logger.info(f"  文件已存在，重命名为: {target_path.name}")
            else:
                self.logger.info(f"  文件已存在，跳过: {target_path.name}")
                self.stats['already_exists'] += 1
                return None
        
        return target_path
    
    def check_file_size(self, file_path: Path) -> bool:
        """检查文件大小是否超过限制"""
        if self.max_file_size <= 0:
            return True
        
        file_size = file_path.stat().st_size
        if file_size > self.max_file_size:
            self.logger.warning(f"  文件过大 ({file_size / 1024 / 1024:.2f}MB > {self.max_file_size / 1024 / 1024:.0f}MB)")
            self.stats['skipped_size_limit'] += 1
            return False
        
        return True
    
    def process_single_file(self, file_path: Path) -> Tuple[bool, Optional[str], Optional[float]]:
        """处理单个文件"""
        try:
            # 检查文件大小
            if not self.check_file_size(file_path):
                return False, None, None
            
            # 预测分类
            predicted_class, confidence, info = predict_file(
                str(file_path), 
                verbose=self.config['prediction']['verbose']
            )
            
            # 检查是否提取到文本
            if self.skip_empty_text and not info.get('text_available', False):
                self.logger.info(f"  跳过: 未提取到文本内容")
                self.stats['skipped_empty_text'] += 1
                return False, None, None
            
            # 获取目标路径
            target_path = self.get_target_path(file_path, predicted_class, confidence)
            
            if target_path is None:
                return False, predicted_class, confidence
            
            # 执行移动或复制
            if self.move_files:
                shutil.move(str(file_path), str(target_path))
                action = "移动"
            else:
                shutil.copy2(str(file_path), str(target_path))
                action = "复制"
            
            self.logger.info(f"  {action}: {file_path.name} -> {target_path}")
            
            # 记录统计
            self.stats['moved'] += 1
            self.stats['category_stats'][predicted_class] = \
                self.stats['category_stats'].get(predicted_class, 0) + 1
            
            # 记录移动信息
            self.moved_records.append({
                'source': str(file_path),
                'target': str(target_path),
                'category': predicted_class,
                'confidence': confidence,
                'text_available': info.get('text_available', False),
                'embedded_used': info.get('embedded_used', False),
                'file_type': file_path.suffix,
                'action': action,
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            })
            
            return True, predicted_class, confidence
            
        except Exception as e:
            self.logger.error(f"  处理失败: {file_path.name} - {e}")
            self.stats['failed'] += 1
            return False, None, None
    
    def find_files(self) -> List[Path]:
        """查找所有需要处理的文件"""
        files = []
        
        # 支持的扩展名（不区分大小写）
        extensions = [ext.lower() for ext in self.supported_formats]
        
        if self.config['advanced']['recursive_scan']:
            # 递归搜索
            for ext in extensions:
                files.extend(self.source_dir.rglob(f"*{ext}"))
                files.extend(self.source_dir.rglob(f"*{ext.upper()}"))
        else:
            # 仅搜索当前目录
            for ext in extensions:
                files.extend(self.source_dir.glob(f"*{ext}"))
                files.extend(self.source_dir.glob(f"*{ext.upper()}"))
        
        # 去重
        files = list(set(files))
        
        return files
    
    def save_moved_list(self):
        """保存移动文件列表"""
        if not self.config['logging'].get('save_moved_list', True):
            return
        
        if not self.moved_records:
            self.logger.info("没有移动/复制的文件记录")
            return
        
        moved_list_file = self.config['logging'].get('moved_list_file', 'moved_files.csv')
        moved_list_path = Path(moved_list_file)
        
        try:
            with open(moved_list_path, 'w', newline='', encoding='utf-8-sig') as f:
                fieldnames = ['source', 'target', 'category', 'confidence', 
                             'text_available', 'embedded_used', 'file_type', 
                             'action', 'timestamp']
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(self.moved_records)
            
            self.logger.info(f"移动记录已保存到: {moved_list_path}")
        except Exception as e:
            self.logger.error(f"保存移动记录失败: {e}")
    
    def print_summary(self):
        """打印处理摘要"""
        print("\n" + "="*70)
        print("处理完成 - 统计摘要")
        print("="*70)
        print(f"📊 总计扫描: {self.stats['total']} 个文件")
        print(f"✅ 成功处理: {self.stats['moved']} 个")
        print(f"❌ 处理失败: {self.stats['failed']} 个")
        
        if self.stats['skipped_low_confidence'] > 0:
            print(f"⚠️  置信度不足: {self.stats['skipped_low_confidence']} 个")
        if self.stats['skipped_size_limit'] > 0:
            print(f"⚠️  超过大小限制: {self.stats['skipped_size_limit']} 个")
        if self.stats['skipped_empty_text'] > 0:
            print(f"⚠️  无文本内容: {self.stats['skipped_empty_text']} 个")
        if self.stats['already_exists'] > 0:
            print(f"⚠️  文件已存在: {self.stats['already_exists']} 个")
        
        if self.stats['category_stats']:
            print("\n📁 类别分布:")
            for category, count in sorted(self.stats['category_stats'].items(), 
                                         key=lambda x: x[1], reverse=True):
                percentage = count / self.stats['moved'] * 100 if self.stats['moved'] > 0 else 0
                print(f"   {category:10s}: {count:4d} 个 ({percentage:5.1f}%)")
        
        # 记录到日志
        self.logger.info(f"处理完成 - 总计:{self.stats['total']}, 成功:{self.stats['moved']}, 失败:{self.stats['failed']}")
    
    def run(self):
        """运行主流程"""
        print("\n" + "="*70)
        print("文件分类移动程序")
        print("="*70)
        print(f"源目录: {self.source_dir}")
        print(f"目标目录: {self.target_base_dir}")
        print(f"置信度阈值: {self.threshold:.0%}")
        print(f"操作模式: {'移动' if self.move_files else '复制'}")
        print("="*70)
        
        # 查找文件
        files = self.find_files()
        self.stats['total'] = len(files)
        
        if not files:
            print(f"\n未找到支持的文件 (格式: {', '.join(self.supported_formats)})")
            return
        
        print(f"\n找到 {len(files)} 个文件，开始处理...\n")
        
        # 处理每个文件
        for i, file_path in enumerate(files, 1):
            print(f"\n[{i}/{len(files)}] 处理: {file_path.name}")
            self.process_single_file(file_path)
        
        # 保存记录
        self.save_moved_list()
        
        # 打印统计
        self.print_summary()
    
    def preview(self):
        """预览模式：只显示将要处理的文件分类结果，不实际移动"""
        print("\n" + "="*70)
        print("预览模式 - 仅显示分类结果，不实际移动文件")
        print("="*70)
        
        files = self.find_files()
        self.stats['total'] = len(files)
        
        if not files:
            print(f"\n未找到支持的文件")
            return
        
        print(f"\n找到 {len(files)} 个文件，预览分类结果...\n")
        
        results = []
        for i, file_path in enumerate(files, 1):
            print(f"\n[{i}/{len(files)}] {file_path.name}")
            try:
                predicted_class, confidence, info = predict_file(
                    str(file_path), 
                    verbose=False
                )
                
                status = ""
                if confidence < self.threshold:
                    status = " [置信度不足]"
                
                print(f"   → {predicted_class} (置信度: {confidence:.2%}){status}")
                if info.get('embedded_used'):
                    print(f"   📎 包含嵌入的DOCX文件")
                
                results.append({
                    'file': file_path.name,
                    'predicted_class': predicted_class,
                    'confidence': confidence,
                    'meets_threshold': confidence >= self.threshold,
                    'embedded_used': info.get('embedded_used', False)
                })
                
                self.stats['category_stats'][predicted_class] = \
                    self.stats['category_stats'].get(predicted_class, 0) + 1
                    
            except Exception as e:
                print(f"   ❌ 预测失败: {e}")
                self.stats['failed'] += 1
        
        # 打印摘要
        self.print_summary()
        
        # 可选：保存预览结果
        save_preview = input("\n是否保存预览结果到CSV? (y/N): ").strip().lower()
        if save_preview == 'y':
            preview_file = Path("preview_results.csv")
            with open(preview_file, 'w', newline='', encoding='utf-8-sig') as f:
                writer = csv.DictWriter(f, fieldnames=['file', 'predicted_class', 
                                                       'confidence', 'meets_threshold', 
                                                       'embedded_used'])
                writer.writeheader()
                writer.writerows(results)
            print(f"预览结果已保存到: {preview_file}")


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='文件分类移动工具')
    parser.add_argument('--config', '-c', default='file_classifier_config.yaml',
                       help='配置文件路径 (默认: file_classifier_config.yaml)')
    parser.add_argument('--preview', '-p', action='store_true',
                       help='预览模式，只显示分类结果不移动文件')
    parser.add_argument('--source', '-s', help='源目录（覆盖配置文件中的设置）')
    parser.add_argument('--target', '-t', help='目标目录（覆盖配置文件中的设置）')
    parser.add_argument('--threshold', '-th', type=float, 
                       help='置信度阈值（覆盖配置文件中的设置）')
    
    args = parser.parse_args()
    
    try:
        # 创建移动器
        mover = FileClassifierMover(args.config)
        
        # 覆盖命令行参数
        if args.source:
            mover.source_dir = Path(args.source)
        if args.target:
            mover.target_base_dir = Path(args.target)
        if args.threshold is not None:
            mover.threshold = args.threshold
        
        if args.preview:
            mover.preview()
        else:
            # 确认执行
            print(f"\n将{'移动' if mover.move_files else '复制'}符合条件的文件到类别文件夹")
            confirm = input("是否继续? (y/N): ").strip().lower()
            if confirm == 'y':
                mover.run()
            else:
                print("已取消")
                
    except KeyboardInterrupt:
        print("\n\n用户中断")
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())