#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
文件分类移动程序
根据分类结果将PPTX、DOCX文件移动到对应的类别文件夹
支持置信度阈值过滤
支持临时文件清理
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
            'skipped_temp_file': 0,  # 跳过的临时文件计数
            'cleaned_temp_files': 0,  # 🆕 清理的临时文件计数
            'failed': 0,
            'already_exists': 0,
            'category_stats': {}
        }
        
        # 移动记录列表
        self.moved_records = []
        
        # 临时文件清理记录
        self.cleaned_records = []
        
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
            'temp_file_cleanup': {  # 🆕 临时文件清理配置
                'enabled': False,  # 是否启用清理
                'action': 'delete',  # 清理动作: 'delete' 或 'move'
                'target_dir': './temp_cleanup',  # 移动模式下的目标目录
                'patterns': ['~$*', '*.tmp', '*~'],  # 要清理的文件模式
                'min_file_age_minutes': 0,  # 最小文件年龄(分钟)，0表示不限制
                'clean_on_start': True,  # 处理前清理
                'clean_on_end': True,    # 处理后清理
                'dry_run': False,        # 试运行模式，不实际删除/移动
                'keep_empty_dirs': False  # 清理后是否保留空目录
            },
            'logging': {
                'enabled': True,
                'log_file': 'file_mover.log',
                'log_level': 'INFO',
                'save_moved_list': True,
                'moved_list_file': 'moved_files.csv',
                'save_cleanup_list': True,  # 🆕 保存清理记录
                'cleanup_list_file': 'cleaned_files.csv'  # 🆕 清理记录文件
            },
            'advanced': {
                'max_file_size_mb': 100,
                'skip_empty_text': False,
                'recursive_scan': True,
                'skip_temp_files': True  # 是否跳过临时文件（不进行分类）
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
        self.skip_temp_files = self.config['advanced'].get('skip_temp_files', True)
        
        # 🆕 临时文件清理配置
        self.temp_cleanup_config = self.config.get('temp_file_cleanup', {})
        self.cleanup_enabled = self.temp_cleanup_config.get('enabled', False)
        self.cleanup_action = self.temp_cleanup_config.get('action', 'delete')
        self.cleanup_target_dir = Path(self.temp_cleanup_config.get('target_dir', './temp_cleanup'))
        self.cleanup_patterns = self.temp_cleanup_config.get('patterns', ['~$*', '*.tmp', '*~'])
        self.min_file_age_minutes = self.temp_cleanup_config.get('min_file_age_minutes', 0)
        self.clean_on_start = self.temp_cleanup_config.get('clean_on_start', True)
        self.clean_on_end = self.temp_cleanup_config.get('clean_on_end', True)
        self.dry_run = self.temp_cleanup_config.get('dry_run', False)
        self.keep_empty_dirs = self.temp_cleanup_config.get('keep_empty_dirs', False)
        
        # 日志
        setup_logging(self.config)
        self.logger = logging.getLogger(__name__)
        
        # 如果启用清理且动作是移动，创建目标目录
        if self.cleanup_enabled and self.cleanup_action == 'move':
            self.cleanup_target_dir.mkdir(parents=True, exist_ok=True)
    
    def is_temp_file(self, file_path: Path) -> bool:
        """
        判断是否为临时文件
        临时文件特征：
        1. 以 ~$ 开头（Office临时文件）
        2. 以 .tmp 结尾
        3. 文件名包含 "~" 字符
        """
        if not self.skip_temp_files:
            return False
        
        file_name = file_path.name
        
        # 检查 ~$ 开头的文件（Office临时文件）
        if file_name.startswith('~$'):
            self.logger.debug(f"识别为临时文件 (Office临时文件): {file_name}")
            return True
        
        # 检查 .tmp 结尾的临时文件
        if file_name.lower().endswith('.tmp'):
            self.logger.debug(f"识别为临时文件 (.tmp文件): {file_name}")
            return True
        
        # 检查包含 ~ 的文件（通常是备份或临时文件）
        if '~' in file_name and not file_name.endswith(('.pptx', '.docx', '.ppt', '.doc')):
            self.logger.debug(f"识别为临时文件 (包含~字符): {file_name}")
            return True
        
        return False
    
    def should_cleanup_file(self, file_path: Path) -> bool:
        """
        判断文件是否应该被清理
        根据配置的patterns和文件年龄
        """
        if not self.cleanup_enabled:
            return False
        
        file_name = file_path.name
        
        # 检查是否匹配清理模式
        matched = False
        for pattern in self.cleanup_patterns:
            if pattern.startswith('~$'):
                # 处理 ~$* 模式
                if pattern == '~$*' and file_name.startswith('~$'):
                    matched = True
                    break
            elif pattern.endswith('*'):
                # 处理前缀匹配
                prefix = pattern[:-1]
                if file_name.startswith(prefix):
                    matched = True
                    break
            elif pattern.startswith('*'):
                # 处理后缀匹配
                suffix = pattern[1:]
                if file_name.endswith(suffix):
                    matched = True
                    break
            else:
                # 精确匹配或使用fnmatch
                from fnmatch import fnmatch
                if fnmatch(file_name, pattern):
                    matched = True
                    break
        
        if not matched:
            return False
        
        # 检查文件年龄
        if self.min_file_age_minutes > 0:
            try:
                file_mtime = datetime.fromtimestamp(file_path.stat().st_mtime)
                file_age = (datetime.now() - file_mtime).total_seconds() / 60
                if file_age < self.min_file_age_minutes:
                    self.logger.debug(f"文件年龄不足 ({file_age:.1f}分钟 < {self.min_file_age_minutes}分钟)，跳过清理: {file_name}")
                    return False
            except Exception as e:
                self.logger.warning(f"检查文件年龄失败: {e}")
        
        return True
    
    def cleanup_temp_files(self, scan_directory: Path = None, is_startup: bool = False):
        """
        清理临时文件
        :param scan_directory: 要扫描的目录，默认为源目录
        :param is_startup: 是否为启动时清理（用于日志显示）
        """
        if not self.cleanup_enabled:
            return
        
        if is_startup and not self.clean_on_start:
            return
        if not is_startup and not self.clean_on_end:
            return
        
        scan_dir = scan_directory or self.source_dir
        phase = "启动时" if is_startup else "完成后"
        
        if self.dry_run:
            self.logger.info(f"🔍 {phase}临时文件清理 (试运行模式 - 不实际{'删除' if self.cleanup_action == 'delete' else '移动'}文件)")
        else:
            self.logger.info(f"🧹 {phase}临时文件清理 (动作: {'删除' if self.cleanup_action == 'delete' else f'移动到 {self.cleanup_target_dir}'})")
        
        # 查找所有临时文件
        temp_files = []
        
        # 递归查找匹配模式的文件
        for pattern in self.cleanup_patterns:
            try:
                # 转换pattern为glob格式
                glob_pattern = pattern.replace('*', '*')
                found = list(scan_dir.rglob(glob_pattern))
                for f in found:
                    if f.is_file() and self.should_cleanup_file(f):
                        temp_files.append(f)
            except Exception as e:
                self.logger.debug(f"搜索模式 {pattern} 失败: {e}")
        
        # 去重
        temp_files = list(set(temp_files))
        
        if not temp_files:
            self.logger.info(f"  未找到需要清理的临时文件")
            return
        
        self.logger.info(f"  找到 {len(temp_files)} 个临时文件")
        
        # 处理每个临时文件
        for temp_file in temp_files:
            try:
                if self.dry_run:
                    # 试运行模式，只记录
                    self.logger.info(f"    [试运行] 将{'删除' if self.cleanup_action == 'delete' else '移动'}: {temp_file}")
                    self.stats['cleaned_temp_files'] += 1
                    self.cleaned_records.append({
                        'file': str(temp_file),
                        'action': self.cleanup_action,
                        'status': 'dry_run',
                        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    })
                    continue
                
                if self.cleanup_action == 'delete':
                    # 删除文件
                    temp_file.unlink()
                    self.logger.info(f"    ✅ 已删除: {temp_file.name}")
                    action_str = "删除"
                    
                elif self.cleanup_action == 'move':
                    # 移动到指定目录
                    # 保持相对路径结构
                    rel_path = temp_file.relative_to(scan_dir)
                    target_path = self.cleanup_target_dir / rel_path
                    target_path.parent.mkdir(parents=True, exist_ok=True)
                    
                    # 处理文件名冲突
                    if target_path.exists():
                        counter = 1
                        stem = target_path.stem
                        suffix = target_path.suffix
                        while target_path.exists():
                            new_name = f"{stem}_{counter}{suffix}"
                            target_path = target_path.parent / new_name
                            counter += 1
                    
                    shutil.move(str(temp_file), str(target_path))
                    self.logger.info(f"    ✅ 已移动: {temp_file.name} -> {target_path}")
                    action_str = "移动"
                
                self.stats['cleaned_temp_files'] += 1
                self.cleaned_records.append({
                    'file': str(temp_file),
                    'target': str(target_path) if self.cleanup_action == 'move' else '',
                    'action': action_str,
                    'status': 'success',
                    'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                })
                
            except Exception as e:
                self.logger.error(f"    ❌ 处理失败 {temp_file.name}: {e}")
                self.cleaned_records.append({
                    'file': str(temp_file),
                    'action': self.cleanup_action,
                    'status': f'failed: {e}',
                    'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                })
        
        # 清理空目录（如果不保留空目录）
        if not self.keep_empty_dirs and not self.dry_run and self.cleanup_action == 'delete':
            self._remove_empty_directories(scan_dir)
        
        self.logger.info(f"  {phase}临时文件清理完成，共处理 {self.stats['cleaned_temp_files']} 个文件")
    
    def _remove_empty_directories(self, directory: Path):
        """递归删除空目录"""
        try:
            # 从最深层开始删除
            for dirpath in sorted(directory.rglob('*'), key=lambda p: len(p.parts), reverse=True):
                if dirpath.is_dir():
                    try:
                        if not any(dirpath.iterdir()):
                            dirpath.rmdir()
                            self.logger.debug(f"删除空目录: {dirpath}")
                    except Exception:
                        pass
        except Exception as e:
            self.logger.debug(f"清理空目录时出错: {e}")
    
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
            # 检查是否为临时文件
            if self.is_temp_file(file_path):
                self.stats['skipped_temp_file'] += 1
                self.logger.info(f"  ⏭️  跳过临时文件: {file_path.name}")
                return False, None, None
            
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
        """查找所有需要处理的文件（自动过滤临时文件）"""
        files = []
        
        # 支持的扩展名（不区分大小写）
        extensions = [ext.lower() for ext in self.supported_formats]
        
        if self.config['advanced']['recursive_scan']:
            # 递归搜索
            for ext in extensions:
                found_files = list(self.source_dir.rglob(f"*{ext}"))
                found_files.extend(self.source_dir.rglob(f"*{ext.upper()}"))
                
                # 过滤临时文件
                for f in found_files:
                    if not self.is_temp_file(f):
                        files.append(f)
                    else:
                        self.logger.debug(f"搜索时跳过临时文件: {f.name}")
        else:
            # 仅搜索当前目录
            for ext in extensions:
                found_files = list(self.source_dir.glob(f"*{ext}"))
                found_files.extend(self.source_dir.glob(f"*{ext.upper()}"))
                
                # 过滤临时文件
                for f in found_files:
                    if not self.is_temp_file(f):
                        files.append(f)
                    else:
                        self.logger.debug(f"搜索时跳过临时文件: {f.name}")
        
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
    
    def save_cleanup_list(self):
        """保存临时文件清理记录"""
        if not self.config['logging'].get('save_cleanup_list', True):
            return
        
        if not self.cleaned_records:
            self.logger.info("没有临时文件清理记录")
            return
        
        cleanup_list_file = self.config['logging'].get('cleanup_list_file', 'cleaned_files.csv')
        cleanup_list_path = Path(cleanup_list_file)
        
        try:
            with open(cleanup_list_path, 'w', newline='', encoding='utf-8-sig') as f:
                fieldnames = ['file', 'target', 'action', 'status', 'timestamp']
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(self.cleaned_records)
            
            self.logger.info(f"清理记录已保存到: {cleanup_list_path}")
        except Exception as e:
            self.logger.error(f"保存清理记录失败: {e}")
    
    def print_summary(self):
        """打印处理摘要"""
        print("\n" + "="*70)
        print("处理完成 - 统计摘要")
        print("="*70)
        print(f"📊 总计扫描: {self.stats['total']} 个文件")
        print(f"✅ 成功处理: {self.stats['moved']} 个")
        print(f"❌ 处理失败: {self.stats['failed']} 个")
        
        if self.stats['skipped_temp_file'] > 0:
            print(f"⏭️  临时文件跳过: {self.stats['skipped_temp_file']} 个")
        if self.stats['cleaned_temp_files'] > 0:
            print(f"🧹 临时文件清理: {self.stats['cleaned_temp_files']} 个")
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
        self.logger.info(f"处理完成 - 总计:{self.stats['total']}, 成功:{self.stats['moved']}, 失败:{self.stats['failed']}, 临时文件跳过:{self.stats['skipped_temp_file']}, 临时文件清理:{self.stats['cleaned_temp_files']}")
    
    def run(self):
        """运行主流程"""
        print("\n" + "="*70)
        print("文件分类移动程序")
        print("="*70)
        print(f"源目录: {self.source_dir}")
        print(f"目标目录: {self.target_base_dir}")
        print(f"置信度阈值: {self.threshold:.0%}")
        print(f"操作模式: {'移动' if self.move_files else '复制'}")
        print(f"临时文件过滤: {'启用' if self.skip_temp_files else '禁用'}")
        if self.cleanup_enabled:
            print(f"🧹 临时文件清理: {'启用' if self.cleanup_enabled else '禁用'} ({self.cleanup_action})")
            if self.dry_run:
                print(f"   ⚠️  试运行模式 - 不会实际删除/移动文件")
        print("="*70)
        
        # 启动时清理临时文件
        if self.cleanup_enabled and self.clean_on_start:
            self.cleanup_temp_files(is_startup=True)
        
        # 查找文件
        files = self.find_files()
        self.stats['total'] = len(files)
        
        if not files:
            print(f"\n未找到支持的文件 (格式: {', '.join(self.supported_formats)})")
        else:
            print(f"\n找到 {len(files)} 个文件，开始处理...\n")
            
            # 处理每个文件
            for i, file_path in enumerate(files, 1):
                print(f"\n[{i}/{len(files)}] 处理: {file_path.name}")
                self.process_single_file(file_path)
        
        # 完成后清理临时文件
        if self.cleanup_enabled and self.clean_on_end:
            self.cleanup_temp_files(is_startup=False)
        
        # 保存记录
        self.save_moved_list()
        self.save_cleanup_list()
        
        # 打印统计
        self.print_summary()
    
    def preview(self):
        """预览模式：只显示将要处理的文件分类结果，不实际移动"""
        print("\n" + "="*70)
        print("预览模式 - 仅显示分类结果，不实际移动文件")
        print("="*70)
        
        # 显示清理配置
        if self.cleanup_enabled:
            print(f"🧹 临时文件清理: {'启用' if self.cleanup_enabled else '禁用'} ({self.cleanup_action})")
            if self.clean_on_start:
                print(f"   ⚠️  启动时会清理临时文件（预览模式不会执行）")
        
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
                    'full_path': str(file_path),
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
                writer = csv.DictWriter(f, fieldnames=['file', 'full_path', 'predicted_class', 
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
    parser.add_argument('--include-temp', action='store_true',
                       help='包含临时文件（默认跳过~$开头的临时文件）')
    parser.add_argument('--cleanup', action='store_true',
                       help='启用临时文件清理（覆盖配置文件设置）')
    parser.add_argument('--cleanup-action', choices=['delete', 'move'],
                       help='清理动作: delete(删除) 或 move(移动)')
    parser.add_argument('--dry-run', action='store_true',
                       help='试运行模式，只显示将要清理的文件，不实际执行')
    
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
        if args.include_temp:
            mover.skip_temp_files = False
        if args.cleanup:
            mover.cleanup_enabled = True
        if args.cleanup_action:
            mover.cleanup_action = args.cleanup_action
        if args.dry_run:
            mover.dry_run = True
            mover.cleanup_enabled = True  # 试运行模式自动启用清理
        
        if args.preview:
            mover.preview()
        else:
            # 确认执行
            print(f"\n将{'移动' if mover.move_files else '复制'}符合条件的文件到类别文件夹")
            if mover.skip_temp_files:
                print("⚠️  临时文件（~$开头）将被自动跳过")
            if mover.cleanup_enabled:
                if mover.dry_run:
                    print(f"🧹 试运行模式: 将显示要清理的临时文件，不实际{'删除' if mover.cleanup_action == 'delete' else '移动'}")
                else:
                    print(f"🧹 临时文件清理: 将{ '删除' if mover.cleanup_action == 'delete' else f'移动到 {mover.cleanup_target_dir}'} 临时文件")
            
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