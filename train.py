#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PPTX学科分类器训练脚本 (TextCNN) - 优化版
支持8类别：语文、数学、英语、物理、化学、生物、班会
核心特性：
1. 文本权重70%，文件名权重30%（增强文本重要性）
2. 文件名数据增强（模拟不同命名习惯）
3. 多尺度卷积提取文本特征（增加kernel_size=2）
4. 🔥 PPTX解析缓存（避免重复解析，大幅加速）
5. 类别权重处理（解决小样本类别不平衡）
6. 🎯 支持指定科目只使用缓存（跳过解析，仅从缓存读取）
7. 👤 人名去除（在缓存阶段使用jieba识别并过滤人名）
8. 📊 词表截断（限制在MAX_NB_WORDS，如20000词）
9. 💾 自动保存无依赖的Tokenizer（用于推理）
10. 📝 输出JSON格式词表（便于外部工具使用）
"""

import os
import re
import pickle
import json
import random
import numpy as np
from pathlib import Path
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight

# 在导入tensorflow之前设置环境变量
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Embedding, Conv1D, GlobalMaxPooling1D, Dense, Dropout, Input, Concatenate
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical
import jieba
import jieba.posseg as pseg
from pptx import Presentation

# ========== 优化后的配置参数 ==========
DATA_ROOT = "../data"
CACHE_DIR = "../cache"
CATEGORIES = ["语文", "数学", "英语", "物理", "化学", "生物", "班会"]

# 🎯 指定只使用缓存的科目（跳过解析，仅从已有缓存读取）
SKIP_CATEGORIES = []  # 例如: ["语文", "数学"] 表示语文和数学只从缓存读取

# 词向量配置（增强语义表示）
MAX_NB_WORDS = 20000          # 词表最大大小（在20000词处截断）
MAX_SEQUENCE_LENGTH = 1000    # 增加序列长度
EMBEDDING_DIM = 150           # 增强Embedding维度

# 文件名特征配置
MAX_FILENAME_LENGTH = 32
FILENAME_EMBEDDING_DIM = 32
MAX_FILENAME_WORDS = 5000     # 文件名词表最大大小

# 平衡权重（增强文本重要性）
TEXT_WEIGHT = 0.70            # 提高文本权重
FILENAME_WEIGHT = 0.30        # 降低文件名权重

# 数据增强配置（减少噪声）
ENABLE_FILENAME_AUGMENTATION = True
FILENAME_AUGMENTATION_COUNT = 3  # 减少变体数量

# 训练配置
BATCH_SIZE = 12               # 更精细的梯度更新
EPOCHS = 40                   # 更多训练轮次
VALIDATION_SPLIT = 0.15
TEST_SPLIT = 0.15

# 交叉验证配置
ENABLE_CROSS_VALIDATION = False
CV_FOLDS = 5

# 缓存配置
ENABLE_CACHE = True
CACHE_VERSION = "v4"          # 更新缓存版本（因为添加了人名去除功能）
FORCE_REFRESH_CACHE = False

# 类别权重配置
USE_CLASS_WEIGHTS = True      # 启用类别权重（处理小样本）

# 人名去除配置
REMOVE_PERSON_NAMES = True    # 是否去除人名
KEEP_SINGLE_CHAR_NAMES = False # 是否保留单字人名（通常单字可能是误判）

# 停用词表
STOPWORDS = set([
    '的', '了', '是', '我', '你', '他', '她', '它', '我们', '你们', '他们',
    '这', '那', '有', '在', '不', '和', '与', '就', '都', '而', '及', '或',
    '一个', '这个', '那个', '这些', '那些', '这里', '那里', '然后', '因为',
    '所以', '但是', '如果', '虽然', '然而', '并且', '或者'
])

# 有意义的单字词
MEANINGFUL_SINGLE_CHARS = {'圆', '力', '氧', '氢', '碳', '钠', '酸', '碱', '盐', 
                           '电', '光', '声', '热', '诗', '词', '歌', '曲', '数',
                           '方', '程', '函', '数', '角', '形', '体', '积'}

# 常见姓氏（用于人名识别增强）
COMMON_SURNAMES = set([
    '李', '王', '张', '刘', '陈', '杨', '赵', '黄', '周', '吴', '徐', '孙', '马', '朱',
    '胡', '林', '郭', '何', '高', '郑', '罗', '梁', '谢', '宋', '唐', '邓', '萧', '冯',
    '韩', '曹', '彭', '曾', '肖', '田', '董', '潘', '袁', '于', '蒋', '蔡', '余', '杜',
    '苏', '吕', '丁', '沈', '任', '姚', '卢', '傅', '钟', '姜', '崔', '谭', '廖', '范',
    '汪', '陆', '金', '石', '戴', '贾', '韦', '夏', '邱', '方', '侯', '邹', '熊', '孟',
    '秦', '白', '江', '阎', '薛', '尹', '段', '雷', '黎', '史', '龙', '陶', '贺', '顾',
    '毛', '郝', '龚', '邵', '万', '钱', '严', '赖', '覃', '洪', '武', '莫', '孔', '汤',
    '向', '常', '温', '康', '施', '文', '牛', '樊', '葛', '邢'
])

# ---------- 人名识别和去除模块 ----------
class PersonNameRemover:
    """使用jieba词性标注识别并去除人名"""
    
    def __init__(self):
        # 加载jieba词性标注功能
        # 人名词性标签: nr, nr1, nr2, nrj, nrf, nrs, ns, nt, nz 中的人名相关
        self.person_tags = {'nr', 'nr1', 'nr2', 'nrj', 'nrf', 'nrs'}
        self._compile_name_patterns()
    
    def _compile_name_patterns(self):
        """编译人名相关的正则模式"""
        # 匹配常见姓氏开头的2-4字中文词
        surnames_pattern = '|'.join(re.escape(s) for s in COMMON_SURNAMES)
        self.surname_pattern = re.compile(f'[{surnames_pattern}][\\u4e00-\\u9fa5]{{1,3}}')
        
        # 匹配"XX老师"、"XX同学"等称呼模式
        self.title_pattern = re.compile(r'([\u4e00-\u9fa5]{2,4})(?:老师|同学|教授|博士|校长|主任|院长|书记)')
        
        # 匹配"XX说"、"XX认为"等动作模式
        self.action_pattern = re.compile(r'([\u4e00-\u9fa5]{2,4})(?:说|道|问|答|认为|表示|指出|强调|建议|提出)')
    
    def remove_names_by_pos(self, text):
        """
        使用jieba词性标注去除人名
        返回去除人名后的文本
        """
        if not text or not REMOVE_PERSON_NAMES:
            return text
        
        try:
            words = pseg.cut(text)
            filtered_words = []
            
            for word, flag in words:
                # 判断是否为人名
                is_person = flag in self.person_tags
                
                # 对于单字词，如果不是有意义单字且被标记为人名，则过滤
                if is_person:
                    if len(word) == 1 and KEEP_SINGLE_CHAR_NAMES:
                        # 检查是否为有意义单字
                        if word in MEANINGFUL_SINGLE_CHARS:
                            filtered_words.append(word)
                    # 否则跳过（不添加）
                else:
                    filtered_words.append(word)
            
            return ''.join(filtered_words)
        except Exception as e:
            # 如果词性标注失败，返回原文本
            print(f"    警告: 词性标注人名去除失败 ({e})")
            return text
    
    def remove_names_by_regex(self, text):
        """
        使用正则表达式辅助去除可能的人名
        作为词性标注的补充
        """
        if not text:
            return text
        
        result = text
        
        # 替换"XX老师"、"XX同学"等
        result = self.title_pattern.sub(r'\1', result)
        
        # 替换"XX说"、"XX认为"等
        result = self.action_pattern.sub(r'\1', result)
        
        # 移除单独的姓氏（但保留有意义的单字）
        def replace_surname(match):
            surname = match.group(0)
            if surname not in MEANINGFUL_SINGLE_CHARS:
                return ''
            return surname
        
        # 注意：这个匹配可能过于激进，只在特定上下文中使用
        # 这里只在文本较短或特定条件下使用
        
        return result
    
    def remove_all_names(self, text, use_regex=True):
        """
        综合使用多种方法去除人名
        优先使用词性标注，正则作为补充
        """
        if not text or not REMOVE_PERSON_NAMES:
            return text
        
        # 先使用词性标注去除
        result = self.remove_names_by_pos(text)
        
        # 可选：使用正则表达式进一步清理
        if use_regex:
            result = self.remove_names_by_regex(result)
        
        # 清理多余空格
        result = re.sub(r'\s+', ' ', result)
        
        return result.strip()


# 全局人名去除器实例
_name_remover = None

def get_name_remover():
    """获取全局人名去除器实例"""
    global _name_remover
    if _name_remover is None:
        _name_remover = PersonNameRemover()
    return _name_remover


def remove_person_names_from_text(text):
    """
    从文本中去除人名（缓存阶段使用）
    """
    if not REMOVE_PERSON_NAMES or not text:
        return text
    
    remover = get_name_remover()
    return remover.remove_all_names(text)


# ---------- 缓存管理类 ----------
class PPTXCache:
    """PPTX解析缓存管理器"""
    
    def __init__(self, cache_dir=CACHE_DIR, version=CACHE_VERSION):
        self.cache_dir = Path(cache_dir)
        self.version = version
        self.text_cache_file = self.cache_dir / f"text_cache_{version}.json"
        self.metadata_file = self.cache_dir / f"cache_metadata_{version}.json"
        
        # 创建缓存目录
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # 加载现有缓存
        self.text_cache = self._load_cache()
        self.metadata = self._load_metadata()
        self._last_was_cache_hit = False
        
        # 统计人名去除情况
        self.names_removed_count = 0
        self.name_removal_stats = {}
    
    def _get_file_hash(self, filepath):
        """计算文件的哈希值（用于检测文件是否变化）"""
        filepath = Path(filepath)
        if not filepath.exists():
            return None
        
        stat = filepath.stat()
        return f"{stat.st_mtime}_{stat.st_size}"
    
    def _load_cache(self):
        """加载文本缓存"""
        if not ENABLE_CACHE or FORCE_REFRESH_CACHE:
            return {}
        
        if self.text_cache_file.exists():
            try:
                with open(self.text_cache_file, 'r', encoding='utf-8') as f:
                    cache = json.load(f)
                print(f"  加载文本缓存: {len(cache)} 个文件")
                return cache
            except Exception as e:
                print(f"  警告: 加载缓存失败 ({e})，将重新生成")
                return {}
        return {}
    
    def _load_metadata(self):
        """加载缓存元数据"""
        if not ENABLE_CACHE or FORCE_REFRESH_CACHE:
            return {}
        
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except:
                return {}
        return {}
    
    def _save_cache(self):
        """保存缓存到磁盘"""
        if not ENABLE_CACHE:
            return
        
        try:
            with open(self.text_cache_file, 'w', encoding='utf-8') as f:
                json.dump(self.text_cache, f, ensure_ascii=False, indent=2)
            with open(self.metadata_file, 'w', encoding='utf-8') as f:
                json.dump(self.metadata, f, ensure_ascii=False, indent=2)
            print(f"  缓存已保存: {len(self.text_cache)} 个文件")
            
            # 保存人名去除统计
            stats_file = self.cache_dir / f"name_removal_stats_{self.version}.json"
            with open(stats_file, 'w', encoding='utf-8') as f:
                json.dump({
                    'names_removed_count': self.names_removed_count,
                    'name_removal_stats': self.name_removal_stats,
                    'removal_enabled': REMOVE_PERSON_NAMES,
                    'cache_version': self.version
                }, f, ensure_ascii=False, indent=2)
                
        except Exception as e:
            print(f"  警告: 保存缓存失败 ({e})")
    
    def get(self, filepath):
        """获取缓存的文件内容"""
        if not ENABLE_CACHE:
            self._last_was_cache_hit = False
            return None
        
        filepath = str(filepath)
        current_hash = self._get_file_hash(filepath)
        
        if filepath in self.text_cache:
            cached_hash = self.metadata.get(filepath, {}).get('hash')
            if cached_hash == current_hash:
                self._last_was_cache_hit = True
                return self.text_cache[filepath]
        
        self._last_was_cache_hit = False
        return None
    
    def set(self, filepath, text_content, original_text_length=0):
        """设置缓存（在保存前应用人名去除）"""
        if not ENABLE_CACHE:
            return
        
        filepath = str(filepath)
        
        # 在缓存阶段应用人名去除
        if REMOVE_PERSON_NAMES and text_content:
            original_len = len(text_content)
            cleaned_content = remove_person_names_from_text(text_content)
            removed_count = original_len - len(cleaned_content)
            
            if removed_count > 0:
                self.names_removed_count += removed_count
                
                # 统计各文件的人名去除情况
                filename = Path(filepath).name
                self.name_removal_stats[filename] = {
                    'original_length': original_len,
                    'cleaned_length': len(cleaned_content),
                    'removed_chars': removed_count,
                    'removal_ratio': removed_count / original_len if original_len > 0 else 0
                }
            
            text_content = cleaned_content
        
        self.text_cache[filepath] = text_content
        self.metadata[filepath] = {
            'hash': self._get_file_hash(filepath),
            'cached_at': datetime.now().isoformat(),
            'file_size': Path(filepath).stat().st_size if Path(filepath).exists() else 0,
            'names_removed': REMOVE_PERSON_NAMES
        }
    
    def clear(self):
        """清空缓存"""
        self.text_cache = {}
        self.metadata = {}
        self.names_removed_count = 0
        self.name_removal_stats = {}
        if self.text_cache_file.exists():
            self.text_cache_file.unlink()
        if self.metadata_file.exists():
            self.metadata_file.unlink()
        print("  缓存已清空")
    
    def save(self):
        """保存缓存"""
        self._save_cache()
    
    def get_stats(self):
        """获取缓存统计信息"""
        return {
            'cached_files': len(self.text_cache),
            'cache_file_size': self.text_cache_file.stat().st_size if self.text_cache_file.exists() else 0,
            'metadata_file_size': self.metadata_file.stat().st_size if self.metadata_file.exists() else 0,
            'names_removed_total': self.names_removed_count,
            'files_with_names_removed': len(self.name_removal_stats)
        }
    
    def has_cache(self, filepath):
        """检查文件是否有有效缓存"""
        if not ENABLE_CACHE:
            return False
        
        filepath = str(filepath)
        current_hash = self._get_file_hash(filepath)
        
        if filepath in self.text_cache:
            cached_hash = self.metadata.get(filepath, {}).get('hash')
            return cached_hash == current_hash
        return False
    
    def print_name_removal_summary(self):
        """打印人名去除摘要"""
        if not REMOVE_PERSON_NAMES:
            print("\n人名去除: 已禁用")
            return
        
        print(f"\n人名去除统计:")
        print(f"  总共去除字符数: {self.names_removed_count}")
        print(f"  涉及文件数: {len(self.name_removal_stats)}")
        
        # 显示去除最多的5个文件
        if self.name_removal_stats:
            sorted_files = sorted(
                self.name_removal_stats.items(),
                key=lambda x: x[1]['removed_chars'],
                reverse=True
            )[:5]
            
            if sorted_files:
                print(f"  去除最多人名的文件:")
                for filename, stats in sorted_files:
                    print(f"    - {filename[:40]}: 去除 {stats['removed_chars']} 字符 ({stats['removal_ratio']*100:.1f}%)")


# 全局缓存实例
_ppt_cache = None

def get_cache():
    """获取全局缓存实例"""
    global _ppt_cache
    if _ppt_cache is None:
        _ppt_cache = PPTXCache()
    return _ppt_cache


# ---------- 文件名数据增强函数 ----------
def augment_filename(filename):
    """生成文件名的多种变体（减少变体数量）"""
    if not filename:
        return [filename]
    
    variants = [filename]
    
    # 变体1：移除所有数字
    variant1 = re.sub(r'\d+', '', filename)
    variant1 = re.sub(r'\s+', ' ', variant1).strip()
    if variant1 and variant1 != filename:
        variants.append(variant1)
    
    # 变体2：只保留中文（最重要的变体）
    variant2 = re.sub(r'[a-zA-Z0-9]', '', filename)
    variant2 = re.sub(r'\s+', ' ', variant2).strip()
    if variant2 and variant2 != filename and variant2 != variant1:
        variants.append(variant2)
    
    # 去重
    variants = list(dict.fromkeys(variants))
    return variants[:FILENAME_AUGMENTATION_COUNT]


def process_filename_text(filename_text):
    """处理文件名文本：分词、过滤停用词"""
    if not filename_text:
        return ""
    
    words = jieba.cut(filename_text)
    filtered = []
    for w in words:
        if not w or w.strip() == '':
            continue
        if w not in STOPWORDS:
            if len(w) > 1 or w in MEANINGFUL_SINGLE_CHARS:
                filtered.append(w)
    
    if not filtered:
        filtered = [w for w in words if w.strip()]
    
    return " ".join(filtered)


# ---------- PPTX 文本提取函数 ----------
def extract_text_from_pptx(pptx_path, force_parse=False):
    """从PPTX文件中提取所有文本内容（带缓存，并在缓存时去除人名）"""
    cache = get_cache()
    filepath_str = str(pptx_path)
    
    cached_text = cache.get(filepath_str)
    if cached_text is not None:
        return cached_text
    
    if not force_parse:
        return None
    
    text_parts = []
    pptx_path_obj = Path(pptx_path) if isinstance(pptx_path, str) else pptx_path
    file_name = pptx_path_obj.name
    
    try:
        prs = Presentation(pptx_path)
        
        for slide in prs.slides:
            for shape in slide.shapes:
                try:
                    if hasattr(shape, "text") and shape.text:
                        text_parts.append(shape.text.strip())
                    
                    if hasattr(shape, "table"):
                        for row in shape.table.rows:
                            for cell in row.cells:
                                if cell.text:
                                    text_parts.append(cell.text.strip())
                    
                    if hasattr(shape, "text_frame") and shape.text_frame:
                        for paragraph in shape.text_frame.paragraphs:
                            if paragraph.text:
                                text_parts.append(paragraph.text.strip())
                except:
                    continue
            
            try:
                if slide.has_notes_slide:
                    notes_slide = slide.notes_slide
                    if notes_slide.notes_text_frame and notes_slide.notes_text_frame.text:
                        text_parts.append(notes_slide.notes_text_frame.text.strip())
            except:
                pass
            
            try:
                if slide.shapes.title and slide.shapes.title.text:
                    text_parts.append(slide.shapes.title.text.strip())
            except:
                pass
        
        result = " ".join(text_parts)
        
        # 在保存到缓存时应用人名去除（在set方法内部进行）
        cache.set(filepath_str, result)
        return result
        
    except Exception as e:
        print(f"    读取失败 {file_name}: {str(e)[:100]}")
        cache.set(filepath_str, "")
        return ""


def extract_filename_features(filepath):
    """从文件路径中提取文件名特征"""
    filepath_obj = Path(filepath) if isinstance(filepath, str) else filepath
    filename = filepath_obj.stem
    
    cleaned = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9]', ' ', filename)
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    
    return cleaned, filename


def clean_text(text):
    """基础清洗：保留中文、英文、数字、常用标点"""
    if not text:
        return ""
    
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    text = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9，。！？；：""''（）【】《》、 ]', '', text)
    text = re.sub(r'\s+', ' ', text)
    
    return text.strip()


def cut_words(text):
    """使用jieba进行分词，过滤停用词"""
    if not text:
        return ""
    
    words = jieba.cut(text)
    filtered = []
    for w in words:
        if not w or w.strip() == '':
            continue
        if w not in STOPWORDS:
            filtered.append(w)
    
    if not filtered:
        filtered = [w for w in words if w.strip()]
    
    if not filtered:
        return "空文本"
    
    return " ".join(filtered)


def process_ppt_file(pptx_file, cat, label_idx, stats, skip_parsing=False):
    """处理单个PPT文件"""
    filename_raw, original_filename = extract_filename_features(str(pptx_file))
    filename_processed = process_filename_text(filename_raw)
    
    if skip_parsing:
        cache = get_cache()
        cached_text = cache.get(str(pptx_file))
        if cached_text is None:
            stats['cache_missing_skip'] += 1
            return None, None, None, False
        raw_text = cached_text
    else:
        raw_text = extract_text_from_pptx(str(pptx_file), force_parse=True)
    
    # 注意：raw_text 已经从缓存中获取，且已经在缓存阶段去除了人名
    # 所以这里不需要再次去除
    
    if raw_text:
        cleaned = clean_text(raw_text)
        if cleaned:
            segmented = cut_words(cleaned)
            if segmented:
                stats['text_success'] += 1
                text_content = segmented
                if len(cleaned) < 20:
                    stats['text_short'] += 1
            else:
                text_content = "空文本"
        else:
            text_content = "空文本"
    else:
        text_content = "空文本"
        stats['text_empty'] += 1
    
    return text_content, filename_processed, original_filename, True


def load_data_with_cache(data_root):
    """带缓存的加载函数"""
    texts = []
    filenames = []
    labels = []
    
    stats = {
        'total_files': 0,
        'extract_failed': 0,
        'text_empty': 0,
        'text_short': 0,
        'text_success': 0,
        'filename_success': 0,
        'augmented_count': 0,
        'success': 0,
        'cache_hits': 0,
        'cache_misses': 0,
        'cache_missing_skip': 0,
        'skipped_categories': 0,
        'by_category': {}
    }
    
    cache = get_cache()
    skip_categories_set = set(SKIP_CATEGORIES)
    
    if skip_categories_set:
        print(f"\n🎯 跳过解析的科目（仅使用缓存）: {', '.join(skip_categories_set)}")
        print(f"   ⚠️  注意：这些科目的文件如果不在缓存中将被跳过")
    
    if REMOVE_PERSON_NAMES:
        print(f"\n👤 人名去除: 已启用（在缓存阶段使用jieba词性标注去除）")
    else:
        print(f"\n👤 人名去除: 已禁用")
    
    for label_idx, cat in enumerate(CATEGORIES):
        cat_path = os.path.join(data_root, cat)
        if not os.path.isdir(cat_path):
            print(f"警告: 目录不存在 {cat_path}")
            stats['by_category'][cat] = {'total': 0, 'success': 0, 'augmented': 0, 'skipped': 0, 'cache_missing': 0}
            continue
        
        pptx_files = list(Path(cat_path).glob("*.pptx")) + list(Path(cat_path).glob("*.PPTX"))
        
        skip_parsing = cat in skip_categories_set
        if skip_parsing:
            stats['skipped_categories'] += 1
        
        stats['by_category'][cat] = {
            'total': len(pptx_files), 
            'success': 0, 
            'augmented': 0,
            'skipped': 0 if not skip_parsing else len(pptx_files),
            'cache_missing': 0
        }
        stats['total_files'] += len(pptx_files)
        
        if not pptx_files:
            print(f"警告: {cat} 目录下没有找到PPTX文件")
            continue
        
        skip_msg = " [仅缓存模式]" if skip_parsing else ""
        print(f"\n处理 {cat} 目录{skip_msg}，共 {len(pptx_files)} 个文件")
        
        for idx, pptx_file in enumerate(pptx_files):
            text_content, filename_processed, original_filename, success = process_ppt_file(
                pptx_file, cat, label_idx, stats, skip_parsing=skip_parsing
            )
            
            if not success:
                stats['by_category'][cat]['cache_missing'] += 1
                if (idx + 1) % 20 == 0:
                    print(f"  ... 已处理 {idx + 1}/{len(pptx_files)} 个文件")
                continue
            
            if hasattr(cache, '_last_was_cache_hit'):
                if cache._last_was_cache_hit:
                    stats['cache_hits'] += 1
                else:
                    stats['cache_misses'] += 1
            
            filename_variants = [filename_processed]
            if ENABLE_FILENAME_AUGMENTATION and filename_processed:
                augmented_variants = augment_filename(original_filename)
                for variant in augmented_variants:
                    if variant and variant != original_filename:
                        variant_processed = process_filename_text(variant)
                        if variant_processed and variant_processed != filename_processed:
                            filename_variants.append(variant_processed)
                
                filename_variants = list(dict.fromkeys(filename_variants))[:FILENAME_AUGMENTATION_COUNT]
            
            for fname_var in filename_variants:
                if fname_var:
                    texts.append(text_content)
                    filenames.append(fname_var)
                    labels.append(label_idx)
                    stats['success'] += 1
                    stats['by_category'][cat]['success'] += 1
                    
                    if fname_var != filename_processed:
                        stats['augmented_count'] += 1
                        stats['by_category'][cat]['augmented'] += 1
            
            if (idx + 1) % 20 == 0:
                print(f"  ... 已处理 {idx + 1}/{len(pptx_files)} 个文件")
        
        cat_stats = stats['by_category'][cat]
        if skip_parsing:
            print(f"  {cat}: 原始{cat_stats['total']}个文件 → "
                  f"成功{cat_stats['success']}个样本（含{cat_stats['augmented']}个增强），"
                  f"缓存缺失{cat_stats['cache_missing']}个文件")
        else:
            print(f"  {cat}: 原始{cat_stats['total']}个文件 → {cat_stats['success']}个样本（含{cat_stats['augmented']}个增强）")
    
    return texts, filenames, labels, stats


def print_stats(stats):
    """打印详细统计信息"""
    print("\n" + "="*50)
    print("数据加载统计报告（带缓存）")
    print("="*50)
    print(f"总文件数:        {stats['total_files']}")
    print(f"生成样本数:      {stats['success']}")
    print(f"增强样本数:      {stats['augmented_count']}")
    if stats['success'] > 0:
        print(f"  └─ 增强比例:   {stats['augmented_count']/stats['success']*100:.1f}%")
    
    if stats.get('cache_missing_skip', 0) > 0:
        print(f"\n⚠️  跳过解析模式中因缓存缺失跳过的文件: {stats['cache_missing_skip']}")
    
    print(f"\n文本状态:")
    print(f"  有文本内容:    {stats['text_success']}")
    print(f"  文本过短:      {stats['text_short']}")
    print(f"  文本为空:      {stats['text_empty']}")
    
    cache = get_cache()
    cache_stats = cache.get_stats()
    print(f"\n缓存统计:")
    print(f"  缓存文件数:    {cache_stats['cached_files']}")
    print(f"  缓存命中:      {stats.get('cache_hits', 0)}")
    print(f"  缓存未命中:    {stats.get('cache_misses', 0)}")
    
    # 打印人名去除统计
    if REMOVE_PERSON_NAMES:
        print(f"\n👤 人名去除统计:")
        print(f"  总共去除字符:  {cache_stats['names_removed_total']}")
        print(f"  涉及文件数:    {cache_stats['files_with_names_removed']}")
    
    print("\n各类别统计:")
    for cat, info in stats['by_category'].items():
        if info['total'] > 0:
            if info.get('cache_missing', 0) > 0:
                print(f"  {cat}: {info['total']}个文件 → {info['success']}个样本（{info['augmented']}增强，{info['cache_missing']}缓存缺失）")
            else:
                print(f"  {cat}: {info['total']}个文件 → {info['success']}个样本（{info['augmented']}增强）")
    print("="*50)


def build_optimized_multimodal_model(vocab_size, max_seq_len, max_filename_len, 
                                      filename_vocab_size, num_classes):
    """
    构建优化版的多模态模型
    特点：
    1. 增加kernel_size=2的卷积核（捕获更短模式）
    2. 降低融合层Dropout率（保留更多信息）
    3. 增加文本分支复杂度
    """
    
    # 文本分支 - 增强版
    text_input = Input(shape=(max_seq_len,), name='text_input')
    text_embedding = Embedding(vocab_size, EMBEDDING_DIM, name='text_embedding')(text_input)
    
    # 多尺度卷积（增加kernel_size=2）
    conv_2 = Conv1D(filters=128, kernel_size=2, activation='relu', padding='same', name='text_conv_2')(text_embedding)
    conv_3 = Conv1D(filters=128, kernel_size=3, activation='relu', padding='same', name='text_conv_3')(text_embedding)
    conv_4 = Conv1D(filters=128, kernel_size=4, activation='relu', padding='same', name='text_conv_4')(text_embedding)
    conv_5 = Conv1D(filters=128, kernel_size=5, activation='relu', padding='same', name='text_conv_5')(text_embedding)
    
    pool_2 = GlobalMaxPooling1D(name='text_pool_2')(conv_2)
    pool_3 = GlobalMaxPooling1D(name='text_pool_3')(conv_3)
    pool_4 = GlobalMaxPooling1D(name='text_pool_4')(conv_4)
    pool_5 = GlobalMaxPooling1D(name='text_pool_5')(conv_5)
    
    # 拼接所有尺度特征
    text_multi = Concatenate(name='text_multi')([pool_2, pool_3, pool_4, pool_5])
    text_dense = Dense(128, activation='relu', name='text_dense')(text_multi)
    text_dropout = Dropout(0.5, name='text_dropout')(text_dense)
    
    # 文件名分支
    filename_input = Input(shape=(max_filename_len,), name='filename_input')
    filename_embedding = Embedding(filename_vocab_size, FILENAME_EMBEDDING_DIM, name='filename_embedding')(filename_input)
    filename_conv = Conv1D(filters=64, kernel_size=2, activation='relu', name='filename_conv')(filename_embedding)
    filename_pool = GlobalMaxPooling1D(name='filename_pool')(filename_conv)
    filename_dense = Dense(32, activation='relu', name='filename_dense')(filename_pool)
    filename_dropout = Dropout(0.3, name='filename_dropout')(filename_dense)
    
    # 融合层 - 降低Dropout率
    combined = Concatenate(name='combined')([text_dropout, filename_dropout])
    final_dense = Dense(64, activation='relu', name='final_dense')(combined)
    final_dropout = Dropout(0.3, name='final_dropout')(final_dense)  # 0.5 → 0.3
    output = Dense(num_classes, activation='softmax', name='output')(final_dropout)
    
    model = Model(inputs=[text_input, filename_input], outputs=output)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    return model


def save_tokenizers_without_keras(text_tokenizer, filename_tokenizer, categories, config):
    """
    保存无Keras依赖的Tokenizer文件（用于推理）
    在20000词处截断（只保留前MAX_NB_WORDS个词）
    """
    print("\n" + "="*50)
    print("保存无依赖的Tokenizer文件（词表截断）")
    print("="*50)
    
    # 处理文本Tokenizer - 截断到MAX_NB_WORDS
    full_word_index = text_tokenizer.word_index
    full_vocab_size = len(full_word_index)
    
    # 截断：只保留前MAX_NB_WORDS个最常见的词
    # Tokenizer的word_index已经是按频率降序排列的
    truncated_word_index = {}
    truncated_index_word = {}
    
    for i, (word, idx) in enumerate(full_word_index.items()):
        if i < MAX_NB_WORDS:
            truncated_word_index[word] = idx
            truncated_index_word[idx] = word
    
    actual_vocab_size = len(truncated_word_index)
    
    print(f"  文本Tokenizer:")
    print(f"    - 原始词表大小: {full_vocab_size}")
    print(f"    - 截断后词表大小: {actual_vocab_size} (限制在 {MAX_NB_WORDS})")
    print(f"    - 截断比例: {(1 - actual_vocab_size/full_vocab_size)*100:.1f}%")
    
    text_tokenizer_data = {
        'word_index': truncated_word_index,
        'index_word': truncated_index_word,
        'word_counts': {word: text_tokenizer.word_counts.get(word, 0) 
                       for word in truncated_word_index.keys()},
        'document_count': text_tokenizer.document_count,
        'vocab_size': actual_vocab_size,
        'max_words': MAX_NB_WORDS,
        'is_truncated': True,
        'original_vocab_size': full_vocab_size
    }
    
    # 处理文件名Tokenizer - 截断到MAX_FILENAME_WORDS
    full_filename_word_index = filename_tokenizer.word_index
    full_filename_vocab_size = len(full_filename_word_index)
    
    truncated_filename_word_index = {}
    truncated_filename_index_word = {}
    
    for i, (word, idx) in enumerate(full_filename_word_index.items()):
        if i < MAX_FILENAME_WORDS:
            truncated_filename_word_index[word] = idx
            truncated_filename_index_word[idx] = word
    
    actual_filename_vocab_size = len(truncated_filename_word_index)
    
    print(f"\n  文件名Tokenizer:")
    print(f"    - 原始词表大小: {full_filename_vocab_size}")
    print(f"    - 截断后词表大小: {actual_filename_vocab_size} (限制在 {MAX_FILENAME_WORDS})")
    print(f"    - 截断比例: {(1 - actual_filename_vocab_size/full_filename_vocab_size)*100:.1f}%")
    
    filename_tokenizer_data = {
        'word_index': truncated_filename_word_index,
        'index_word': truncated_filename_index_word,
        'word_counts': {word: filename_tokenizer.word_counts.get(word, 0)
                       for word in truncated_filename_word_index.keys()},
        'document_count': filename_tokenizer.document_count,
        'vocab_size': actual_filename_vocab_size,
        'max_words': MAX_FILENAME_WORDS,
        'is_truncated': True,
        'original_vocab_size': full_filename_vocab_size
    }
    
    # 保存文件
    with open('text_tokenizer_none.pkl', 'wb') as f:
        pickle.dump(text_tokenizer_data, f)
    print(f"\n✅ 已保存: text_tokenizer_none.pkl")
    
    with open('filename_tokenizer_none.pkl', 'wb') as f:
        pickle.dump(filename_tokenizer_data, f)
    print(f"✅ 已保存: filename_tokenizer_none.pkl")
    
    # 保存类别和配置
    with open('categories.pkl', 'wb') as f:
        pickle.dump(categories, f)
    
    # 更新配置，记录截断信息
    config['max_nb_words'] = MAX_NB_WORDS
    config['max_filename_words'] = MAX_FILENAME_WORDS
    config['actual_text_vocab_size'] = actual_vocab_size
    config['actual_filename_vocab_size'] = actual_filename_vocab_size
    
    with open('config_optimized.pkl', 'wb') as f:
        pickle.dump(config, f)
    
    print(f"✅ 已保存: categories.pkl")
    print(f"✅ 已保存: config_optimized.pkl")
    
    # 打印汇总信息
    print("\n" + "="*50)
    print("📊 词表截断汇总:")
    print(f"  文本: {full_vocab_size} → {actual_vocab_size} (保留前{MAX_NB_WORDS}词)")
    print(f"  文件名: {full_filename_vocab_size} → {actual_filename_vocab_size} (保留前{MAX_FILENAME_WORDS}词)")
    print(f"  建议 embedding_dim: {min(EMBEDDING_DIM, actual_vocab_size, actual_filename_vocab_size)}")
    print("="*50)
    
    return text_tokenizer_data, filename_tokenizer_data


def save_json_vocabularies(text_tokenizer, filename_tokenizer):
    """
    保存JSON格式的词表文件（截断在MAX_NB_WORDS和MAX_FILENAME_WORDS）
    便于外部工具读取和使用
    """
    print("\n" + "="*50)
    print("保存JSON格式词表文件")
    print("="*50)
    
    # ========== 1. 保存文本词表 ==========
    full_word_index = text_tokenizer.word_index
    
    # 截断：只保留前MAX_NB_WORDS个最常见的词
    truncated_word_index = {}
    truncated_word_counts = {}
    
    for i, (word, idx) in enumerate(full_word_index.items()):
        if i < MAX_NB_WORDS:
            truncated_word_index[word] = idx
            truncated_word_counts[word] = text_tokenizer.word_counts.get(word, 0)
    
    # 构建文本词表JSON
    text_vocab_json = {
        "metadata": {
            "type": "text_vocabulary",
            "version": CACHE_VERSION,
            "max_words": MAX_NB_WORDS,
            "original_size": len(full_word_index),
            "actual_size": len(truncated_word_index),
            "is_truncated": True,
            "created_at": datetime.now().isoformat(),
            "description": "文本分类器的词表，按词频降序排列，仅保留前{}个最常见的词".format(MAX_NB_WORDS)
        },
        "vocabulary": [
            {
                "index": idx,
                "word": word,
                "frequency": truncated_word_counts.get(word, 0)
            }
            for word, idx in truncated_word_index.items()
        ],
        # 提供两种访问方式：按索引和按词
        "word_to_index": truncated_word_index,
        "index_to_word": {str(idx): word for word, idx in truncated_word_index.items()}
    }
    
    # 按索引排序（方便查看）
    text_vocab_json["vocabulary"].sort(key=lambda x: x["index"])
    
    # 保存文本词表
    with open('text_vocabulary.json', 'w', encoding='utf-8') as f:
        json.dump(text_vocab_json, f, ensure_ascii=False, indent=2)
    print(f"✅ 已保存: text_vocabulary.json")
    print(f"   - 词表大小: {len(truncated_word_index)} 词")
    print(f"   - 原始大小: {len(full_word_index)} 词")
    print(f"   - 文件大小: {os.path.getsize('text_vocabulary.json') / 1024:.1f} KB")
    
    # 同时保存一个更简洁的版本（仅词列表，按频率排序）
    simple_text_vocab = {
        "metadata": {
            "type": "text_vocabulary_simple",
            "max_words": MAX_NB_WORDS,
            "actual_size": len(truncated_word_index)
        },
        "words_by_frequency": list(truncated_word_index.keys()),
        "frequency_map": truncated_word_counts
    }
    
    with open('text_vocabulary_simple.json', 'w', encoding='utf-8') as f:
        json.dump(simple_text_vocab, f, ensure_ascii=False, indent=2)
    print(f"✅ 已保存: text_vocabulary_simple.json")
    
    # ========== 2. 保存文件名词表 ==========
    full_filename_word_index = filename_tokenizer.word_index
    
    # 截断
    truncated_filename_word_index = {}
    truncated_filename_word_counts = {}
    
    for i, (word, idx) in enumerate(full_filename_word_index.items()):
        if i < MAX_FILENAME_WORDS:
            truncated_filename_word_index[word] = idx
            truncated_filename_word_counts[word] = filename_tokenizer.word_counts.get(word, 0)
    
    # 构建文件名词表JSON
    filename_vocab_json = {
        "metadata": {
            "type": "filename_vocabulary",
            "version": CACHE_VERSION,
            "max_words": MAX_FILENAME_WORDS,
            "original_size": len(full_filename_word_index),
            "actual_size": len(truncated_filename_word_index),
            "is_truncated": True,
            "created_at": datetime.now().isoformat(),
            "description": "文件名词表，按词频降序排列，仅保留前{}个最常见的词".format(MAX_FILENAME_WORDS)
        },
        "vocabulary": [
            {
                "index": idx,
                "word": word,
                "frequency": truncated_filename_word_counts.get(word, 0)
            }
            for word, idx in truncated_filename_word_index.items()
        ],
        "word_to_index": truncated_filename_word_index,
        "index_to_word": {str(idx): word for word, idx in truncated_filename_word_index.items()}
    }
    
    # 按索引排序
    filename_vocab_json["vocabulary"].sort(key=lambda x: x["index"])
    
    # 保存文件名词表
    with open('filename_vocabulary.json', 'w', encoding='utf-8') as f:
        json.dump(filename_vocab_json, f, ensure_ascii=False, indent=2)
    print(f"\n✅ 已保存: filename_vocabulary.json")
    print(f"   - 词表大小: {len(truncated_filename_word_index)} 词")
    print(f"   - 原始大小: {len(full_filename_word_index)} 词")
    print(f"   - 文件大小: {os.path.getsize('filename_vocabulary.json') / 1024:.1f} KB")
    
    # 简洁版本
    simple_filename_vocab = {
        "metadata": {
            "type": "filename_vocabulary_simple",
            "max_words": MAX_FILENAME_WORDS,
            "actual_size": len(truncated_filename_word_index)
        },
        "words_by_frequency": list(truncated_filename_word_index.keys()),
        "frequency_map": truncated_filename_word_counts
    }
    
    with open('filename_vocabulary_simple.json', 'w', encoding='utf-8') as f:
        json.dump(simple_filename_vocab, f, ensure_ascii=False, indent=2)
    print(f"✅ 已保存: filename_vocabulary_simple.json")
    
    # ========== 3. 保存词频统计报告 ==========
    # 生成词频统计报告
    word_frequencies = [(word, count) for word, count in truncated_word_counts.items()]
    word_frequencies.sort(key=lambda x: x[1], reverse=True)
    
    # 前100个高频词
    top_100_words = [
        {"rank": i+1, "word": word, "frequency": freq}
        for i, (word, freq) in enumerate(word_frequencies[:100])
    ]
    
    # 词频分布统计
    freq_buckets = {
        "1-10": 0,
        "11-50": 0,
        "51-100": 0,
        "101-500": 0,
        "501-1000": 0,
        "1001+": 0
    }
    
    for _, freq in word_frequencies:
        if freq <= 10:
            freq_buckets["1-10"] += 1
        elif freq <= 50:
            freq_buckets["11-50"] += 1
        elif freq <= 100:
            freq_buckets["51-100"] += 1
        elif freq <= 500:
            freq_buckets["101-500"] += 1
        elif freq <= 1000:
            freq_buckets["501-1000"] += 1
        else:
            freq_buckets["1001+"] += 1
    
    word_frequency_report = {
        "metadata": {
            "type": "word_frequency_report",
            "created_at": datetime.now().isoformat(),
            "vocabulary_size": len(truncated_word_index),
            "total_word_count": sum(truncated_word_counts.values())
        },
        "top_100_words": top_100_words,
        "frequency_distribution": freq_buckets,
        "statistics": {
            "min_frequency": min(truncated_word_counts.values()) if truncated_word_counts else 0,
            "max_frequency": max(truncated_word_counts.values()) if truncated_word_counts else 0,
            "avg_frequency": sum(truncated_word_counts.values()) / len(truncated_word_counts) if truncated_word_counts else 0,
            "median_frequency": sorted(truncated_word_counts.values())[len(truncated_word_counts)//2] if truncated_word_counts else 0
        }
    }
    
    with open('word_frequency_report.json', 'w', encoding='utf-8') as f:
        json.dump(word_frequency_report, f, ensure_ascii=False, indent=2)
    print(f"\n✅ 已保存: word_frequency_report.json")
    
    # ========== 4. 保存类别映射 ==========
    category_mapping = {
        "metadata": {
            "type": "category_mapping",
            "created_at": datetime.now().isoformat(),
            "num_categories": len(CATEGORIES)
        },
        "categories": CATEGORIES,
        "index_to_category": {str(i): cat for i, cat in enumerate(CATEGORIES)},
        "category_to_index": {cat: i for i, cat in enumerate(CATEGORIES)}
    }
    
    with open('category_mapping.json', 'w', encoding='utf-8') as f:
        json.dump(category_mapping, f, ensure_ascii=False, indent=2)
    print(f"✅ 已保存: category_mapping.json")
    
    # 打印汇总
    print("\n" + "="*50)
    print("📊 JSON词表文件汇总:")
    print(f"  文本词表 (完整): text_vocabulary.json ({len(truncated_word_index)} 词)")
    print(f"  文本词表 (简洁): text_vocabulary_simple.json")
    print(f"  文件名词表 (完整): filename_vocabulary.json ({len(truncated_filename_word_index)} 词)")
    print(f"  文件名词表 (简洁): filename_vocabulary_simple.json")
    print(f"  词频统计报告: word_frequency_report.json")
    print(f"  类别映射: category_mapping.json")
    print("="*50)
    
    return True


# ---------- 测试人名去除功能 ----------
def test_name_removal():
    """测试人名去除功能"""
    print("\n" + "="*50)
    print("测试人名去除功能")
    print("="*50)
    
    test_texts = [
        "今天李明同学在课堂上回答了问题。",
        "王老师说这道题需要认真思考。",
        "张三、李四和王五一起参加了数学竞赛。",
        "习近平主席发表了重要讲话。",
        "鲁迅的作品《呐喊》很有名。",
        "张华、李萍和王芳都是优秀的学生代表。",
        "陈景润在数学领域有重要贡献。",
        "钱学森是中国航天事业的奠基人。",
    ]
    
    remover = get_name_remover()
    
    for text in test_texts:
        cleaned = remover.remove_all_names(text)
        print(f"\n原文: {text}")
        print(f"清理后: {cleaned}")
        print(f"去除字符数: {len(text) - len(cleaned)}")
    
    print("\n" + "="*50)


# ---------- 主流程 ----------
def main():
    print("="*50)
    print("PPTX学科分类器训练（优化版）")
    print(f"支持类别: {', '.join(CATEGORIES)}")
    print(f"文本权重: {TEXT_WEIGHT}, 文件名权重: {FILENAME_WEIGHT}")
    print(f"数据增强: {'启用' if ENABLE_FILENAME_AUGMENTATION else '禁用'}")
    print(f"缓存: {'启用' if ENABLE_CACHE else '禁用'} (版本: {CACHE_VERSION})")
    print(f"类别权重: {'启用' if USE_CLASS_WEIGHTS else '禁用'}")
    print(f"人名去除: {'启用 (缓存阶段)' if REMOVE_PERSON_NAMES else '禁用'}")
    print(f"词表截断: 限制在 {MAX_NB_WORDS} 词")
    
    if SKIP_CATEGORIES:
        print(f"🎯 仅缓存模式科目: {', '.join(SKIP_CATEGORIES)}")
    
    print("="*50)
    
    # 可选：运行测试
    # test_name_removal()
    
    if not os.path.exists(DATA_ROOT):
        print(f"错误: 数据目录不存在 '{DATA_ROOT}'")
        return
    
    if FORCE_REFRESH_CACHE:
        print("\n强制刷新缓存模式...")
        cache = get_cache()
        cache.clear()
    
    # 1. 加载数据
    print("\n步骤1: 加载PPTX文件（使用缓存加速，缓存阶段自动去除人名）...")
    import time
    start_time = time.time()
    texts, filenames, labels, stats = load_data_with_cache(DATA_ROOT)
    load_time = time.time() - start_time
    print(f"数据加载耗时: {load_time:.2f} 秒")
    
    print_stats(stats)
    
    # 打印人名去除详情
    cache = get_cache()
    cache.print_name_removal_summary()
    
    cache.save()
    
    if len(texts) == 0:
        print("\n错误: 未找到任何有效PPTX文件！")
        return
    
    # 2. 划分数据集
    print("\n步骤2: 划分数据集...")
    X_train_text, X_test_text, X_train_filename, X_test_filename, y_train, y_test = train_test_split(
        texts, filenames, labels, test_size=TEST_SPLIT, stratify=labels, random_state=42
    )
    
    val_ratio = VALIDATION_SPLIT / (1 - TEST_SPLIT)
    X_train_text, X_val_text, X_train_filename, X_val_filename, y_train, y_val = train_test_split(
        X_train_text, X_train_filename, y_train, test_size=val_ratio, 
        stratify=y_train, random_state=42
    )
    print(f"训练集: {len(X_train_text)}")
    print(f"验证集: {len(X_val_text)}")
    print(f"测试集: {len(X_test_text)}")
    
    # 3. 构建Tokenizer（使用截断）
    print("\n步骤3: 构建词表并转换序列...")
    print(f"  词表截断: 最多保留 {MAX_NB_WORDS} 个最常见的词")
    
    # 文本Tokenizer - 使用截断
    text_tokenizer = Tokenizer(num_words=MAX_NB_WORDS, oov_token='<UNK>')
    text_tokenizer.fit_on_texts(X_train_text)
    
    # 实际词表大小（由于截断，不会超过MAX_NB_WORDS）
    text_vocab_size = min(MAX_NB_WORDS, len(text_tokenizer.word_index) + 1)
    
    print(f"  文本原始词汇量: {len(text_tokenizer.word_index)}")
    print(f"  文本实际使用词汇量: {text_vocab_size} (截断在 {MAX_NB_WORDS})")
    
    # 文件名Tokenizer - 使用截断
    filename_tokenizer = Tokenizer(num_words=MAX_FILENAME_WORDS, oov_token='<UNK>')
    filename_tokenizer.fit_on_texts(X_train_filename)
    
    filename_vocab_size = min(MAX_FILENAME_WORDS, len(filename_tokenizer.word_index) + 1)
    
    print(f"  文件名字典大小: {len(filename_tokenizer.word_index)}")
    print(f"  文件名实际使用词汇量: {filename_vocab_size} (截断在 {MAX_FILENAME_WORDS})")
    
    def seq_and_pad_text(texts):
        seqs = text_tokenizer.texts_to_sequences(texts)
        return pad_sequences(seqs, maxlen=MAX_SEQUENCE_LENGTH, padding='post', truncating='post')
    
    def seq_and_pad_filename(filenames):
        seqs = filename_tokenizer.texts_to_sequences(filenames)
        return pad_sequences(seqs, maxlen=MAX_FILENAME_LENGTH, padding='post', truncating='post')
    
    X_train_text_pad = seq_and_pad_text(X_train_text)
    X_val_text_pad = seq_and_pad_text(X_val_text)
    X_test_text_pad = seq_and_pad_text(X_test_text)
    
    X_train_filename_pad = seq_and_pad_filename(X_train_filename)
    X_val_filename_pad = seq_and_pad_filename(X_val_filename)
    X_test_filename_pad = seq_and_pad_filename(X_test_filename)
    
    y_train_cat = to_categorical(y_train, num_classes=len(CATEGORIES))
    y_val_cat = to_categorical(y_val, num_classes=len(CATEGORIES))
    y_test_cat = to_categorical(y_test, num_classes=len(CATEGORIES))
    
    print(f"文本输入形状: {X_train_text_pad.shape}")
    print(f"文件名输入形状: {X_train_filename_pad.shape}")
    
    # 4. 构建模型
    print("\n步骤4: 构建优化版多模态模型...")
    model = build_optimized_multimodal_model(
        vocab_size=text_vocab_size,
        max_seq_len=MAX_SEQUENCE_LENGTH,
        max_filename_len=MAX_FILENAME_LENGTH,
        filename_vocab_size=filename_vocab_size,
        num_classes=len(CATEGORIES)
    )
    model.summary()
    
    # 5. 计算类别权重（可选）
    class_weights = None
    if USE_CLASS_WEIGHTS:
        print("\n计算类别权重...")
        class_weights = compute_class_weight(
            'balanced',
            classes=np.unique(y_train),
            y=y_train
        )
        class_weights = dict(enumerate(class_weights))
        for i, cat in enumerate(CATEGORIES):
            print(f"  {cat}: {class_weights[i]:.4f}")
    
    # 6. 训练
    print("\n步骤5: 开始训练...")
    early_stop = EarlyStopping(
        monitor='val_loss', 
        patience=6,                    # 增加耐心
        restore_best_weights=True,
        min_delta=0.0001               # 最小变化阈值
    )
    checkpoint = ModelCheckpoint(
        'best_model_optimized.keras', 
        monitor='val_accuracy', 
        save_best_only=True, 
        mode='max'
    )
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss', 
        factor=0.5, 
        patience=3, 
        min_lr=1e-6,
        cooldown=1                     # 冷却期
    )
    
    history = model.fit(
        [X_train_text_pad, X_train_filename_pad], y_train_cat,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=([X_val_text_pad, X_val_filename_pad], y_val_cat),
        callbacks=[early_stop, checkpoint, reduce_lr],
        class_weight=class_weights,
        verbose=1
    )
    
    # 7. 评估
    print("\n步骤6: 模型评估...")
    loss, acc = model.evaluate([X_test_text_pad, X_test_filename_pad], y_test_cat, verbose=0)
    print(f"测试集准确率: {acc:.4f}")
    
    y_pred_proba = model.predict([X_test_text_pad, X_test_filename_pad])
    y_pred = np.argmax(y_pred_proba, axis=1)
    print("\n详细分类报告:")
    print(classification_report(y_test, y_pred, target_names=CATEGORIES))
    print("混淆矩阵:")
    print(confusion_matrix(y_test, y_pred))
    
    # 8. 保存模型和Tokenizer
    print("\n步骤7: 保存模型...")
    model.save('textcnn_optimized_classifier.keras')
    
    # 保存标准Tokenizer（用于可能的重训练）
    with open('text_tokenizer.pkl', 'wb') as f:
        pickle.dump(text_tokenizer, f)
    with open('filename_tokenizer.pkl', 'wb') as f:
        pickle.dump(filename_tokenizer, f)
    
    # 配置信息
    config = {
        'max_sequence_length': MAX_SEQUENCE_LENGTH,
        'max_filename_length': MAX_FILENAME_LENGTH,
        'categories': CATEGORIES,
        'text_vocab_size': text_vocab_size,
        'filename_vocab_size': filename_vocab_size,
        'embedding_dim': EMBEDDING_DIM,
        'filename_embedding_dim': FILENAME_EMBEDDING_DIM,
        'text_weight': TEXT_WEIGHT,
        'filename_weight': FILENAME_WEIGHT,
        'cache_version': CACHE_VERSION,
        'skip_categories': SKIP_CATEGORIES,
        'use_class_weights': USE_CLASS_WEIGHTS,
        'remove_person_names': REMOVE_PERSON_NAMES,
        'max_nb_words': MAX_NB_WORDS,
        'max_filename_words': MAX_FILENAME_WORDS
    }
    
    # 保存无Keras依赖的Tokenizer（截断版本，用于推理）
    save_tokenizers_without_keras(text_tokenizer, filename_tokenizer, CATEGORIES, config)
    
    # 🆕 保存JSON格式的词表文件
    save_json_vocabularies(text_tokenizer, filename_tokenizer)
    
    print("\n" + "="*50)
    print("训练完成！已保存以下文件:")
    print("  - textcnn_optimized_classifier.keras")
    print("  - best_model_optimized.keras")
    print("  - text_tokenizer.pkl (完整版)")
    print("  - filename_tokenizer.pkl (完整版)")
    print("  - text_tokenizer_none.pkl (无依赖版，截断)")
    print("  - filename_tokenizer_none.pkl (无依赖版，截断)")
    print("  - categories.pkl")
    print("  - config_optimized.pkl")
    print("\n📝 JSON词表文件:")
    print("  - text_vocabulary.json (完整文本词表)")
    print("  - text_vocabulary_simple.json (简洁文本词表)")
    print("  - filename_vocabulary.json (完整文件名词表)")
    print("  - filename_vocabulary_simple.json (简洁文件名词表)")
    print("  - word_frequency_report.json (词频统计报告)")
    print("  - category_mapping.json (类别映射)")
    print(f"\n缓存目录: {CACHE_DIR}/")
    print(f"  - text_cache_{CACHE_VERSION}.json")
    print(f"  - cache_metadata_{CACHE_VERSION}.json")
    if REMOVE_PERSON_NAMES:
        print(f"  - name_removal_stats_{CACHE_VERSION}.json")
    print("="*50)
    
    # 可选：保存训练历史
    try:
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        
        axes[0].plot(history.history['loss'], label='train')
        axes[0].plot(history.history['val_loss'], label='val')
        axes[0].set_title('Model Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].legend()
        
        axes[1].plot(history.history['accuracy'], label='train')
        axes[1].plot(history.history['val_accuracy'], label='val')
        axes[1].set_title('Model Accuracy')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy')
        axes[1].legend()
        
        plt.tight_layout()
        plt.savefig('training_history.png')
        print("训练曲线已保存: training_history.png")
        plt.close()
    except:
        pass


if __name__ == "__main__":
    random.seed(42)
    np.random.seed(42)
    main()
