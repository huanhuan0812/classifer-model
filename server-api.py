#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PPTX学科分类器 API服务
提供RESTful API接口，返回JSON格式预测结果
支持：单文件预测、批量预测、健康检查、关键词提取
"""

import os
import sys
import gc
import warnings
import re
import pickle
import json
import tempfile
from pathlib import Path
from datetime import datetime
from collections import Counter
from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import logging

# 屏蔽所有警告
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# 导入tensorflow
import tensorflow as tf
tf.get_logger().setLevel('ERROR')
tf.autograph.set_verbosity(0)

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import jieba
import jieba.analyse
from pptx import Presentation

# 创建Flask应用
app = Flask(__name__)
CORS(app)  # 允许跨域请求

# ---------- 配置 ----------
MODEL_PATH = "textcnn_balanced_classifier.keras"
TEXT_TOKENIZER_PATH = "text_tokenizer.pkl"
FILENAME_TOKENIZER_PATH = "filename_tokenizer.pkl"
CATEGORIES_PATH = "categories.pkl"
CONFIG_PATH = "config_balanced.pkl"

# 关键词提取配置
KEYWORD_TOP_N = 12           # 提取最多12个关键词
KEYWORD_MIN_N = 5            # 至少提取5个关键词
KEYWORD_WEIGHTED = True      # 使用TF-IDF加权

# 停用词表
STOPWORDS = set([
    '的', '了', '是', '我', '你', '他', '她', '它', '我们', '你们', '他们',
    '这', '那', '有', '在', '不', '和', '与', '就', '都', '而', '及', '或',
    '一个', '这个', '那个', '那些', '这些', '这里', '那里', '然后', '因为',
    '所以', '但是', '如果', '虽然', '然而', '并且', '或者', '可以', '可能',
    '应该', '需要', '没有', '自己', '什么', '哪个', '如何', '为什么', '也',
    '还', '被', '把', '给', '让', '去', '年', '月', '日', '时', '分', '秒',
    '第', '一', '二', '三', '四', '五', '六', '七', '八', '九', '十',
    '上', '下', '中', '大', '小', '多', '少', '高', '低', '长', '短'
])

# 学科相关的高权重词（用于关键词加权）
SUBJECT_KEYWORDS = {
    '语文': ['古诗', '文言文', '散文', '小说', '诗歌', '作者', '阅读', '写作', '作文', '修辞', '语法', '汉字', '成语', '唐诗', '宋词'],
    '数学': ['函数', '方程', '几何', '代数', '三角', '数列', '概率', '统计', '向量', '导数', '积分', '公式', '定理', '证明', '计算'],
    '英语': ['单词', '语法', '句型', '阅读', '听力', '写作', '翻译', '时态', '从句', '词汇', '发音', '对话', '短文', '字母'],
    '物理': ['力', '运动', '能量', '电场', '磁场', '电路', '光学', '热学', '量子', '相对论', '速度', '加速度', '质量', '密度', '压强'],
    '化学': ['反应', '元素', '化合物', '方程式', '原子', '分子', '离子', '酸碱', '氧化', '还原', '实验', '试剂', '催化剂', '溶液'],
    '生物': ['细胞', '基因', '生物', '植物', '动物', '生态', '进化', '遗传', '细菌', '病毒', '蛋白质', '酶', '光合作用', '呼吸作用'],
    '班会': ['主题', '活动', '班级', '同学', '老师', '纪律', '安全', '卫生', '德育', '心理健康', '团队', '合作', '文明', '礼仪']
}

# 抑制jieba输出
jieba.setLogLevel(logging.ERROR)
logging.getLogger().setLevel(logging.ERROR)

# ---------- 全局变量 ----------
_model = None
_text_tokenizer = None
_filename_tokenizer = None
_categories = None
_config = None

# 预测计数器（用于内存管理）
_prediction_count = 0


# ---------- 关键词提取函数 ----------
def extract_keywords(text, top_n=KEYWORD_TOP_N, min_n=KEYWORD_MIN_N):
    """
    从文本中提取关键词
    使用多种方法：TF-IDF、TextRank、词频统计
    返回：关键词列表，按重要性排序
    """
    if not text or len(text.strip()) < 10:
        return ["无足够文本内容"]
    
    keywords_set = set()
    
    # 方法1：TF-IDF 提取
    try:
        tfidf_keywords = jieba.analyse.extract_tags(
            text, topK=top_n, withWeight=False, allowPOS=()
        )
        keywords_set.update(tfidf_keywords[:top_n//2])
    except:
        pass
    
    # 方法2：TextRank 提取
    try:
        textrank_keywords = jieba.analyse.textrank(
            text, topK=top_n, withWeight=False, allowPOS=('ns', 'n', 'vn', 'v', 'an')
        )
        keywords_set.update(textrank_keywords[:top_n//2])
    except:
        pass
    
    # 方法3：词频统计（过滤停用词）
    words = jieba.cut(text, HMM=False)
    word_freq = Counter()
    for w in words:
        w = w.strip()
        if len(w) >= 2 and w not in STOPWORDS and not w.isdigit():
            word_freq[w] += 1
    
    # 取频率最高的词
    freq_keywords = [w for w, _ in word_freq.most_common(top_n)]
    keywords_set.update(freq_keywords[:top_n//2])
    
    # 方法4：学科相关关键词匹配（提高相关性）
    for category, subject_words in SUBJECT_KEYWORDS.items():
        for sw in subject_words:
            if sw in text and len(sw) >= 2:
                keywords_set.add(sw)
    
    # 转换为列表并排序
    keywords = list(keywords_set)
    
    # 按在文本中的位置和长度重新排序（更重要的词排在前面）
    def keyword_score(kw):
        score = 0
        # 长度加分（2-4字最佳）
        if 2 <= len(kw) <= 4:
            score += 10
        # 在文本中出现次数
        score += text.count(kw) * 5
        # 学科关键词加分
        for subject_words in SUBJECT_KEYWORDS.values():
            if kw in subject_words:
                score += 20
                break
        return score
    
    keywords.sort(key=keyword_score, reverse=True)
    
    # 确保返回 min_n 到 top_n 个关键词
    if len(keywords) < min_n:
        # 如果关键词不足，添加原始文本中的常见词
        common_words = [w for w, _ in word_freq.most_common(top_n * 2) 
                       if w not in keywords and len(w) >= 2]
        keywords.extend(common_words[:min_n - len(keywords)])
    
    return keywords[:top_n]


def extract_filename_keywords(filename_text, top_n=5):
    """
    从文件名中提取关键词
    """
    if not filename_text:
        return []
    
    # 文件名通常较短，直接分词后过滤
    words = jieba.cut(filename_text, HMM=False)
    keywords = []
    for w in words:
        w = w.strip()
        if len(w) >= 2 and w not in STOPWORDS and not w.isdigit():
            keywords.append(w)
    
    # 去重并限制数量
    seen = set()
    unique_keywords = []
    for kw in keywords:
        if kw not in seen:
            seen.add(kw)
            unique_keywords.append(kw)
    
    return unique_keywords[:top_n]


# ---------- 模型加载 ----------
def load_models():
    """加载模型和分词器"""
    global _model, _text_tokenizer, _filename_tokenizer, _categories, _config
    
    print("正在加载模型...", end=' ', flush=True)
    _model = load_model(MODEL_PATH, compile=False)
    print("✓", end=' ')
    
    print("加载文本分词器...", end=' ', flush=True)
    with open(TEXT_TOKENIZER_PATH, 'rb') as f:
        _text_tokenizer = pickle.load(f)
    print("✓", end=' ')
    
    print("加载文件名词典...", end=' ', flush=True)
    with open(FILENAME_TOKENIZER_PATH, 'rb') as f:
        _filename_tokenizer = pickle.load(f)
    print("✓", end=' ')
    
    print("加载类别映射...", end=' ', flush=True)
    with open(CATEGORIES_PATH, 'rb') as f:
        _categories = pickle.load(f)
    print("✓", end=' ')
    
    print("加载配置...", end=' ', flush=True)
    with open(CONFIG_PATH, 'rb') as f:
        _config = pickle.load(f)
    print("✓")
    
    print(f"\n模型已就绪！支持 {len(_categories)} 个类别: {', '.join(_categories)}")
    
    # 预热模型
    try:
        dummy_text = np.zeros((1, _config['max_sequence_length']), dtype=np.int32)
        dummy_filename = np.zeros((1, _config['max_filename_length']), dtype=np.int32)
        _model.predict([dummy_text, dummy_filename], verbose=0)
        print("模型预热完成\n")
    except:
        pass


# ---------- PPTX 文本提取 ----------
def extract_text_from_pptx(pptx_path):
    """从PPTX文件中提取所有文本内容"""
    text_parts = []
    prs = None
    
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
            
            del slide
        
        result = " ".join(text_parts)
        
    except Exception as e:
        result = ""
    finally:
        if prs is not None:
            del prs
        text_parts.clear()
        del text_parts
    
    return result


# ---------- 文本预处理 ----------
def clean_text(text):
    """基础清洗"""
    if not text:
        return ""
    
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    text = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9，。！？；：""''（）【】《》、 ]', '', text)
    text = re.sub(r'\s+', ' ', text)
    
    return text.strip()


def cut_words(text):
    """分词并过滤停用词"""
    if not text:
        return ""
    
    words = jieba.cut(text, HMM=False)
    filtered = []
    for w in words:
        if not w or w.strip() == '':
            continue
        if w not in STOPWORDS:
            filtered.append(w)
    
    if not filtered:
        words = jieba.cut(text, HMM=False)
        filtered = [w for w in words if w.strip()]
    
    if not filtered:
        return "空文本"
    
    result = " ".join(filtered)
    filtered.clear()
    del filtered
    
    return result


def extract_filename_features(filepath):
    """提取文件名特征"""
    filepath_obj = Path(filepath) if isinstance(filepath, str) else filepath
    filename = filepath_obj.stem
    
    cleaned = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9]', ' ', filename)
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    
    words = jieba.cut(cleaned, HMM=False)
    filtered = []
    for w in words:
        if not w or w.strip() == '':
            continue
        if w not in STOPWORDS:
            if len(w) > 1 or w in MEANINGFUL_SINGLE_CHARS:
                filtered.append(w)
    
    if not filtered:
        words = jieba.cut(cleaned, HMM=False)
        filtered = [w for w in words if w.strip()]
    
    if not filtered:
        return cleaned
    
    result = " ".join(filtered)
    filtered.clear()
    del filtered
    
    return result


# ---------- 核心预测函数 ----------
def predict_pptx_internal(filepath):
    """
    内部预测函数，返回完整结果（包含关键词）
    """
    global _model, _text_tokenizer, _filename_tokenizer, _categories, _config, _prediction_count
    
    filepath = Path(filepath)
    
    if not filepath.exists():
        raise FileNotFoundError(f"文件不存在: {filepath}")
    
    # 提取特征
    filename_feature = extract_filename_features(str(filepath))
    raw_text = extract_text_from_pptx(str(filepath))
    
    if raw_text:
        cleaned = clean_text(raw_text)
        text_feature = cut_words(cleaned)
        text_available = True
        # 提取关键词（使用原始文本，而非清洗后的）
        keywords = extract_keywords(raw_text)
        filename_keywords = extract_filename_keywords(filename_feature)
    else:
        text_feature = "空文本"
        text_available = False
        keywords = ["无文本内容"]
        filename_keywords = extract_filename_keywords(filename_feature)
    
    # 转换为序列
    text_seq = _text_tokenizer.texts_to_sequences([text_feature])
    text_pad = pad_sequences(text_seq, maxlen=_config['max_sequence_length'], 
                            padding='post', truncating='post')
    
    filename_seq = _filename_tokenizer.texts_to_sequences([filename_feature])
    filename_pad = pad_sequences(filename_seq, maxlen=_config['max_filename_length'],
                                padding='post', truncating='post')
    
    # 预测
    import io
    old_stdout = sys.stdout
    sys.stdout = io.StringIO()
    
    try:
        predictions = _model.predict([text_pad, filename_pad], verbose=0, batch_size=1)
    finally:
        sys.stdout = old_stdout
    
    # 获取结果
    pred_idx = predictions.argmax()
    confidence = float(predictions[0][pred_idx])
    predicted_class = _categories[pred_idx]
    
    # 获取所有类别的置信度
    all_probabilities = {
        cat: float(predictions[0][i]) 
        for i, cat in enumerate(_categories)
    }
    
    # 释放内存
    del text_seq, text_pad, filename_seq, filename_pad, predictions
    
    _prediction_count += 1
    if _prediction_count % 5 == 0:
        gc.collect()
        tf.keras.backend.clear_session()
    
    return {
        'filename': filepath.name,
        'filepath': str(filepath),
        'predicted_class': predicted_class,
        'confidence': confidence,
        'all_probabilities': all_probabilities,
        'text_available': text_available,
        'text_length': len(text_feature.split()),
        'filename_features': filename_feature,
        'keywords': keywords,
        'filename_keywords': filename_keywords
    }


# ---------- API 路由 ----------
@app.route('/health', methods=['GET'])
def health_check():
    """健康检查接口"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': _model is not None,
        'categories': _categories,
        'timestamp': datetime.now().isoformat()
    })


@app.route('/predict', methods=['POST'])
def predict_single():
    """
    单文件预测接口
    请求：POST multipart/form-data，文件字段名为 'file'
    返回：JSON格式的预测结果（包含关键词）
    """
    if _model is None:
        return jsonify({'error': '模型未加载'}), 503
    
    # 检查是否有文件上传
    if 'file' not in request.files:
        return jsonify({'error': '请上传PPTX文件', 'required_field': 'file'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': '未选择文件'}), 400
    
    # 检查文件类型
    if not file.filename.lower().endswith(('.pptx', '.ppt')):
        return jsonify({'error': '不支持的文件格式，请上传PPTX或PPT文件'}), 400
    
    # 保存临时文件
    temp_file = None
    try:
        with tempfile.NamedTemporaryFile(suffix='.pptx', delete=False) as tmp:
            file.save(tmp.name)
            temp_file = tmp.name
        
        # 预测
        result = predict_pptx_internal(temp_file)
        
        # 添加额外信息
        result['success'] = True
        result['top_3'] = sorted(
            result['all_probabilities'].items(), 
            key=lambda x: x[1], reverse=True
        )[:3]
        
        # 添加关键词统计
        result['keyword_count'] = len(result['keywords'])
        
        return jsonify(result)
        
    except FileNotFoundError as e:
        return jsonify({'error': str(e)}), 404
    except Exception as e:
        return jsonify({'error': f'预测失败: {str(e)}'}), 500
    finally:
        # 清理临时文件
        if temp_file and os.path.exists(temp_file):
            try:
                os.unlink(temp_file)
            except:
                pass


@app.route('/predict/batch', methods=['POST'])
def predict_batch():
    """
    批量文件预测接口
    请求：POST multipart/form-data，文件字段名为 'files'（多文件）
    返回：JSON格式的批量预测结果（包含关键词）
    """
    if _model is None:
        return jsonify({'error': '模型未加载'}), 503
    
    if 'files' not in request.files:
        return jsonify({'error': '请上传PPTX文件', 'required_field': 'files'}), 400
    
    files = request.files.getlist('files')
    if not files or len(files) == 0:
        return jsonify({'error': '未选择任何文件'}), 400
    
    # 过滤有效文件
    valid_files = [f for f in files if f.filename and f.filename.lower().endswith(('.pptx', '.ppt'))]
    
    if len(valid_files) == 0:
        return jsonify({'error': '没有有效的PPTX文件，请上传.pptx或.ppt格式的文件'}), 400
    
    results = []
    errors = []
    
    for file in valid_files:
        temp_file = None
        try:
            with tempfile.NamedTemporaryFile(suffix='.pptx', delete=False) as tmp:
                file.save(tmp.name)
                temp_file = tmp.name
            
            result = predict_pptx_internal(temp_file)
            results.append({
                'success': True,
                'filename': result['filename'],
                'predicted_class': result['predicted_class'],
                'confidence': result['confidence'],
                'text_available': result['text_available'],
                'text_length': result['text_length'],
                'keywords': result['keywords'][:5],  # 批量时只返回前5个关键词
                'filename_keywords': result['filename_keywords']
            })
            
        except Exception as e:
            errors.append({
                'success': False,
                'filename': file.filename,
                'error': str(e)
            })
        finally:
            if temp_file and os.path.exists(temp_file):
                try:
                    os.unlink(temp_file)
                except:
                    pass
    
    # 统计信息
    summary = {}
    for r in results:
        if r['success']:
            cat = r['predicted_class']
            summary[cat] = summary.get(cat, 0) + 1
    
    return jsonify({
        'success': True,
        'total': len(valid_files),
        'success_count': len(results),
        'error_count': len(errors),
        'summary': summary,
        'results': results,
        'errors': errors,
        'timestamp': datetime.now().isoformat()
    })


@app.route('/predict/url', methods=['POST'])
def predict_from_url():
    """
    从URL下载文件并预测
    请求：JSON格式，包含 'url' 字段
    返回：JSON格式的预测结果（包含关键词）
    """
    if _model is None:
        return jsonify({'error': '模型未加载'}), 503
    
    data = request.get_json()
    if not data or 'url' not in data:
        return jsonify({'error': '请提供文件URL', 'required_field': 'url'}), 400
    
    url = data['url']
    
    # 检查URL是否指向PPTX文件
    if not url.lower().endswith(('.pptx', '.ppt')):
        return jsonify({'error': 'URL必须指向PPTX或PPT文件'}), 400
    
    temp_file = None
    try:
        import urllib.request
        
        # 下载文件
        with tempfile.NamedTemporaryFile(suffix='.pptx', delete=False) as tmp:
            urllib.request.urlretrieve(url, tmp.name)
            temp_file = tmp.name
        
        # 预测
        result = predict_pptx_internal(temp_file)
        
        result['success'] = True
        result['source_url'] = url
        result['top_3'] = sorted(
            result['all_probabilities'].items(), 
            key=lambda x: x[1], reverse=True
        )[:3]
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'error': f'下载或预测失败: {str(e)}'}), 500
    finally:
        if temp_file and os.path.exists(temp_file):
            try:
                os.unlink(temp_file)
            except:
                pass


@app.route('/keywords/extract', methods=['POST'])
def extract_keywords_only():
    """
    仅提取关键词接口（不进行预测）
    请求：POST JSON格式，包含 'text' 字段
    返回：提取的关键词
    """
    data = request.get_json()
    if not data or 'text' not in data:
        return jsonify({'error': '请提供文本内容', 'required_field': 'text'}), 400
    
    text = data['text']
    top_n = data.get('top_n', KEYWORD_TOP_N)
    
    keywords = extract_keywords(text, top_n=top_n)
    
    return jsonify({
        'success': True,
        'keywords': keywords,
        'keyword_count': len(keywords),
        'text_length': len(text)
    })


@app.route('/categories', methods=['GET'])
def get_categories():
    """获取所有支持的类别"""
    return jsonify({
        'categories': _categories,
        'count': len(_categories),
        'timestamp': datetime.now().isoformat()
    })


@app.route('/stats', methods=['GET'])
def get_stats():
    """获取服务统计信息"""
    try:
        import psutil
        process = psutil.Process()
        memory_info = process.memory_info()
        memory_mb = memory_info.rss / 1024 / 1024
    except:
        memory_mb = -1
    
    return jsonify({
        'predictions_count': _prediction_count,
        'memory_usage_mb': memory_mb,
        'model_loaded': _model is not None,
        'categories': _categories,
        'keyword_config': {
            'top_n': KEYWORD_TOP_N,
            'min_n': KEYWORD_MIN_N,
            'weighted': KEYWORD_WEIGHTED
        },
        'timestamp': datetime.now().isoformat()
    })


# ---------- 错误处理 ----------
@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': '接口不存在'}), 404


@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': '服务器内部错误'}), 500


# ---------- 启动服务 ----------
def main():
    global _prediction_count
    _prediction_count = 0
    
    # 加载模型
    print("="*50)
    print("PPTX学科分类器 API服务启动（带关键词提取）")
    print("="*50)
    load_models()
    
    # 获取端口配置
    port = int(os.environ.get('PORT', 5000))
    host = os.environ.get('HOST', '0.0.0.0')
    debug = os.environ.get('DEBUG', 'False').lower() == 'true'
    
    print(f"\n服务启动信息:")
    print(f"  地址: http://{host}:{port}")
    print(f"  调试模式: {debug}")
    print(f"\n可用接口:")
    print(f"  GET  /health           - 健康检查")
    print(f"  GET  /categories       - 获取类别列表")
    print(f"  GET  /stats            - 获取统计信息")
    print(f"  POST /predict          - 单文件预测（含关键词）")
    print(f"  POST /predict/batch    - 批量预测（含关键词）")
    print(f"  POST /predict/url      - URL预测（含关键词）")
    print(f"  POST /keywords/extract - 仅提取关键词")
    print("="*50)
    
    app.run(host=host, port=port, debug=debug, threaded=True)


if __name__ == "__main__":
    # 确保有意义的单字词定义
    MEANINGFUL_SINGLE_CHARS = {'圆', '力', '氧', '氢', '碳', '钠', '酸', '碱', '盐', 
                               '电', '光', '声', '热', '诗', '词', '歌', '曲', '数',
                               '方', '程', '函', '数', '角', '形', '体', '积'}
    main()