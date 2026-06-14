# 📘 文件学科分类器训练脚本使用文档（TextCNN 优化版）

## 1. 概述

本脚本用于训练一个基于 **TextCNN** 的多模态分类模型，支持以下学科类别：

- 语文
- 数学
- 英语
- 物理
- 化学
- 生物
- 班会

支持的文件格式：

- `.pptx` / `.ppt`（PowerPoint）
- `.docx`（Word）

核心特性：

- 文本内容权重 70%，文件名权重 30%
- 文件名数据增强
- 多尺度卷积（kernel size = 2,3,4,5）
- 文件解析缓存（避免重复解析）
- 类别权重处理（缓解类别不平衡）
- 支持指定科目仅使用缓存（跳过解析）
- 自动去除人名（基于 jieba 词性标注）
- 词表截断（限制在 20000 词以内）
- 输出 JSON 格式词表，便于外部工具使用

---

## 2. 环境准备

### 2.1 安装依赖

推荐使用 `conda` 或 `pip` 安装以下依赖：

```bash
pip install numpy tensorflow scikit-learn python-pptx jieba python-docx matplotlib
```

### 2.2 验证依赖

```python
import pptx
import docx
import jieba
print("所有依赖已安装")
```

---

## 3. 数据准备

### 3.1 目录结构

```text
../data/
├── 语文/
│   ├── 1.pptx
│   ├── 2.docx
│   └── ...
├── 数学/
├── 英语/
├── 物理/
├── 化学/
├── 生物/
└── 班会/
```

### 3.2 文件命名建议

- 可使用中文、英文、数字组合
- 脚本会自动提取文件名作为辅助特征
- 支持数据增强（如去除数字、仅保留中文等）

---

## 4. 配置说明

脚本中所有关键参数均在 **“优化后的配置参数”** 部分定义，常用配置如下：

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `DATA_ROOT` | 数据目录路径 | `"../data"` |
| `CACHE_DIR` | 缓存目录 | `"../cache"` |
| `CATEGORIES` | 分类类别列表 | 7 个学科 |
| `SKIP_CATEGORIES` | 仅使用缓存的科目 | `[]` |
| `MAX_NB_WORDS` | 最大词表大小 | 20000 |
| `MAX_SEQUENCE_LENGTH` | 文本最大长度 | 1000 |
| `EMBEDDING_DIM` | 文本向量维度 | 150 |
| `TEXT_WEIGHT` / `FILENAME_WEIGHT` | 权重比例 | 0.7 / 0.3 |
| `ENABLE_CACHE` | 是否启用缓存 | `True` |
| `REMOVE_PERSON_NAMES` | 是否去除人名 | `True` |
| `USE_CLASS_WEIGHTS` | 是否使用类别权重 | `True` |
| `EPOCHS` | 训练轮数 | 40 |
| `BATCH_SIZE` | 批量大小 | 12 |

---

## 5. 运行训练

### 5.1 基本运行

```bash
python train.py
```

### 5.2 强制刷新缓存

```python
FORCE_REFRESH_CACHE = True
```

### 5.3 指定科目仅使用缓存

```python
SKIP_CATEGORIES = ["语文", "数学"]
```

> 注意：这些科目的文件必须已存在于缓存中，否则将被跳过。

---

## 6. 训练流程说明

1. **加载文件**  
   - 扫描 `DATA_ROOT` 下的所有文件
   - 使用缓存加速解析（支持 `.pptx`, `.ppt`, `.docx`）
   - 自动去除人名（基于 jieba 词性标注）

2. **数据增强**  
   - 对文件名生成多种变体（去数字、仅中文等）

3. **数据集划分**  
   - 训练集（70%）、验证集（15%）、测试集（15%）

4. **构建词表**  
   - 使用 `Tokenizer` 构建文本和文件名词表
   - 截断到 `MAX_NB_WORDS` 和 `MAX_FILENAME_WORDS`

5. **模型构建**  
   - 文本分支：多尺度 CNN + GlobalMaxPooling
   - 文件名分支：CNN + GlobalMaxPooling
   - 融合后输出分类

6. **训练**  
   - 使用 EarlyStopping、ModelCheckpoint、ReduceLROnPlateau 回调

7. **评估 & 保存**  
   - 输出分类报告、混淆矩阵
   - 保存模型、Tokenizer、JSON 词表等

---

## 7. 输出文件说明

训练完成后，脚本会在当前目录生成以下文件：

| 文件名 | 说明 |
|--------|------|
| `textcnn_optimized_classifier.keras` | 完整训练好的 Keras 模型 |
| `best_model_optimized.keras` | 验证集最佳模型 |
| `text_tokenizer.pkl` | 文本 Tokenizer（完整） |
| `filename_tokenizer.pkl` | 文件名 Tokenizer（完整） |
| `text_tokenizer_none.pkl` | 无依赖 Tokenizer（截断） |
| `filename_tokenizer_none.pkl` | 无依赖 Tokenizer（截断） |
| `categories.pkl` | 类别列表 |
| `config_optimized.pkl` | 训练配置 |
| `text_vocabulary.json` | 文本词表（JSON） |
| `filename_vocabulary.json` | 文件名词表（JSON） |
| `category_mapping.json` | 类别映射 |
| `word_frequency_report.json` | 词频统计报告 |
| `training_history.png` | 训练曲线图 |
| `../cache/file_cache_*.json` | 文件内容缓存 |
| `../cache/name_removal_stats_*.json` | 人名去除统计 |

---

## 8. 常见问题

### Q1：提示 `python-docx` 未安装

```bash
pip install python-docx
```

### Q2：缓存命中率低

- 确保文件未被修改
- 可设置 `FORCE_REFRESH_CACHE = True` 重建缓存

### Q3：某些科目文件未被加载

- 检查目录名是否与 `CATEGORIES` 完全一致
- 检查文件扩展名是否在 `SUPPORTED_EXTENSIONS` 中

### Q4：人名去除是否影响分类效果？

- 去除常见人名（如“李明说”）有助于减少噪声
- 如果效果不佳，可设置 `REMOVE_PERSON_NAMES = False`

### Q5：如何只使用缓存不解析新文件？

- 设置 `SKIP_CATEGORIES = ["语文", "数学"]` 等需要跳过的科目
- 其他科目仍会解析并更新缓存

---

## 9. 示例命令（完整运行）

```bash
# 1. 准备数据
mkdir -p ../data/语文 ../data/数学

# 2. 放入文件
cp example.pptx ../data/语文/
cp example.docx ../data/数学/

# 3. 运行训练
python train.py
```

---

## 10. 使用训练后的模型
[使用预测（进行测试）](./predict.md)
[使用分类](./classifer.md)

---

## 11. 更新计划
1. 添加OCR图片识别
2. 添加更细致的过滤规则 `.moverignore` 文件

---

## 11. 联系与扩展

- 如需增加新类别，请修改 `CATEGORIES` 并确保目录存在
- 如需调整文本/文件名权重，修改 `TEXT_WEIGHT` 和 `FILENAME_WEIGHT`
- 模型推理脚本可基于保存的 Tokenizer 和模型构建

---

**版本**：v5  
**最后更新**：2026-06-06