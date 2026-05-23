# PPTX 学科分类器 - 使用文档

## 项目简介

这是一个基于 TextCNN 的多模态深度学习模型，用于自动分类 PPTX 文件所属的学科类别。系统同时分析**PPT文本内容**（权重70%）和**文件名**（权重30%），支持以下7个类别：

- 语文、数学、英语、物理、化学、生物、班会

## 文件说明

| 文件 | 功能 |
|------|------|
| `classifer.py` | 模型训练脚本 |
| `predict.py` | 命令行预测工具 |
| `server-api.py` | RESTful API 服务 |

---

## 第一部分：环境配置

### 1.1 安装依赖

```bash
pip install tensorflow python-pptx jieba scikit-learn numpy flask flask-cors psutil
```

### 1.2 目录结构

```
项目目录/
├── classifer.py          # 训练脚本
├── predict.py            # 预测工具
├── server-api.py         # API服务
├── ../data/              # 训练数据目录（与脚本同级或按配置）
│   ├── 语文/             # 放入语文PPTX文件
│   ├── 数学/             # 放入数学PPTX文件
│   ├── 英语/             # 放入英语PPTX文件
│   ├── 物理/             # 放入物理PPTX文件
│   ├── 化学/             # 放入化学PPTX文件
│   ├── 生物/             # 放入生物PPTX文件
│   └── 班会/             # 放入班会PPTX文件
└── 输出文件/             # 训练后生成的模型文件
```

---

## 第二部分：模型训练 (`classifer.py`)

### 2.1 基本使用

```bash
python classifer.py
```

### 2.2 核心配置参数（修改脚本开头的配置区）

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `DATA_ROOT` | `"../data"` | 训练数据目录 |
| `TEXT_WEIGHT` | `0.70` | 文本内容权重 |
| `FILENAME_WEIGHT` | `0.30` | 文件名权重 |
| `ENABLE_FILENAME_AUGMENTATION` | `True` | 是否启用文件名数据增强 |
| `MAX_SEQUENCE_LENGTH` | `750` | 文本最大长度 |
| `BATCH_SIZE` | `16` | 批次大小 |
| `EPOCHS` | `32` | 训练轮数 |
| `ENABLE_CROSS_VALIDATION` | `False` | 是否启用交叉验证 |

### 2.3 输出文件

训练完成后生成以下文件：

| 文件 | 用途 |
|------|------|
| `textcnn_balanced_classifier.keras` | 最终模型 |
| `best_model_balanced.keras` | 最佳模型（验证集最优） |
| `text_tokenizer.pkl` | 文本分词器 |
| `filename_tokenizer.pkl` | 文件名词典 |
| `categories.pkl` | 类别映射 |
| `config_balanced.pkl` | 模型配置 |

### 2.4 训练数据要求

- 每个类别目录下放置对应的 `.pptx` 文件
- 建议每个类别至少 **30-50** 个文件以保证效果
- 文件名和文本内容越有学科特征，效果越好

---

## 第三部分：命令行预测 (`predict.py`)

### 3.1 单文件预测

```bash
python predict.py /path/to/your/file.pptx
```

输出示例：
```
==================================================
文件: 三角函数复习.pptx
==================================================
📄 文件名: 三角函数 复习
📝 文本长度: 1250 词

🎯 预测结果: 数学
📊 置信度: 94.32%

📈 详细分类概率:
   数学: 94.32% ██████████████████████████
   物理:  3.12% ███
   化学:  1.56% ██
```

### 3.2 批量预测目录

```bash
# 预测目录下所有PPTX
python predict.py /path/to/directory --batch

# 递归搜索子目录
python predict.py /path/to/directory --batch --recursive
```

### 3.3 交互式模式

```bash
python predict.py --interactive
```

进入交互模式后：
- 直接输入文件路径进行预测
- 输入 `quit` 退出
- 输入 `mem` 查看内存使用（需安装psutil）

### 3.4 命令行参数说明

| 参数 | 简写 | 说明 |
|------|------|------|
| `input` | - | PPTX文件路径或目录路径 |
| `--batch` | `-b` | 启用批量预测模式 |
| `--recursive` | `-r` | 递归搜索子目录 |
| `--interactive` | `-i` | 启用交互式模式 |

---

## 第四部分：API 服务 (`server-api.py`)

### 4.1 启动服务

```bash
python server-api.py
```

可配置环境变量：
```bash
PORT=8080 python server-api.py          # 指定端口
HOST=127.0.0.1 python server-api.py     # 指定主机
DEBUG=true python server-api.py          # 开启调试模式
```

### 4.2 API 接口列表

#### 4.2.1 健康检查
```http
GET /health
```
响应：
```json
{
  "status": "healthy",
  "model_loaded": true,
  "categories": ["语文", "数学", "英语", "物理", "化学", "生物", "班会"],
  "timestamp": "2026-01-15T10:30:00"
}
```

#### 4.2.2 单文件预测（含关键词）
```http
POST /predict
Content-Type: multipart/form-data

file: <PPTX文件>
```
响应：
```json
{
  "success": true,
  "filename": "三角函数复习.pptx",
  "predicted_class": "数学",
  "confidence": 0.9432,
  "keywords": ["三角函数", "正弦", "余弦", "公式", "角度"],
  "filename_keywords": ["三角函数", "复习"],
  "text_length": 1250,
  "top_3": [
    ["数学", 0.9432],
    ["物理", 0.0312],
    ["化学", 0.0156]
  ]
}
```

#### 4.2.3 批量预测
```http
POST /predict/batch
Content-Type: multipart/form-data

files: <多个PPTX文件>
```
响应：包含每个文件的预测结果和汇总统计

#### 4.2.4 URL 预测
```http
POST /predict/url
Content-Type: application/json

{
  "url": "https://example.com/file.pptx"
}
```

#### 4.2.5 仅提取关键词（不预测）
```http
POST /keywords/extract
Content-Type: application/json

{
  "text": "这里是PPTX中提取的文本内容...",
  "top_n": 10
}
```

#### 4.2.6 获取类别列表
```http
GET /categories
```

#### 4.2.7 获取服务统计
```http
GET /stats
```

### 4.3 cURL 调用示例

```bash
# 单文件预测
curl -X POST -F "file=@/path/to/file.pptx" http://localhost:5000/predict

# 批量预测
curl -X POST -F "files=@file1.pptx" -F "files=@file2.pptx" http://localhost:5000/predict/batch

# URL预测
curl -X POST -H "Content-Type: application/json" \
  -d '{"url":"https://example.com/file.pptx"}' \
  http://localhost:5000/predict/url
```

---

## 第五部分：注意事项

### 5.1 环境要求

| 依赖 | 推荐版本 | 说明 |
|------|----------|------|
| Python | 3.8 - 3.10 | TensorFlow 2.13+ 对 3.11+ 支持不完善 |
| TensorFlow | 2.13.0 | 使用 `compile=False` 加载模型 |
| 内存 | 2GB+ | 模型加载和预测需要 |
| 磁盘 | 100MB | 存储模型文件（约40MB） |

### 5.2 常见问题

#### Q1: 模型加载失败？
确保以下文件存在于当前目录：
- `textcnn_balanced_classifier.keras`
- `text_tokenizer.pkl`
- `filename_tokenizer.pkl`
- `categories.pkl`
- `config_balanced.pkl`

#### Q2: 中文分词问题？
脚本已集成 jieba 分词，首次运行会自动下载词典。如遇分词异常：
```python
jieba.set_dictionary('path/to/custom/dict.txt')  # 使用自定义词典
```

#### Q3: 内存持续增长？
脚本已针对交互式使用优化：
- 每5次预测后自动执行 `gc.collect()`
- 批量预测每10个文件清理一次内存
- API服务使用临时文件，预测后自动删除

#### Q4: 预测准确率低？
- 增加训练样本数量（每个类别建议50+）
- 调整 `TEXT_WEIGHT` 和 `FILENAME_WEIGHT` 权重
- 启用文件名数据增强（`ENABLE_FILENAME_AUGMENTATION = True`）
- 启用交叉验证评估泛化能力（`ENABLE_CROSS_VALIDATION = True`）

#### Q5: GPU 支持？
- GPU 会自动使用，无需特殊配置
- 可通过 `CUDA_VISIBLE_DEVICES=""` 禁用 GPU

#### Q6: 不支持 .ppt 格式？
脚本主要支持 `.pptx` 格式。对于旧版 `.ppt` 格式，建议先转换为 `.pptx`。

### 5.3 性能建议

| 场景 | 建议 |
|------|------|
| 单次预测 | 直接使用 `predict.py` |
| 批量预测 | 使用 `predict.py --batch` |
| Web服务 | 使用 `server-api.py`，配置 `threaded=True` |
| 高频调用 | 保持模型常驻内存，重复使用 `predict_pptx_internal` |

### 5.4 安全注意事项

- API 服务默认监听 `0.0.0.0:5000`，生产环境建议：
  - 使用反向代理（Nginx）
  - 添加认证机制
  - 限制文件大小（建议 < 50MB）
- 临时文件自动清理，但高并发下需注意磁盘空间

---

## 第六部分：扩展开发

### 6.1 添加新类别

1. 修改 `classifer.py` 中的 `CATEGORIES` 列表
2. 在 `data/` 目录下创建对应文件夹
3. 放入训练文件，重新训练

### 6.2 调整权重

修改 `classifer.py` 中的：
```python
TEXT_WEIGHT = 0.70      # 文本权重
FILENAME_WEIGHT = 0.30  # 文件名权重
```

### 6.3 自定义停用词

修改脚本中的 `STOPWORDS` 集合：
```python
STOPWORDS = set(['的', '了', '是', ...])  # 添加自定义停用词
```

---

## 联系方式

如有问题，请检查：
1. 所有依赖是否正确安装
2. 模型文件是否完整
3. 数据目录结构是否正确
4. Python 版本是否为 3.8-3.10