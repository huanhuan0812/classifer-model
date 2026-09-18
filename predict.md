# 📘 文件学科分类器预测脚本使用文档（ONNX 版本）

## 1. 概述

本脚本用于对 **PPTX、PPT、DOCX** 文件进行学科分类预测，使用 **ONNX Runtime** 进行推理，具有以下特点：

- **零 Keras 依赖**，无需安装 TensorFlow
- **内存占用小**，推理速度快
- **支持 7 个学科类别**：语文、数学、英语、物理、化学、生物、班会
- **支持文件格式**：`.pptx`、`.ppt`、`.docx`
- **自动提取嵌入的 DOCX 文件内容**（从 PPTX 中）
- **支持交互式、单文件、批量预测**

---

## 2. 环境准备

### 2.1 安装依赖

```bash
pip install numpy onnxruntime jieba python-pptx python-docx
```

### 2.2 验证依赖

```python
import onnxruntime
import jieba
from pptx import Presentation
from docx import Document
print("所有依赖已安装")
```

### 2.3 所需模型文件

使用本脚本前，请确保以下文件存在于项目根目录的 `models/onnx/` 下
（运行 `python train.py` 会自动生成并复制，无需手工转换）：

| 文件名 | 说明 |
|--------|------|
| `textcnn_classifier.onnx` | ONNX 格式模型（训练后自动导出） |
| `text_tokenizer_none.pkl` | 文本分词器（无依赖版） |
| `filename_tokenizer_none.pkl` | 文件名词典（无依赖版） |
| `categories.pkl` | 类别映射 |
| `config_optimized.pkl` | 训练配置 |

> ⚠️ 注意：若模型缺失，可运行 `python train.py` 重新训练，
> 或运行 `python train.py --export-onnx-only` 由已有的 Keras 模型重新导出 ONNX。

---

## 3. 使用方法

### 3.1 命令行参数

```bash
python run/predict_onnx.py [选项] [输入路径]
```

| 参数 | 简写 | 说明 |
|------|------|------|
| `input` | - | 文件路径或目录路径 |
| `--batch` | `-b` | 批量预测模式 |
| `--recursive` | `-r` | 递归搜索子目录（与 `--batch` 配合） |
| `--interactive` | `-i` | 交互式预测模式 |

### 3.2 使用示例

#### ① 单文件预测

```bash
python run/predict_onnx.py 语文.pptx
```

```bash
python run/predict_onnx.py 数学.docx
```

输出示例：

```
==================================================
文件: 语文.pptx
类型: .PPTX
==================================================
📝 文本长度: 234 词
📝 文本可用: 是

🎯 预测结果: 语文
📊 置信度: 96.50%

📈 详细分类概率:
   语文 : 96.50% ████████████████████████████
   数学 :  2.10% █
   英语 :  0.80% ·
   ...
```

#### ② 交互式模式

```bash
python run/predict_onnx.py -i
```

进入交互式界面后，每行输入一个文件路径即可预测。

#### ③ 批量预测（单目录）

```bash
python run/predict_onnx.py /path/to/files --batch
```

#### ④ 批量预测（递归子目录）

```bash
python run/predict_onnx.py /path/to/data --batch --recursive
```

批量预测会生成 CSV 结果文件，格式如：

```
prediction_results_20260606_143022.csv
```

---

## 4. 核心功能说明

### 4.1 文件类型识别

脚本自动根据文件扩展名选择解析器：

| 扩展名 | 解析方式 |
|--------|----------|
| `.pptx` / `.ppt` | 使用 `python-pptx` 提取文本 + 自动提取嵌入 DOCX |
| `.docx` | 使用 `python-docx` 提取文本 |

### 4.2 嵌入 DOCX 提取

对于 PPTX 文件中嵌入的 DOCX 对象（如 Word 附件），脚本会自动：

1. 解压 PPTX（ZIP 格式）
2. 查找 `ppt/embeddings/*.docx`
3. 解析并提取其中的文本内容

> 提示：若 PPTX 本身无文本但嵌入了 DOCX，预测时会标注 `(from embedded)`。

### 4.3 文本预处理流程

```
原始文本 → 清洗（去URL、保留中英文数字）→ jieba分词 → 去停用词 → 空格连接
```

### 4.4 文件名特征提取

- 提取文件名（不含扩展名）
- 去除数字、英文、特殊符号
- 保留中文和有意义单字
- 同样进行分词和去停用词

### 4.5 索引压缩与 OOV 处理

脚本自动将 Tokenizer 词表压缩到模型支持的范围内（`1 ~ MODEL_VOCAB_SIZE-1`），未登录词映射为 `OOV token`（索引 1），确保不会出现索引越界。

---

## 5. 批量预测输出

批量预测会生成 CSV 文件，包含以下字段：

| 字段 | 说明 |
|------|------|
| `file` | 文件完整路径 |
| `file_type` | 文件扩展名 |
| `predicted_class` | 预测类别 |
| `confidence` | 置信度（0~1） |
| `text_available` | 是否提取到文本内容 |
| `embedded_used` | 是否使用了嵌入 DOCX 内容 |

同时会在控制台输出汇总统计：

```
============================================================
批量预测汇总
============================================================
总计: 156 个文件
成功: 152 个
失败: 4 个

文件类型分布:
  .pptx: 89 个
  .docx: 63 个
  .ppt: 4 个

类别分布:
  语文: 45 个 (29.6%)
  数学: 38 个 (25.0%)
  英语: 32 个 (21.1%)
  ...

📄 结果已导出到: ./prediction_results_20260606_143022.csv
```

---

## 6. 常见问题

### Q1：提示 `textcnn_classifier.onnx` 文件不存在

ONNX 模型由训练脚本自动导出，无需单独转换。若 `models/onnx/` 下缺少模型，可执行：

```bash
# 方式一：重新训练（训练结束自动导出 ONNX）
python train.py

# 方式二：仅用已有的 Keras 模型重新导出 ONNX
python train.py --export-onnx-only
```

> 需要额外依赖：`pip install tf2onnx onnx`。

### Q2：提示索引超出范围（如索引 20000）

- 确保使用的 Tokenizer 是 `text_tokenizer_none.pkl`（截断版本）
- 脚本会自动压缩索引，如果仍出现，请检查 `MODEL_VOCAB_SIZE` 配置是否与 ONNX 模型一致

### Q3：DOCX 文件解析失败

- 确保已安装 `python-docx`：`pip install python-docx`
- 如果仍失败，脚本会尝试简单正则提取（降级方案）

### Q4：PPTX 文件提取不到文本

- 检查 PPTX 是否真的包含文本（可能是纯图片）
- 脚本会自动尝试提取嵌入的 DOCX 附件
- 如果仍无文本，会使用“空文本”进行分类（仅依赖文件名）

### Q5：内存占用过高

- 脚本已开启 `gc.collect()` 和对象清理
- 可设置环境变量 `OMP_NUM_THREADS=1` 限制线程数
- 批量预测时每处理一个文件后自动释放内存

---

## 7. 性能优化建议

| 场景 | 建议 |
|------|------|
| 大批量文件 | 使用 `--batch` 模式，结果自动导出 CSV |
| 服务器部署 | 设置 `intra_op_num_threads=1`，避免 CPU 争抢 |
| 实时预测 | 保持模型常驻内存（全局 `_session`） |
| 内存受限环境 | 批量预测时减小 `BATCH_SIZE`（脚本单文件处理） |

---

## 8. 脚本内部类与函数速查

| 名称 | 作用 |
|------|------|
| `SimpleTokenizer` | 轻量级 Tokenizer，压缩索引范围 |
| `load_tokenizer_compressed()` | 加载并压缩 Tokenizer |
| `extract_text_from_pptx()` | 从 PPTX 提取文本（含嵌入 DOCX） |
| `extract_text_from_docx()` | 从 DOCX 提取文本 |
| `predict_file()` | 核心预测函数 |
| `predict_batch()` | 批量预测 + CSV 导出 |
| `interactive_mode()` | 交互式命令行界面 |

---

## 9. 与训练脚本的对应关系

| 训练脚本输出（`models/tensorflow/`） | 预测脚本使用（`models/onnx/`） |
|--------------|--------------|
| `textcnn_optimized_classifier.keras` | 训练结束时自动导出为 `textcnn_classifier.onnx` |
| `text_tokenizer_none.pkl` | ✅ 自动复制后直接使用 |
| `filename_tokenizer_none.pkl` | ✅ 自动复制后直接使用 |
| `categories.pkl` | ✅ 自动复制后直接使用 |
| `config_optimized.pkl` | ✅ 自动复制后读取配置（如序列长度） |

---

## 10. 完整示例流程


1. 步骤1：训练模型（自动生成 .keras / tokenizer 并导出 ONNX）
`python train.py`

2. 步骤2：确认 ONNX 产物
`models/onnx/` 下应包含 `textcnn_classifier.onnx` 等文件
（[ONNX 导出说明](./convert_to_onnx.md)）

3. 步骤3：单文件预测（在项目根目录执行）
`python run/predict_onnx.py 语文.pptx`

4. 步骤4：批量预测
`python run/predict_onnx.py ./test_files --batch --recursive`


---

## 11. 错误码说明

| 返回值 | 说明 |
|--------|------|
| 0 | 正常完成 |
| 1 | 模型加载失败或输入错误 |

---

**版本**：v1.1（ONNX Runtime，脚本位于 `run/`，模型位于 `models/onnx/`）
**兼容训练脚本版本**：v5
**最后更新**：2026-09-18
**最后更新**：2026-06-06