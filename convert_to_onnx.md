# 📘 TextCNN ONNX 导出说明（已集成到 `train.py`）

## 1. 概述

自本次改造起，ONNX 转换**不再是独立脚本**（原 `convert_to_onnx.py` 已删除），
而是作为训练流程的最后一步（步骤 8）内置于 `train.py`：

```bash
python train.py     # 训练 → 评估 → 保存 Keras 产物 → 自动导出 ONNX
```

一次训练即可得到可直接部署的 ONNX 模型与推理所需文件。

### 1.1 特点

- **支持双输入模型**：`text_input`（文本序列）+ `filename_input`（文件名序列）
- **自动读取配置**：从 `models/tensorflow/config_optimized.pkl` 获取输入形状
- **自动复制推理依赖**：Tokenizer / 类别 / 配置复制到 `models/onnx/`
- **双重校验**：`onnx.checker` 完整性校验 + `onnxruntime` 前向校验
- **优雅降级**：缺少 `tf2onnx`/`onnx` 时仅跳过导出，训练结果不受影响

### 1.2 适用场景

- 生产环境部署（无需安装 TensorFlow）
- 资源受限环境（内存占用更小、启动更快）
- 跨平台推理（Windows / Linux / macOS）

---

## 2. 环境准备

```bash
pip install tensorflow tf2onnx onnx onnxruntime numpy
```

验证依赖：

```python
import tf2onnx
import onnx
print(f"tf2onnx: {tf2onnx.__version__}")
print(f"ONNX: {onnx.__version__}")
```

> 未安装 `tf2onnx` / `onnx` 时，`train.py` 会打印提示并跳过 ONNX 导出。

---

## 3. 使用方法

### 3.1 训练并自动导出（默认，推荐）

```bash
python train.py
```

训练结束时会执行“步骤8: 导出ONNX模型”，输出 `models/onnx/textcnn_classifier.onnx`。

### 3.2 仅重新导出（不重新训练）

```bash
# 使用默认路径（models/tensorflow/ 下的 Keras 模型与配置）
python train.py --export-onnx-only

# 指定其它模型/配置
python train.py --export-onnx-only \
  --keras-model models/tensorflow/textcnn_optimized_classifier.keras \
  --config models/tensorflow/config_optimized.pkl
```

### 3.3 关闭自动导出

编辑 `train.py` 顶部配置：

```python
ENABLE_ONNX_EXPORT = False    # 训练后不导出 ONNX
```

---

## 4. 输出文件

### 4.1 输入（`models/tensorflow/`，由训练生成）

| 文件名 | 说明 |
|--------|------|
| `textcnn_optimized_classifier.keras` | 完整训练好的 Keras 模型 |
| `config_optimized.pkl` | 训练配置（含 `max_sequence_length`、`max_filename_length`） |

### 4.2 输出（`models/onnx/`，可直接随 `run/` 部署）

| 文件名 | 说明 |
|--------|------|
| `textcnn_classifier.onnx` | ONNX 模型（opset 13） |
| `text_tokenizer_none.pkl` | 文本分词器（自动复制） |
| `filename_tokenizer_none.pkl` | 文件名分词器（自动复制） |
| `categories.pkl` | 类别列表（自动复制） |
| `config_optimized.pkl` | 训练配置（自动复制） |

---

## 5. 关键配置（`train.py` 顶部）

| 变量 | 说明 | 默认值 |
|------|------|--------|
| `MODEL_ROOT_DIR` | 模型输出根目录 | `models` |
| `TF_MODEL_DIR` | TensorFlow/Keras 产物目录 | `models/tensorflow` |
| `ONNX_MODEL_DIR` | ONNX 产物目录 | `models/onnx` |
| `ENABLE_ONNX_EXPORT` | 是否训练后自动导出 | `True` |
| `ONNX_OPSET` | ONNX opset 版本 | `13` |
| `ONNX_RUNTIME_FILES` | 需要复制到 ONNX 目录的推理文件 | 4 个 `.pkl` |

关键函数：

| 函数 | 作用 |
|------|------|
| `export_onnx_model()` | 核心导出函数（转换 + 校验 + 复制依赖文件） |
| `export_onnx_from_saved()` | `--export-onnx-only` 模式：加载已保存模型后导出 |
| `tf2onnx.convert.from_keras()` | Keras → ONNX 转换 API（显式双输入签名） |
| `onnx.checker.check_model()` | ONNX 模型校验 |

---

## 6. 常见问题

### Q1：`ModuleNotFoundError: No module named 'tf2onnx'`

```bash
pip install tf2onnx onnx
```

训练本身不受影响，只是不会生成 ONNX 文件；可稍后执行
`python train.py --export-onnx-only` 补导出。

### Q2：提示 `未找到Keras模型: models/tensorflow/...keras`

先运行 `python train.py` 完成训练，或用 `--keras-model` 指定已有的 `.keras` 文件。

### Q3：输入张量顺序/名称

转换使用显式输入签名，顺序固定为：

1. `text_input`：`(batch, max_sequence_length)`，`int32`
2. `filename_input`：`(batch, max_filename_length)`，`int32`

`run/predict_onnx.py` 按名称自动匹配输入，无需手工调整。

### Q4：`ValueError: opset version not supported`

在 `train.py` 中调整 `ONNX_OPSET`（如 `12` 或 `15`）后重新导出。

### Q5：ONNX Runtime 前向校验失败

- 确认 `models/onnx/` 与 `models/tensorflow/` 来自同一次训练
- 检查 `config_optimized.pkl` 中的序列长度与模型是否一致
- 前向校验失败不影响模型文件生成，可结合 `run/predict_onnx.py` 实测

---

## 7. 相关文件

| 文件 | 说明 |
|------|------|
| `train.py` | 训练 + ONNX 导出（`export_onnx_model` / `export_onnx_from_saved`） |
| `run/predict_onnx.py` | ONNX 推理脚本（读取 `models/onnx/`） |
| `run/predict_onnx_ocrtest.py` | ONNX 推理 + OCR 图片文字识别 |
| `run/server-api.py` | ONNX 推理 API 服务 |

---

**最后更新**：2026-09-18
