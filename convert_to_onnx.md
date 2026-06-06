# 📘 TextCNN Keras 转 ONNX 模型转换工具使用文档

## 1. 概述

本脚本用于将训练好的 **TextCNN 多模态分类模型**（Keras 格式）转换为 **ONNX 格式**，以便使用 ONNX Runtime 进行高效推理。

### 1.1 特点

- **支持双输入模型**：文本序列 + 文件名序列
- **自动读取配置**：从 `config_optimized.pkl` 获取输入形状
- **验证转换结果**：自动校验 ONNX 模型完整性
- **输出模型信息**：显示输入输出张量名称和形状

### 1.2 适用场景

- 生产环境部署（无需 TensorFlow）
- 资源受限环境（内存占用更小）
- 跨平台推理（Windows/Linux/macOS）
- 与其他 ONNX Runtime 应用集成

---

## 2. 环境准备

### 2.1 安装依赖

```bash
pip install tensorflow tf2onnx onnx numpy
```

### 2.2 验证依赖

```python
import tensorflow as tf
import tf2onnx
import onnx
print(f"TensorFlow: {tf.__version__}")
print(f"tf2onnx: {tf2onnx.__version__}")
print(f"ONNX: {onnx.__version__}")
```

### 2.3 所需输入文件

转换前请确保以下文件存在于当前目录：

| 文件名 | 说明 | 来源 |
|--------|------|------|
| `textcnn_optimized_classifier.keras` | Keras 模型文件 | `train.py` 生成 |
| `config_optimized.pkl` | 模型配置文件 | `train.py` 生成 |

---

## 3. 使用方法

### 3.1 基本用法

```bash
python convert_to_onnx.py
```

### 3.2 自定义路径（如需修改脚本）

编辑脚本中的配置变量：

```python
MODEL_PATH = "textcnn_optimized_classifier.keras"   # 输入模型路径
CONFIG_PATH = "config_optimized.pkl"               # 配置文件路径
OUTPUT_ONNX_PATH = "textcnn_classifier.onnx"       # 输出 ONNX 路径
```

### 3.3 执行输出示例

```
正在加载Keras模型...
加载配置...
输入形状: text_input=(None, 1000), filename_input=(None, 32)
正在转换为ONNX格式...
✅ 转换成功！模型已保存到: textcnn_classifier.onnx

验证ONNX模型...
✅ ONNX模型验证通过！

模型输入:
  - text_input: [0, 1000]
  - filename_input: [0, 32]

模型输出:
  - output: [0, 7]
```

---

## 4. 转换原理说明

### 4.1 双输入模型结构

原始 Keras 模型定义：

```python
text_input = Input(shape=(max_seq_len,), name='text_input')
filename_input = Input(shape=(max_filename_len,), name='filename_input')
# ... 网络层 ...
output = Dense(num_classes, activation='softmax', name='output')
model = Model(inputs=[text_input, filename_input], outputs=output)
```

### 4.2 ONNX 转换要点

| 参数 | 设置 | 说明 |
|------|------|------|
| `input_signature` | `[text_input, filename_input]` | 明确指定两个输入的 TensorSpec |
| `opset` | `13` | 平衡兼容性与功能支持 |
| `output_path` | 指定路径 | 保存转换后的 ONNX 模型 |

### 4.3 输入形状说明

| 输入名称 | 形状 | 说明 |
|----------|------|------|
| `text_input` | `[batch_size, max_seq_len]` | 文本序列，默认 max_seq_len=1000 |
| `filename_input` | `[batch_size, max_filename_len]` | 文件名序列，默认 max_filename_len=32 |

> 动态 batch 维度（`None`）会被保留，ONNX 中表示为 `0`。

---

## 5. 输出文件说明

### 5.1 生成的文件

| 文件名 | 说明 |
|--------|------|
| `textcnn_classifier.onnx` | ONNX 格式模型文件 |

### 5.2 模型信息

转换完成后，脚本会输出：

- **输入张量名称**：`text_input`, `filename_input`
- **输出张量名称**：`output`（Softmax 概率分布）
- **输出维度**：`[batch_size, num_classes]`（7 个类别）

---

## 6. 常见问题

### Q1：提示 `No module named 'tf2onnx'`

```bash
pip install tf2onnx
```

### Q2：转换时出现 `TypeError: not all arguments converted`

- 确保模型是 **双输入** 模型
- 检查 `input_signature` 是否正确匹配模型输入

### Q3：转换后 ONNX 模型无法加载

```python
import onnx
model = onnx.load("textcnn_classifier.onnx")
onnx.checker.check_model(model)  # 校验模型
```

常见原因：
- TensorFlow 版本与 tf2onnx 不兼容
- 尝试降低 `opset` 版本（如 `opset=12`）

### Q4：ONNX 模型输入顺序与预期不符

ONNX 会保持输入定义的顺序。在 `predict_onnx.py` 中使用：

```python
input_names = [inp.name for inp in session.get_inputs()]
# input_names 顺序: ['text_input', 'filename_input']
```

### Q5：模型包含自定义层导致转换失败

- TextCNN 模型使用的都是标准 Keras 层（Conv1D, GlobalMaxPooling1D 等），一般无问题
- 如有自定义层，需先注册或改用标准层

---

## 7. 转换后的验证步骤

### 7.1 使用 ONNX Runtime 验证

```python
import onnxruntime as ort
import numpy as np

# 加载模型
session = ort.InferenceSession("textcnn_classifier.onnx")

# 准备测试输入
text_input = np.random.randint(0, 1000, (1, 1000)).astype(np.int32)
filename_input = np.random.randint(0, 100, (1, 32)).astype(np.int32)

# 推理
outputs = session.run(
    ["output"], 
    {"text_input": text_input, "filename_input": filename_input}
)

print(f"输出形状: {outputs[0].shape}")  # (1, 7)
```

### 7.2 与 Keras 模型对比

```python
# Keras 推理
keras_output = keras_model.predict([text_input, filename_input])

# ONNX 推理
onnx_output = session.run(["output"], {...})[0]

# 对比差异
diff = np.abs(keras_output - onnx_output).max()
print(f"最大差异: {diff}")  # 通常 < 1e-5
```

---

## 8. 性能优化建议

| 优化方向 | 建议 |
|----------|------|
| **模型简化** | 转换时使用 `opset=13`（平衡性能与兼容性） |
| **推理加速** | 使用 ONNX Runtime 的 CPU/GPU 提供程序 |
| **内存优化** | 启用 `enable_cpu_mem_arena=True` |
| **量化** | 可进一步转换为 INT8 ONNX 模型 |

---

## 9. 完整工作流程

```bash
# 步骤1：训练 Keras 模型
python train.py

# 步骤2：转换为 ONNX
python convert_to_onnx.py

# 步骤3：使用 ONNX 模型推理
python predict_onnx.py 语文.pptx
```

---

## 10. 脚本代码速查

| 函数/变量 | 作用 |
|-----------|------|
| `MODEL_PATH` | 输入 Keras 模型路径 |
| `CONFIG_PATH` | 输入配置文件路径 |
| `OUTPUT_ONNX_PATH` | 输出 ONNX 模型路径 |
| `convert_model()` | 主转换函数 |
| `tf2onnx.convert.from_keras()` | Keras → ONNX 转换 API |
| `onnx.checker.check_model()` | ONNX 模型校验 |

---

## 11. 兼容性说明

| 组件 | 最低版本 | 推荐版本 |
|------|----------|----------|
| TensorFlow | 2.4.0 | 2.13+ |
| tf2onnx | 1.9.0 | 1.14+ |
| ONNX | 1.8.0 | 1.14+ |
| ONNX Runtime | 1.8.0 | 1.16+ |

---

## 12. 错误处理

| 错误信息 | 解决方案 |
|----------|----------|
| `FileNotFoundError: textcnn_optimized_classifier.keras` | 先运行 `train.py` 训练模型 |
| `ValueError: opset version not supported` | 降低 `opset` 参数（如 `opset=12`） |
| `TypeError: Failed to convert elements` | 检查输入张量数据类型是否为 `int32` |

---

**版本**：v1.0  
**配套训练脚本版本**：v5  
**最后更新**：2026-06-06