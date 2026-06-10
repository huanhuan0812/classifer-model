# 文件分类移动工具 - 使用文档

## 📖 目录

1. [简介](#简介)
2. [安装指南](#安装指南)
3. [快速开始](#快速开始)
4. [配置文件详解](#配置文件详解)
5. [使用教程](#使用教程)
6. [高级功能](#高级功能)
7. [常见问题](#常见问题)
8. [故障排除](#故障排除)

---

## 简介

### 工具概述

文件分类移动工具是一个基于AI的文件自动分类系统，能够智能识别PPTX、PPT、DOCX文件的内容，并自动将其移动到对应的学科类别文件夹中。

### 主要功能

- 🎯 **智能分类**：支持7个类别（语文、数学、英语、物理、化学、生物、班会）
- 📊 **置信度控制**：可设置阈值，低置信度文件单独处理
- 📁 **自动整理**：按类别创建文件夹并移动文件
- 🔍 **预览模式**：先查看分类结果，确认后再操作
- 📝 **详细日志**：记录所有操作和统计信息
- ⚙️ **灵活配置**：所有参数通过配置文件管理

### 支持的文件格式

| 格式 | 说明 | 文本提取 |
|------|------|----------|
| .pptx | PowerPoint 2007+ | ✅ 完整支持 |
| .ppt | PowerPoint 97-2003 | ✅ 基础支持 |
| .docx | Word 2007+ | ✅ 完整支持 |

---

## 安装指南

### 系统要求

- **操作系统**：Windows 10/11, Linux, macOS
- **Python版本**：3.8 或更高
- **内存要求**：建议 4GB 以上
- **硬盘空间**：至少 500MB（包含模型文件）

### 安装步骤

#### 步骤1：安装Python

访问 [python.org](https://python.org) 下载并安装 Python 3.8+。

验证安装：
```bash
python --version
```

#### 步骤2：下载项目文件

将以下文件放在同一目录下：
- `predict_onnx.py` - 分类器核心文件
- `file_mover.py` - 文件移动主程序
- `file_classifier_config.yaml` - 配置文件
- 模型文件（.onnx, .pkl文件）

#### 步骤3：安装依赖包

打开终端（命令提示符），进入项目目录：

```bash
# Windows
cd C:\your_project_path

# Linux/Mac
cd /your_project_path
```

安装依赖：
```bash
pip install -r requirements.txt
```

如果不想创建requirements.txt，可以直接安装：
```bash
pip install onnxruntime python-pptx jieba python-docx pyyaml
```

#### 步骤4：验证安装

运行预览模式测试：
```bash
python file_mover.py --preview
```

如果看到帮助信息，说明安装成功。

---

## 快速开始

### 最简单的使用方式

1. **准备文件**
   ```bash
   # 创建输入文件夹
   mkdir input
   
   # 将PPTX/DOCX文件放入 input 文件夹
   ```

2. **编辑配置文件**
   
   打开 `file_classifier_config.yaml`，设置源目录：
   ```yaml
   paths:
     source_dir: "./input"      # 你的文件所在文件夹
     target_base_dir: "./output" # 分类后的文件存放位置
   ```

3. **预览模式（推荐）**
   ```bash
   python file_mover.py --preview
   ```
   
   查看每个文件的分类结果，确认是否准确。

4. **执行移动**
   ```bash
   python file_mover.py
   ```
   
   输入 `y` 确认，程序将自动分类并移动文件。

### 示例演示

假设你有以下文件：
```
input/
├── 三角函数讲解.pptx
├── 文言文翻译.docx
├── 英语语法练习.pptx
└── 细胞分裂.ppt
```

运行后，文件将被组织为：
```
output/
├── 数学/
│   └── 三角函数讲解.pptx
├── 语文/
│   └── 文言文翻译.docx
├── 英语/
│   └── 英语语法练习.pptx
└── 生物/
    └── 细胞分裂.ppt
```

---

## 配置文件详解

### 完整配置文件示例

```yaml
# 路径配置
paths:
  source_dir: "./input"           # 源文件夹路径
  target_base_dir: "./output"      # 目标基础文件夹路径

# 预测配置
prediction:
  threshold: 0.7                   # 置信度阈值（0-1）
  verbose: true                    # 是否显示详细信息
  supported_formats:               # 支持的文件格式
    - ".pptx"
    - ".ppt"
    - ".docx"

# 文件处理配置
file_handling:
  move_files: true                 # true=移动, false=复制
  create_category_dirs: true       # 自动创建类别目录
  overwrite: false                 # 覆盖已存在文件
  keep_original_name: true         # 保持原文件名
  conflict_resolution: "rename"    # 冲突处理: rename/skip

# 分类配置
categories:
  target_categories: []            # 空=所有类别
  exclude_categories: []           # 排除的类别
  low_confidence_action: "move_to_uncertain"  # 低置信度处理
  uncertain_folder_name: "_uncertain_low_confidence"  # 不确定文件夹名

# 日志配置
logging:
  enabled: true                    # 启用日志
  log_file: "file_mover.log"       # 日志文件
  log_level: "INFO"                # DEBUG/INFO/WARNING/ERROR
  save_moved_list: true            # 保存移动记录
  moved_list_file: "moved_files.csv"

# 高级选项
advanced:
  max_file_size_mb: 100            # 文件大小限制(MB)
  skip_empty_text: false           # 跳过无文本文件
  recursive_scan: true             # 递归扫描子文件夹
```

### 配置参数说明

#### 路径配置 (paths)

| 参数 | 类型 | 说明 | 示例 |
|------|------|------|------|
| source_dir | string | 源文件夹路径 | `"./my_files"` |
| target_base_dir | string | 目标文件夹路径 | `"./classified"` |

#### 预测配置 (prediction)

| 参数 | 类型 | 说明 | 默认值 |
|------|------|------|--------|
| threshold | float | 置信度阈值，0-1之间 | 0.7 |
| verbose | boolean | 是否显示详细输出 | true |
| supported_formats | list | 支持的文件格式列表 | [".pptx", ".ppt", ".docx"] |

**阈值建议**：
- `0.9-1.0`：非常严格，只移动高确信度的文件
- `0.7-0.9`：平衡模式，推荐使用
- `0.5-0.7`：宽松模式，会移动更多文件
- `<0.5`：不推荐，可能误分类

#### 文件处理 (file_handling)

| 参数 | 类型 | 说明 | 选项 |
|------|------|------|------|
| move_files | boolean | 移动还是复制 | true(移动)/false(复制) |
| overwrite | boolean | 是否覆盖同名文件 | true/false |
| keep_original_name | boolean | 是否保持原文件名 | true/false |
| conflict_resolution | string | 文件名冲突处理 | "rename"/"skip" |

#### 分类配置 (categories)

| 参数 | 类型 | 说明 |
|------|------|------|
| target_categories | list | 只处理这些类别，空列表表示全部 |
| exclude_categories | list | 排除这些类别 |
| low_confidence_action | string | "move_to_uncertain" 或 "ignore" |
| uncertain_folder_name | string | 低置信度文件存放文件夹名 |

**类别名称**：
- 语文、数学、英语、物理、化学、生物、班会

#### 日志配置 (logging)

| 参数 | 类型 | 说明 |
|------|------|------|
| enabled | boolean | 是否启用日志 |
| log_file | string | 日志文件路径 |
| log_level | string | 日志级别 |
| save_moved_list | boolean | 是否保存CSV记录 |

#### 高级选项 (advanced)

| 参数 | 类型 | 说明 |
|------|------|------|
| max_file_size_mb | int | 最大文件大小(MB)，0=无限制 |
| skip_empty_text | boolean | 跳过无法提取文本的文件 |
| recursive_scan | boolean | 是否递归扫描子文件夹 |

---

## 使用教程

### 基础用法

#### 1. 命令行参数

```bash
# 显示帮助
python file_mover.py -h

# 使用默认配置运行
python file_mover.py

# 预览模式（不实际移动）
python file_mover.py --preview

# 指定配置文件
python file_mover.py --config my_config.yaml

# 覆盖源目录
python file_mover.py --source ./my_files

# 覆盖目标目录
python file_mover.py --target ./my_output

# 覆盖置信度阈值
python file_mover.py --threshold 0.8

# 组合使用
python file_mover.py --source ./data --target ./result --threshold 0.85 --preview
```

#### 2. 交互式运行

```bash
# 启动程序
python file_mover.py

# 程序会显示配置信息并要求确认
==================================================
文件分类移动程序
==================================================
源目录: ./input
目标目录: ./output
置信度阈值: 70%
操作模式: 移动
==================================================

找到 10 个文件，开始处理...

是否继续? (y/N): y   # 输入y确认

# 程序开始处理并显示进度
```

### 实际应用场景

#### 场景1：教师整理课件

**需求**：将下载的各种课件文件按学科分类

**配置**：
```yaml
paths:
  source_dir: "./下载的课件"
  target_base_dir: "./学科分类"

prediction:
  threshold: 0.7
```

**操作**：
```bash
python file_mover.py
```

#### 场景2：批量处理大文件

**需求**：处理超过100MB的文件，只移动高置信度的文件

**配置**：
```yaml
prediction:
  threshold: 0.85

advanced:
  max_file_size_mb: 200
```

**操作**：
```bash
python file_mover.py --threshold 0.85
```

#### 场景3：只处理某些学科

**需求**：只需要数学和英语的文件

**配置**：
```yaml
categories:
  target_categories: ["数学", "英语"]
  low_confidence_action: "ignore"
```

#### 场景4：测试新文件

**需求**：先用预览模式检查分类效果

**操作**：
```bash
# 先预览
python file_mover.py --preview

# 确认无误后再执行
python file_mover.py
```

---

## 高级功能

### 1. 自定义类别目录

通过配置文件可以自定义不确定文件的存放位置：

```yaml
categories:
  low_confidence_action: "move_to_uncertain"
  uncertain_folder_name: "需要人工审核"
```

### 2. 批量处理子文件夹

启用递归扫描，处理所有子文件夹中的文件：

```yaml
advanced:
  recursive_scan: true
```

### 3. 日志和记录

启用详细日志和CSV记录：

```yaml
logging:
  enabled: true
  log_level: "DEBUG"  # 详细调试信息
  save_moved_list: true
  moved_list_file: "分类记录.csv"
```

生成的CSV文件包含：
- 源文件路径
- 目标文件路径
- 分类类别
- 置信度
- 处理时间等

### 4. 文件冲突处理策略

**重命名模式**：
```yaml
file_handling:
  overwrite: false
  conflict_resolution: "rename"
```
同名文件自动添加数字后缀（如 `file_1.pptx`）

**跳过模式**：
```yaml
file_handling:
  conflict_resolution: "skip"
```
跳过已存在的文件，记录到日志

### 5. 命令行快速配置

不需要修改配置文件，直接通过命令行参数覆盖：

```bash
# 只处理特定格式
python file_mover.py --source ./mixed_files

# 高阈值严格模式
python file_mover.py --threshold 0.9

# 复制而非移动（保留原文件）
# 在配置文件中设置 move_files: false
```

### 6. 批量处理脚本

创建批处理文件 `batch_process.bat` (Windows)：

```batch
@echo off
echo 开始处理文件...
python file_mover.py --source ./folder1 --target ./result1
python file_mover.py --source ./folder2 --target ./result2
echo 处理完成！
pause
```

或 Shell 脚本 `batch_process.sh` (Linux/Mac)：

```bash
#!/bin/bash
echo "开始处理文件..."
python file_mover.py --source ./folder1 --target ./result1
python file_mover.py --source ./folder2 --target ./result2
echo "处理完成！"
```

---

## 常见问题

### Q1: 程序运行很慢怎么办？

**A**: 
- 减少同时处理的文件数量
- 降低日志级别（改为 WARNING）
- 增加内存或使用更快的CPU
- 分批处理文件

### Q2: 有些文件没有被分类？

**A**: 可能原因：
1. 置信度低于阈值 - 检查 `_uncertain_low_confidence` 文件夹
2. 文件无法提取文本 - 检查文件是否损坏
3. 文件格式不支持 - 确认是 .pptx/.ppt/.docx

### Q3: 分类结果不准确怎么办？

**A**:
- 提高阈值（如0.85）只保留高置信度结果
- 检查文件是否包含足够的文字内容
- 文件名是否有意义（分类器同时使用文件名和内容）
- 对于简短或空文件，分类效果会受影响

### Q4: 如何批量测试分类效果？

**A**: 使用预览模式：
```bash
python file_mover.py --preview
```
会显示所有文件的分类结果而不实际移动。

### Q5: 可以同时处理多个文件夹吗？

**A**: 
可以，有两种方式：
1. 设置 `recursive_scan: true` 扫描子文件夹
2. 编写批处理脚本依次处理多个文件夹

### Q6: 文件被移动后找不到了？

**A**:
- 检查目标文件夹 `target_base_dir`
- 查看日志文件 `file_mover.log`
- 查看CSV记录 `moved_files.csv`

### Q7: 如何恢复被移动的文件？

**A**:
- 使用CSV记录文件中的路径信息
- 手动从目标文件夹移回
- 或使用复制模式（`move_files: false`）保留原文件

### Q8: 支持哪些语言的文件？

**A**: 
主要支持中文文件，对英文也有一定识别能力。文件名和内容中的中英文都会被使用。

---

## 故障排除

### 错误：ModuleNotFoundError

**错误信息**：
```
ModuleNotFoundError: No module named 'onnxruntime'
```

**解决方法**：
```bash
pip install onnxruntime python-pptx jieba python-docx pyyaml
```

### 错误：文件不存在

**错误信息**：
```
FileNotFoundError: 文件不存在: textcnn_classifier.onnx
```

**解决方法**：
- 确保所有模型文件（.onnx, .pkl）都在当前目录
- 检查文件名是否正确
- 检查文件权限

### 错误：ONNX模型加载失败

**错误信息**：
```
❌ 模型加载失败: ... 
```

**解决方法**：
1. 检查ONNX文件是否完整
2. 重新下载模型文件
3. 更新onnxruntime：`pip install --upgrade onnxruntime`

### 错误：内存不足

**错误信息**：
```
MemoryError
```

**解决方法**：
1. 减少批处理文件数量
2. 设置文件大小限制：
   ```yaml
   advanced:
     max_file_size_mb: 50
   ```
3. 分批处理文件

### 错误：权限被拒绝

**错误信息**：
```
PermissionError: [Errno 13] Permission denied
```

**解决方法**：
- 关闭正在使用的文件
- 以管理员身份运行（Windows）
- 检查文件夹写入权限

### 日志文件太大

**解决方法**：
```yaml
logging:
  log_level: "WARNING"  # 降低日志级别
  log_file: "file_mover.log"
```

定期清理日志文件：
```bash
# 删除旧日志
rm file_mover.log

# 或清空日志
> file_mover.log
```

### 程序卡住不动

**解决方法**：
1. 按 `Ctrl+C` 中断程序
2. 检查是否有超大文件
3. 减少文件数量后重试
4. 查看任务管理器确认是否在运行

### 获取更多帮助

- **查看日志**：检查 `file_mover.log` 文件
- **运行诊断**：使用预览模式测试
- **联系支持**：提供日志文件和配置文件

---

## 附录

### 文件结构

```
项目文件夹/
├── predict_onnx.py              # 分类器核心
├── file_mover.py                # 移动程序
├── file_classifier_config.yaml  # 配置文件
├── textcnn_classifier.onnx      # ONNX模型
├── text_tokenizer_none.pkl      # 文本分词器
├── filename_tokenizer_none.pkl  # 文件名分词器
├── categories.pkl               # 类别映射
├── config_optimized.pkl         # 配置
├── input/                       # 源文件目录
├── output/                      # 输出目录
├── file_mover.log              # 运行日志
└── moved_files.csv             # 移动记录
```

### 配置文件模板

提供了多个预设模板供参考：

**严格模式配置**（高精度）：
```yaml
prediction:
  threshold: 0.9
advanced:
  max_file_size_mb: 50
categories:
  low_confidence_action: "ignore"
```

**快速模式配置**（高效率）：
```yaml
prediction:
  threshold: 0.6
  verbose: false
logging:
  log_level: "WARNING"
```

**安全模式配置**（保留原文件）：
```yaml
file_handling:
  move_files: false  # 复制而非移动
  overwrite: false
  conflict_resolution: "rename"
```

### 更新日志

**v1.0.0** (2024-01)
- 初始版本发布
- 支持PPTX、DOCX文件分类
- 置信度阈值控制
- 配置文件管理

---

## 技术支持

如有问题，请提供以下信息：
1. 操作系统和Python版本
2. 错误截图或日志内容
3. 配置文件内容
4. 文件样例（如可行）

---

**祝使用愉快！** 🎉