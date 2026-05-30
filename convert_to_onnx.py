#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
将TextCNN Keras模型转换为ONNX格式
注意：由于模型有双输入（文本序列和文件名序列），需要使用特殊处理
"""

import pickle
import numpy as np
import tf2onnx
import tensorflow as tf

# 配置路径
MODEL_PATH = "textcnn_optimized_classifier.keras"
CONFIG_PATH = "config_optimized.pkl"
OUTPUT_ONNX_PATH = "textcnn_classifier.onnx"

def convert_model():
    print("正在加载Keras模型...")
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    
    print("加载配置...")
    with open(CONFIG_PATH, 'rb') as f:
        config = pickle.load(f)
    
    # 获取输入形状
    max_seq_len = config['max_sequence_length']
    max_filename_len = config['max_filename_length']
    
    print(f"输入形状: text_input=(None, {max_seq_len}), filename_input=(None, {max_filename_len})")
    
    # 指定输入签名
    text_input = tf.TensorSpec(shape=[None, max_seq_len], dtype=tf.int32, name='text_input')
    filename_input = tf.TensorSpec(shape=[None, max_filename_len], dtype=tf.int32, name='filename_input')
    
    print("正在转换为ONNX格式...")
    
    # 转换为ONNX
    model_proto, _ = tf2onnx.convert.from_keras(
        model,
        input_signature=[text_input, filename_input],
        opset=13,  # 使用opset 13
        output_path=OUTPUT_ONNX_PATH
    )
    
    print(f"✅ 转换成功！模型已保存到: {OUTPUT_ONNX_PATH}")
    
    # 验证模型
    print("\n验证ONNX模型...")
    import onnx
    onnx_model = onnx.load(OUTPUT_ONNX_PATH)
    onnx.checker.check_model(onnx_model)
    print("✅ ONNX模型验证通过！")
    
    # 打印模型信息
    print(f"\n模型输入:")
    for inp in onnx_model.graph.input:
        print(f"  - {inp.name}: {[d.dim_value for d in inp.type.tensor_type.shape.dim]}")
    
    print(f"\n模型输出:")
    for out in onnx_model.graph.output:
        print(f"  - {out.name}: {[d.dim_value for d in out.type.tensor_type.shape.dim]}")
    
    return OUTPUT_ONNX_PATH

if __name__ == "__main__":
    convert_model()
