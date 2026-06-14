# test_cpp_direct.py
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import ctypes
import json
import os
from ctypes import c_char_p, c_int, POINTER

def test_cpp_direct():
    # 加载库
    lib_path = "./libFastProcessor.dylib"
    if not os.path.exists(lib_path):
        # 尝试从build目录复制
        import shutil
        build_lib = "../build/libFastProcessor.1.0.0.dylib"
        if os.path.exists(build_lib):
            shutil.copy(build_lib, lib_path)
            print(f"已复制库文件: {build_lib} -> {lib_path}")
        else:
            print(f"找不到库文件: {lib_path}")
            return
    
    lib = ctypes.CDLL(lib_path)
    
    # 设置函数签名
    lib.InitializeProcessor.argtypes = [c_char_p, c_char_p]
    lib.InitializeProcessor.restype = c_int
    
    lib.ProcessFileContent.argtypes = [c_char_p]
    lib.ProcessFileContent.restype = c_char_p
    
    lib.SegmentOnly.argtypes = [c_char_p]
    lib.SegmentOnly.restype = c_char_p
    
    lib.FreeString.argtypes = [c_char_p]
    lib.CleanupProcessor.argtypes = []
    
    # 初始化
    dict_path = b"./dict"
    model_path = b"./textcnn_classifier.onnx"
    
    print("=" * 60)
    print("C++ 处理器测试 (Python调用)")
    print("=" * 60)
    
    print("\n初始化...")
    if lib.InitializeProcessor(dict_path, model_path) == 0:
        print("初始化失败")
        return
    print("初始化成功")
    
    # 测试文本（Python字符串会自动编码为UTF-8）
    test_texts = [
        "这是一篇关于数学的文章，讨论二次函数和一元二次方程的解法。",
        "物理课上学习了牛顿第二定律，F=ma，这是经典力学的基础。",
        "英语学习中，词汇量很重要，每天背诵20个单词效果很好。",
        "化学实验：将氢气在氯气中燃烧，生成氯化氢气体。",
        "生物课上学习了细胞的结构，包括细胞膜、细胞质和细胞核。"
    ]
    
    print("\n分词测试:")
    print("-" * 40)
    test_str = "我爱北京天安门"
    result = lib.SegmentOnly(test_str.encode('utf-8'))
    if result:
        segmented = result.decode('utf-8')
        print(f"原文: {test_str}")
        print(f"分词: {segmented}")
        lib.FreeString(result)
    
    print("\n完整处理测试:")
    print("-" * 40)
    
    import time
    for i, text in enumerate(test_texts, 1):
        print(f"\n{i}. {text[:40]}...")
        
        start = time.time()
        result = lib.ProcessFileContent(text.encode('utf-8'))
        elapsed = (time.time() - start) * 1000
        
        if result:
            result_str = result.decode('utf-8')
            try:
                data = json.loads(result_str)
                print(f"   类别: {data.get('predicted_class', 'N/A')}")
                print(f"   置信度: {data.get('confidence', 0):.4f}")
                print(f"   耗时: {elapsed:.0f}ms")
                if data.get('segmented'):
                    print(f"   分词: {data['segmented'][:80]}...")
            except json.JSONDecodeError:
                print(f"   结果: {result_str[:100]}")
            lib.FreeString(result)
    
    # 清理
    lib.CleanupProcessor()
    print("\n测试完成")

if __name__ == "__main__":
    test_cpp_direct()