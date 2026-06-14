// cpp/test_processor.cpp
#include <iostream>
#include <string>
#include <vector>
#include <chrono>
#include <cstring>

// 声明C接口（与file_processor.h保持一致）
extern "C" {
    int InitializeProcessor(const char* dict_path, const char* model_path);
    char* ProcessFileContent(const char* content);
    char* ProcessMultipleFiles(const char** contents, int file_count);
    char* SegmentOnly(const char* text);
    void FreeString(char* str);
    void CleanupProcessor();
    int IsProcessorInitialized();
}

int main() {
    std::cout << "========================================" << std::endl;
    std::cout << "文件处理器测试程序" << std::endl;
    std::cout << "========================================" << std::endl;
    
    // 获取当前工作目录下的dict路径
    std::string dict_path = "./dict";
    std::string model_path = "./textcnn_classifier.onnx";
    
    std::cout << "\n1. 初始化处理器..." << std::endl;
    std::cout << "   词典路径: " << dict_path << std::endl;
    std::cout << "   模型路径: " << model_path << std::endl;
    
    int ret = InitializeProcessor(dict_path.c_str(), model_path.c_str());
    if (ret == 0) {
        std::cerr << "❌ 初始化失败" << std::endl;
        return 1;
    }
    std::cout << "✅ 初始化成功" << std::endl;
    
    // 测试文本
    std::vector<std::string> test_texts = {
        "这是一篇关于数学的文章，讨论二次函数和一元二次方程的解法。",
        "物理课上学习了牛顿第二定律，F=ma，这是经典力学的基础。",
        "英语学习中，词汇量很重要，每天背诵20个单词效果很好。",
        "化学实验：将氢气在氯气中燃烧，生成氯化氢气体。",
        "生物课上学习了细胞的结构，包括细胞膜、细胞质和细胞核。"
    };
    
    // 2. 单文件处理测试
    std::cout << "\n2. 单文件处理测试..." << std::endl;
    std::cout << "----------------------------------------" << std::endl;
    
    for (size_t i = 0; i < test_texts.size(); ++i) {
        const auto& text = test_texts[i];
        std::cout << "\n文本 " << (i+1) << ": " << text.substr(0, 50) << "..." << std::endl;
        
        auto start = std::chrono::high_resolution_clock::now();
        char* result_ptr = ProcessFileContent(text.c_str());
        auto end = std::chrono::high_resolution_clock::now();
        auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
        
        if (result_ptr) {
            std::string result(result_ptr);
            FreeString(result_ptr);
            std::cout << "   耗时: " << ms << "ms" << std::endl;
            std::cout << "   结果预览: " << result.substr(0, 200) << "..." << std::endl;
        } else {
            std::cout << "   ❌ 处理失败" << std::endl;
        }
    }
    
    // 3. 批量处理测试
    std::cout << "\n3. 批量处理测试..." << std::endl;
    std::cout << "----------------------------------------" << std::endl;
    
    std::vector<const char*> c_strings;
    for (const auto& text : test_texts) {
        c_strings.push_back(text.c_str());
    }
    
    auto start = std::chrono::high_resolution_clock::now();
    char* batch_result = ProcessMultipleFiles(c_strings.data(), (int)c_strings.size());
    auto end = std::chrono::high_resolution_clock::now();
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    
    if (batch_result) {
        std::cout << "✅ 批量处理成功" << std::endl;
        std::cout << "   文件数: " << test_texts.size() << std::endl;
        std::cout << "   总耗时: " << ms << "ms" << std::endl;
        std::cout << "   平均耗时: " << ms / test_texts.size() << "ms/文件" << std::endl;
        
        // 解析结果（简单显示）
        std::string result_str(batch_result);
        if (result_str.length() > 200) {
            std::cout << "   结果预览: " << result_str.substr(0, 200) << "..." << std::endl;
        } else {
            std::cout << "   结果: " << result_str << std::endl;
        }
        
        FreeString(batch_result);
    } else {
        std::cout << "❌ 批量处理失败" << std::endl;
    }
    
    // 4. 仅分词测试
    std::cout << "\n4. 仅分词测试..." << std::endl;
    std::cout << "----------------------------------------" << std::endl;
    
    std::string test_sentence = "我爱北京天安门，天安门上太阳升";
    char* seg_result = SegmentOnly(test_sentence.c_str());
    if (seg_result) {
        std::cout << "原文: " << test_sentence << std::endl;
        std::cout << "分词: " << seg_result << std::endl;
        FreeString(seg_result);
    } else {
        std::cout << "❌ 分词失败" << std::endl;
    }
    
    // 5. 检查处理器状态
    std::cout << "\n5. 处理器状态..." << std::endl;
    std::cout << "----------------------------------------" << std::endl;
    int status = IsProcessorInitialized();
    std::cout << "处理器状态: " << (status ? "已初始化" : "未初始化") << std::endl;
    
    // 6. 清理资源
    std::cout << "\n6. 清理资源..." << std::endl;
    CleanupProcessor();
    std::cout << "✅ 清理完成" << std::endl;
    
    std::cout << "\n========================================" << std::endl;
    std::cout << "测试完成" << std::endl;
    std::cout << "========================================" << std::endl;
    
    return 0;
}