// cpp/file_processor.cpp
#include "file_processor.h"
#include <iostream>
#include <sstream>
#include <regex>
#include <chrono>
#include <cstring>
#include <algorithm>
#include <unordered_set>
#include <fstream>

#include "cppjieba/Jieba.hpp"
#include <onnxruntime_cxx_api.h>

// ========== 配置常量 - 根据模型输出调整 ==========
const int MODEL_VOCAB_SIZE = 20000;
const int MAX_SEQ_LENGTH = 1000;        // 模型要求: 1000
const int MAX_FILENAME_LENGTH = 32;     // 模型要求: 32
const int NUM_CLASSES = 7;              // 7个类别

// 停用词表
const std::unordered_set<std::string> STOPWORDS = {
    "的", "了", "是", "我", "你", "他", "她", "它", "我们", "你们", "他们",
    "这", "那", "有", "在", "不", "和", "与", "就", "都", "而", "及", "或",
    "一个", "这个", "那个", "那些", "这些", "这里", "那里", "然后", "因为",
    "所以", "但是", "如果", "虽然", "然而", "并且", "或者"
};

const std::unordered_set<std::string> MEANINGFUL_SINGLE_CHARS = {
    "圆", "力", "氧", "氢", "碳", "钠", "酸", "碱", "盐",
    "电", "光", "声", "热", "诗", "词", "歌", "曲", "数",
    "方", "程", "函", "角", "形", "体", "积"
};

// 全局单例处理器
class SharedProcessor {
private:
    static SharedProcessor* instance_;
    static std::mutex mutex_;
    
    std::unique_ptr<cppjieba::Jieba> jieba_;
    std::unique_ptr<Ort::Session> session_;
    std::unique_ptr<Ort::MemoryInfo> memory_info_;
    Ort::Env env_;
    
    std::regex url_pattern_;
    std::regex special_char_pattern_;
    std::regex whitespace_pattern_;
    
    bool initialized_;
    int vocab_size_;
    
    // 存储tokenizer的word_index（需要从Python导出的pickle加载）
    std::unordered_map<std::string, int> word_index_;
    
    SharedProcessor(const std::string& dict_path, const std::string& model_path)
        : env_(ORT_LOGGING_LEVEL_WARNING, "SharedProcessor")
        , url_pattern_(R"(https?://[^\s]+|www\.[^\s]+)")
        , special_char_pattern_(R"([^\u4e00-\u9fa5a-zA-Z0-9\s])")
        , whitespace_pattern_(R"(\s+)")
        , initialized_(false)
        , vocab_size_(MODEL_VOCAB_SIZE) {
        
        initialize(dict_path, model_path);
    }
    
    void loadWordIndex(const std::string& tokenizer_path) {
        // TODO: 从Python导出的pickle文件加载word_index
        // 目前使用简化版本，实际应该加载训练时的tokenizer
        std::cout << "[C++] 警告: 使用简化哈希函数，可能与训练时不匹配" << std::endl;
    }
    
    void initialize(const std::string& dict_path, const std::string& model_path) {
        try {
            // 1. 初始化jieba
            std::cout << "[C++] 加载jieba词典: " << dict_path << std::endl;
            jieba_ = std::make_unique<cppjieba::Jieba>(
                dict_path + "/jieba.dict.utf8",
                dict_path + "/hmm_model.utf8",
                dict_path + "/user.dict.utf8",
                dict_path + "/idf.utf8",
                dict_path + "/stop_words.utf8"
            );
            std::cout << "[C++] jieba加载完成" << std::endl;
            
            // 2. 加载ONNX模型
            std::cout << "[C++] 加载ONNX模型: " << model_path << std::endl;
            Ort::SessionOptions session_options;
            session_options.SetIntraOpNumThreads(2);
            session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
            
            session_ = std::make_unique<Ort::Session>(env_, model_path.c_str(), session_options);
            memory_info_ = std::make_unique<Ort::MemoryInfo>(
                Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault)
            );
            
            // 打印模型信息
            Ort::AllocatorWithDefaultOptions allocator;
            size_t num_inputs = session_->GetInputCount();
            std::cout << "[C++] 模型输入数量: " << num_inputs << std::endl;
            for (size_t i = 0; i < num_inputs; ++i) {
                auto name = session_->GetInputNameAllocated(i, allocator);
                auto type_info = session_->GetInputTypeInfo(i);
                auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
                auto shape = tensor_info.GetShape();
                std::cout << "[C++]   输入[" << i << "]: " << name.get() << " shape=[";
                for (size_t j = 0; j < shape.size(); ++j) {
                    if (j > 0) std::cout << ",";
                    if (shape[j] == -1) std::cout << "batch";
                    else std::cout << shape[j];
                }
                std::cout << "]" << std::endl;
            }
            
            initialized_ = true;
            std::cout << "[C++] 初始化完成" << std::endl;
        } catch (const std::exception& e) {
            std::cerr << "[C++] 初始化失败: " << e.what() << std::endl;
            initialized_ = false;
        }
    }
    
public:
    static SharedProcessor* GetInstance(const std::string& dict_path = "", 
                                        const std::string& model_path = "") {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!instance_) {
            if (dict_path.empty() || model_path.empty()) {
                throw std::runtime_error("首次调用必须提供路径");
            }
            instance_ = new SharedProcessor(dict_path, model_path);
        }
        return instance_;
    }
    
    static void DestroyInstance() {
        std::lock_guard<std::mutex> lock(mutex_);
        delete instance_;
        instance_ = nullptr;
    }
    
    bool isInitialized() const { return initialized_; }
    
    std::string cleanText(const std::string& text) {
        if (text.empty()) return "";
        
        std::string cleaned = std::regex_replace(text, url_pattern_, "");
        cleaned = std::regex_replace(cleaned, special_char_pattern_, " ");
        cleaned = std::regex_replace(cleaned, whitespace_pattern_, " ");
        
        size_t start = cleaned.find_first_not_of(" \t\n\r");
        size_t end = cleaned.find_last_not_of(" \t\n\r");
        if (start == std::string::npos) return "";
        
        return cleaned.substr(start, end - start + 1);
    }

    bool isValidUTF8(const std::string& s) {
        int bytes = 0;
        for (unsigned char c : s) {
            if (bytes == 0) {
                if ((c & 0x80) == 0) continue;
                if ((c & 0xE0) == 0xC0) bytes = 1;
                else if ((c & 0xF0) == 0xE0) bytes = 2;
                else if ((c & 0xF8) == 0xF0) bytes = 3;
                else return false;
            } else {
                if ((c & 0xC0) != 0x80) return false;
                bytes--;
            }
        }
        return bytes == 0;
    }
    
    std::string segmentText(const std::string& text) {
        if (text.empty()) return "";
        
        // 验证UTF-8
        if (!isValidUTF8(text)) {
            std::cerr << "[C++] 警告: 输入文本不是有效的UTF-8" << std::endl;
            return "";
        }

        std::vector<std::string> words;
        jieba_->Cut(text, words, true);
        
        std::vector<std::string> filtered;
        for (const auto& w : words) {
            if (w.empty()) continue;
            if (STOPWORDS.find(w) != STOPWORDS.end()) continue;
            if (w.length() == 1 && MEANINGFUL_SINGLE_CHARS.find(w) == MEANINGFUL_SINGLE_CHARS.end()) {
                continue;
            }
            filtered.push_back(w);
        }
        
        if (filtered.empty()) {
            filtered = words;
        }
        
        std::stringstream ss;
        for (size_t i = 0; i < filtered.size(); ++i) {
            if (i > 0) ss << " ";
            ss << filtered[i];
        }
        return ss.str();
    }
    
    // 使用与Python训练时相同的哈希方法
    int32_t wordToIndex(const std::string& word) {
        // 简化版哈希，实际应该加载训练时的word_index
        // 这里使用与Python tokenizer兼容的方法
        uint32_t hash = 2166136261u;  // FNV-1a offset basis
        for (char c : word) {
            hash ^= static_cast<unsigned char>(c);
            hash *= 16777619u;
        }
        // 映射到 [1, vocab_size-1]
        int idx = (hash % (vocab_size_ - 1)) + 1;
        return idx;
    }
    
    std::vector<int32_t> textToSequence(const std::string& text, int max_len) {
        if (text.empty()) return std::vector<int32_t>(max_len, 0);
        
        std::vector<std::string> words;
        std::stringstream ss(text);
        std::string word;
        while (ss >> word) {
            words.push_back(word);
        }
        
        std::vector<int32_t> seq;
        seq.reserve(max_len);
        
        for (const auto& w : words) {
            int idx = wordToIndex(w);
            // 确保索引在有效范围内 [1, vocab_size-1]
            if (idx >= vocab_size_) idx = 1;
            if (idx == 0) idx = 1;
            seq.push_back(idx);
            if ((int)seq.size() >= max_len) break;
        }
        
        // Padding到max_len
        seq.resize(max_len, 0);
        return seq;
    }
    
    std::vector<int32_t> filenameToSequence(const std::string& filename, int max_len) {
        // 简化版：从文件名中提取特征
        if (filename.empty()) return std::vector<int32_t>(max_len, 0);
        
        // 去除扩展名，只取文件名主体
        std::string name = filename;
        size_t dot_pos = name.rfind('.');
        if (dot_pos != std::string::npos) {
            name = name.substr(0, dot_pos);
        }
        
        // 分词
        std::vector<std::string> words;
        jieba_->Cut(name, words, true);
        
        std::vector<int32_t> seq;
        seq.reserve(max_len);
        
        for (const auto& w : words) {
            int idx = wordToIndex(w);
            if (idx >= vocab_size_) idx = 1;
            if (idx == 0) idx = 1;
            seq.push_back(idx);
            if ((int)seq.size() >= max_len) break;
        }
        
        seq.resize(max_len, 0);
        return seq;
    }
    
    std::pair<int, float> infer(const std::vector<int32_t>& text_seq,
                                 const std::vector<int32_t>& filename_seq) {
        if (!session_) {
            return std::make_pair(0, 0.0f);
        }
        
        try {
            std::vector<int64_t> text_shape = {1, (int64_t)text_seq.size()};
            std::vector<int64_t> filename_shape = {1, (int64_t)filename_seq.size()};
            
            // 创建输入tensor
            Ort::Value text_tensor = Ort::Value::CreateTensor<int32_t>(
                *memory_info_, 
                const_cast<int32_t*>(text_seq.data()), 
                text_seq.size(),
                text_shape.data(), 
                text_shape.size()
            );
            
            Ort::Value filename_tensor = Ort::Value::CreateTensor<int32_t>(
                *memory_info_, 
                const_cast<int32_t*>(filename_seq.data()), 
                filename_seq.size(),
                filename_shape.data(), 
                filename_shape.size()
            );
            
            // 使用模型实际的输入名称
            const char* input_names[] = {"text_input", "filename_input"};
            const char* output_names[] = {"output"};
            
            std::vector<Ort::Value> input_tensors;
            input_tensors.push_back(std::move(text_tensor));
            input_tensors.push_back(std::move(filename_tensor));
            
            auto output_tensors = session_->Run(
                Ort::RunOptions{nullptr},
                input_names, input_tensors.data(), input_tensors.size(),
                output_names, 1
            );
            
            float* output_data = output_tensors[0].GetTensorMutableData<float>();
            
            // 找到最大概率的类别
            int pred_class = 0;
            float max_prob = output_data[0];
            for (int i = 1; i < NUM_CLASSES; ++i) {
                if (output_data[i] > max_prob) {
                    max_prob = output_data[i];
                    pred_class = i;
                }
            }
            
            return std::make_pair(pred_class, max_prob);
        } catch (const std::exception& e) {
            std::cerr << "[C++] 推理失败: " << e.what() << std::endl;
            return std::make_pair(0, 0.0f);
        }
    }
    
    std::string processFile(const std::string& content, const std::string& filename = "") {
        if (!initialized_) {
            return "{\"error\":\"处理器未初始化\"}";
        }
        
        auto start = std::chrono::high_resolution_clock::now();
        
        std::string cleaned = cleanText(content);
        std::string segmented = segmentText(cleaned);
        auto seg_end = std::chrono::high_resolution_clock::now();
        
        std::vector<int32_t> text_seq = textToSequence(segmented, MAX_SEQ_LENGTH);
        std::vector<int32_t> filename_seq = filenameToSequence(filename, MAX_FILENAME_LENGTH);
        
        auto [pred_class, confidence] = infer(text_seq, filename_seq);
        auto infer_end = std::chrono::high_resolution_clock::now();
        
        auto clean_seg_ms = std::chrono::duration_cast<std::chrono::milliseconds>(seg_end - start).count();
        auto infer_ms = std::chrono::duration_cast<std::chrono::milliseconds>(infer_end - seg_end).count();
        
        // 类别映射（应与Python训练时一致）
        const char* categories[] = {"语文", "数学", "英语", "物理", "化学", "生物", "班会"};
        const char* pred_category = (pred_class >= 0 && pred_class < 7) ? categories[pred_class] : "未知";
        
        std::string result = "{";
        result += "\"predicted_class\":\"" + std::string(pred_category) + "\",";
        result += "\"confidence\":" + std::to_string(confidence) + ",";
        result += "\"segmented\":\"" + escapeJson(segmented) + "\",";
        result += "\"performance\":{";
        result += "\"clean_seg_ms\":" + std::to_string(clean_seg_ms) + ",";
        result += "\"inference_ms\":" + std::to_string(infer_ms);
        result += "}";
        result += "}";
        
        return result;
    }
    
    std::string segmentOnly(const std::string& text) {
        if (!initialized_) return "";
        std::string cleaned = cleanText(text);
        return segmentText(cleaned);
    }
    
private:
    // 修复 escapeJson 函数
    std::string escapeJson(const std::string& s) {
        std::string result;
        result.reserve(s.size() * 2);
        
        for (unsigned char c : s) {
            if (c == '"') {
                result += "\\\"";
            } else if (c == '\\') {
                result += "\\\\";
            } else if (c == '\n') {
                result += "\\n";
            } else if (c == '\r') {
                result += "\\r";
            } else if (c == '\t') {
                result += "\\t";
            } else if (c < 0x20) {
                // 控制字符
                char buf[7];
                snprintf(buf, sizeof(buf), "\\u%04x", c);
                result += buf;
            } else {
            // UTF-8 字符直接保留
            result += c;
        }
    }   
        return result;
    }
};

// 静态成员初始化
SharedProcessor* SharedProcessor::instance_ = nullptr;
std::mutex SharedProcessor::mutex_;

// C接口实现
extern "C" {

API int InitializeProcessor(const char* dict_path, const char* model_path) {
    try {
        SharedProcessor::GetInstance(dict_path, model_path);
        return 1;
    } catch (const std::exception& e) {
        std::cerr << "[C++] 初始化失败: " << e.what() << std::endl;
        return 0;
    }
}

API char* ProcessFileContent(const char* content) {
    if (!content) return nullptr;
    
    try {
        auto* processor = SharedProcessor::GetInstance();
        if (!processor || !processor->isInitialized()) {
            return nullptr;
        }
        
        std::string result = processor->processFile(content, "");
        char* out = new char[result.size() + 1];
        std::strcpy(out, result.c_str());
        return out;
    } catch (const std::exception& e) {
        std::cerr << "[C++] 处理失败: " << e.what() << std::endl;
        return nullptr;
    }
}

API char* ProcessFileWithName(const char* content, const char* filename) {
    if (!content) return nullptr;
    
    try {
        auto* processor = SharedProcessor::GetInstance();
        if (!processor || !processor->isInitialized()) {
            return nullptr;
        }
        
        std::string fname = filename ? filename : "";
        std::string result = processor->processFile(content, fname);
        char* out = new char[result.size() + 1];
        std::strcpy(out, result.c_str());
        return out;
    } catch (const std::exception& e) {
        std::cerr << "[C++] 处理失败: " << e.what() << std::endl;
        return nullptr;
    }
}

API char* ProcessMultipleFiles(const char** contents, int file_count) {
    if (!contents || file_count <= 0) return nullptr;
    
    try {
        auto* processor = SharedProcessor::GetInstance();
        if (!processor || !processor->isInitialized()) {
            return nullptr;
        }
        
        std::string result = "[";
        for (int i = 0; i < file_count; ++i) {
            if (i > 0) result += ",";
            if (contents[i]) {
                result += processor->processFile(contents[i], "");
            } else {
                result += "{\"error\":\"null content\"}";
            }
        }
        result += "]";
        
        char* out = new char[result.size() + 1];
        std::strcpy(out, result.c_str());
        return out;
    } catch (const std::exception& e) {
        std::cerr << "[C++] 批量处理失败: " << e.what() << std::endl;
        return nullptr;
    }
}

API char* SegmentOnly(const char* text) {
    if (!text) return nullptr;
    
    try {
        auto* processor = SharedProcessor::GetInstance();
        if (!processor || !processor->isInitialized()) {
            return nullptr;
        }
        
        std::string result = processor->segmentOnly(text);
        char* out = new char[result.size() + 1];
        std::strcpy(out, result.c_str());
        return out;
    } catch (const std::exception& e) {
        std::cerr << "[C++] 分词失败: " << e.what() << std::endl;
        return nullptr;
    }
}

API void FreeString(char* str) {
    delete[] str;
}

API void CleanupProcessor() {
    SharedProcessor::DestroyInstance();
}

API int IsProcessorInitialized() {
    try {
        auto* processor = SharedProcessor::GetInstance();
        return processor && processor->isInitialized() ? 1 : 0;
    } catch (...) {
        return 0;
    }
}

} // extern "C"