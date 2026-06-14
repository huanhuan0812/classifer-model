// cpp/file_processor.h
#ifndef FILE_PROCESSOR_H
#define FILE_PROCESSOR_H

#ifdef _WIN32
    #ifdef EXPORT_DLL
        #define API __declspec(dllexport)
    #else
        #define API __declspec(dllimport)
    #endif
#else
    #define API __attribute__((visibility("default")))
#endif

extern "C" {
    API int InitializeProcessor(const char* dict_path, const char* model_path);
    API char* ProcessFileContent(const char* content);
    API char* ProcessFileWithName(const char* content, const char* filename);
    API char* ProcessMultipleFiles(const char** contents, int file_count);
    API char* SegmentOnly(const char* text);
    API void FreeString(char* str);
    API void CleanupProcessor();
    API int IsProcessorInitialized();
}

#endif