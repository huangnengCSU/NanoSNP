#ifndef __CPP_AUX_HPP
#define __CPP_AUX_HPP

#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include <string>
#include <vector>

#define HBN_LOG_ARGS_DEFAULT    __FILE__, __FUNCTION__, __LINE__
#define HBN_LOG_ARGS_GENERIC    file, func, line
#define HBN_LOG_PARAMS_GENERIC  const char* file, const char* func, const int line

void
hbn_exception(const char* expr, HBN_LOG_PARAMS_GENERIC, const char* fmt, ...);

#define __hbn_assert(expr, ...) \
    do { \
        if (!(expr)) { \
            hbn_exception(#expr, __VA_ARGS__, NULL); \
            abort(); \
        } \
    } while(0)

#define hbn_assert(expr, args...) __hbn_assert(expr, HBN_LOG_ARGS_DEFAULT, ##args)

FILE* safe_fopen(const char* path, const char* mode);

int safe_fclose(FILE* stream);

size_t safe_fwrite(const void* buf, size_t size, size_t nmemb, FILE* stream);

void create_directory(const char* path);

void
split_line(const std::string& line, const char* delimiter, std::vector<std::string>& tokens);

extern uint8_t nst_nt16_table[];
extern uint8_t nst_nt4_table[];

#endif // __CPP_AUX_HPP