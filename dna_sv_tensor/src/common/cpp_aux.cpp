#include "cpp_aux.hpp"

#include <sys/stat.h>
#include <time.h>
#include <unistd.h>
#include <cstdarg>

using namespace std;

FILE* safe_fopen(const char* path, const char* mode)
{
    FILE* stream = 0;
    if (strcmp(path, "-") == 0) return strstr(mode, "r") ? stdin : stdout;

    if ((stream = fopen(path, mode)) == 0) {
        const char* y = strerror(errno);
        fprintf(stderr, "fail to open file '%s' with mode '%s': %s\n", path, mode, y);
        abort();
    }
    return stream;
}

size_t safe_fwrite(const void* buf, size_t size, size_t nmemb, FILE* stream)
{
    size_t ret = fwrite(buf, size, nmemb, stream);
    if (ret != nmemb) {
        fprintf(stderr, "%s", strerror(errno));
        abort();
    }
    return ret;
}

void create_directory(const char* path)
{
    if ((access(path, F_OK) != 0)
        &&
        (mkdir(path, S_IRWXU) != 0)) {
        fprintf(stderr, "Failed to create directory %s: %s\n", path, strerror(errno));
        abort();
    }
}

void
split_line(const std::string& line, const char* delimiter, std::vector<std::string>& tokens)
{
    // skip delimiter at the beginning
    string::size_type last_pos = line.find_first_not_of(delimiter, 0);
    // find first "non-delimiter"
    string::size_type pos = line.find_first_of(delimiter, last_pos);

    while (string::npos != pos || string::npos != last_pos) {
        // found a token, add it to the vector
        tokens.push_back(line.substr(last_pos, pos - last_pos));
        // skip delimiter, note the "not_of"
        last_pos = line.find_first_not_of(delimiter, pos);
        // find next "non-delimiter"
        pos = line.find_first_of(delimiter, last_pos);
    }
}

int safe_fclose(FILE* stream)
{
    int ret = fclose(stream);
    if (ret != 0) {
        fprintf(stderr, "%s\n", strerror(errno));
        abort();
    }
    return ret;
}

void hbn_exception(const char* expr, HBN_LOG_PARAMS_GENERIC, const char* fmt, ...)
{
    fprintf(stderr, "Assertion Failed At '%s:%s:%d'\n", file, func, line);
    fprintf(stderr, "\tExpression: '%s'\n", expr);
    if (!fmt) return;
    fprintf(stderr, "Context Information:\n");
    va_list ap;
    va_start(ap, fmt);
    vfprintf(stderr, fmt, ap);
    va_end(ap);
    fprintf(stderr, "\n");
    abort();
}

uint8_t nst_nt4_table[256] = {
	4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4, 
	4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4, 
	4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,  4, 5 /*'-'*/, 4, 4,
	4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4, 
	4, 0, 4, 1,  4, 4, 4, 2,  4, 4, 4, 4,  4, 4, 4, 4, 
	4, 4, 4, 4,  3, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4, 
	4, 0, 4, 1,  4, 4, 4, 2,  4, 4, 4, 4,  4, 4, 4, 4, 
	4, 4, 4, 4,  3, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4, 
	4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4, 
	4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4, 
	4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4, 
	4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4, 
	4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4, 
	4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4, 
	4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4, 
	4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4
};

uint8_t nst_nt16_table[256] = {
    16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, //15
	16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, // 31
	16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 15, 16, 16, // 47
	16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, // 63
	16, 0, 10, 1, 11, 16, 16, 2, 12, 16, 16, 7, 16, 6, 14, 16,
	16, 16, 4, 9, 3, 16, 13, 8, 16, 5, 16, 16, 16, 16, 16, 16,
	16, 0, 10, 1, 11, 16, 16, 2, 12, 16, 16, 7, 16, 6, 14, 16,
	16, 16, 4, 9, 3, 16, 13, 8, 16, 5, 16, 16, 16, 16, 16, 16,
	16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16,
	16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16,
	16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16,
	16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16,
	16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16,
	16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16,
	16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16,
	16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16
};