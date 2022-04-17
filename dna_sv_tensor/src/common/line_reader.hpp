#ifndef __LINE_READER_HPP
#define __LINE_READER_HPP

#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include <string>

class FileReader
{
public:
    FileReader(const char* path);
    ~FileReader();

    int get_char();
    void unget_char();

private:
const char*     m_path;
FILE*           m_file;
char*           m_buffer;
int             m_buffer_idx;
int             m_buffer_size;
int             m_max_buffer_size;
int             m_last_read_char;
int             m_unget_char;
};

class LineReader
{
public:
    LineReader(const char* path);
    ~LineReader();

    bool getline(std::string& line);
    void ungetline();

private:
    FileReader* m_reader;
    std::string m_line;
    bool        m_ungetline;
};

#endif // __LINE_READER_HPP