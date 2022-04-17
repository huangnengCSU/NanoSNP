#include "line_reader.hpp"

FileReader::FileReader(const char* path)
{
    m_path = path;
    if (strcmp(path, "-") == 0) {
        m_file = stdin;
    } else {
        if ((m_file = fopen(path, "r")) == 0) {
            const char* y = strerror(errno);
            fprintf(stderr, "Failed to open file '%s' for reading: %s\n", path, y);
            abort();
        }
    }

    m_max_buffer_size = 1 * 1024 * 1024;
    m_buffer = (char*)malloc(m_max_buffer_size * sizeof(char));
    m_buffer_idx = 0;
    m_buffer_size = 0;
    
    m_last_read_char = -1;
    m_unget_char = 0;
}

FileReader::~FileReader()
{
    int ret = fclose(m_file);
    if (ret != 0) {
        const char* y = strerror(errno);
        fprintf(stderr, "Fail to close file '%s': %s\n", m_path, y);
        abort();
    }
    if (m_buffer) free(m_buffer);
}

void FileReader::unget_char()
{
    m_unget_char = 1;
    if (m_last_read_char == EOF) {
        fprintf(stderr, "invalid last_read_char: %d\n", m_last_read_char);
        abort();
    }
}

int FileReader::get_char()
{
    if (m_unget_char) {
        if (m_last_read_char == EOF) {
            fprintf(stderr, "invalid last_read_char: %d\n", m_last_read_char);
            abort();
        }
        m_unget_char = 0;
        return m_last_read_char;
    }

    if (m_buffer_idx < m_buffer_size) {
        m_last_read_char = m_buffer[m_buffer_idx];
        ++m_buffer_idx;
        return m_last_read_char;
    }

    m_buffer_idx = 0;
    m_buffer_size = fread(m_buffer, sizeof(char), m_max_buffer_size, m_file);
    if (m_buffer_size < m_max_buffer_size) {
        if (ferror(m_file)) {
            const char* y = strerror(errno);
            fprintf(stderr, "File '%s' read error: %s\n", m_path, y);
            abort();
        }
    }
    if (m_buffer_size == 0) return EOF;

    m_last_read_char = m_buffer[m_buffer_idx];
    m_last_read_char = m_buffer[m_buffer_idx];
    ++m_buffer_idx;
    return m_last_read_char;
}

LineReader::LineReader(const char* path)
{
    m_reader = new FileReader(path);
    m_ungetline = false;
}

LineReader::~LineReader()
{
    delete m_reader;
}

void LineReader::ungetline()
{
    m_ungetline = true;
}

bool LineReader::getline(std::string& line)
{
    if (m_ungetline) {
        m_ungetline = false;
        line = m_line;
        return true;
    }

    line.clear();
    while (1) {
        int c = m_reader->get_char();
        if (c == EOF) break;
        line += c;
        if (c == '\r') {
            c = m_reader->get_char();
            if (c == '\n') {
                line += c;
                break;
            } else if (c == EOF) {
                break;
            } else {
                m_reader->unget_char();
            }
        } else if (c == '\n') {
            break;
        }
    }
    if (line.empty()) return false;

    if (line.size() && line.back() == '\n') line.pop_back();
    if (line.size() && line.back() == '\r') line.pop_back();
    return true;
}