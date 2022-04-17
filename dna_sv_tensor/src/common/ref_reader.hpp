#ifndef __REF_READER_HPP
#define __REF_READER_HPP

#include "cpp_aux.hpp"
#include <cstdlib>
#include <map>

struct faidx_struct {
    std::string name;
    size_t seq_size;
    size_t seq_fasta_file_offset;
    size_t seq_base_list_offset;
    int base_per_line;
    int byte_per_line;
};

class reference_struct 
{
public:
    reference_struct(const char* fasta_path, bool load_sequence = true);
    ~reference_struct();

    size_t seq_offset(const std::string& name) const;
    size_t seq_offset(int oid) const;
    int seq_size(const std::string& name) const;
    int seq_size(int oid) const;
    int seq_count() const;
    const char* seq_name(int oid) const;
    int seq_id(const std::string& name) const;
    const char* seq(int oid) const;
    const char* seq(const std::string& name) const;

private:
    const char* m_fasta_path;
    std::map<std::string, int> m_name2id_map;
    std::vector<faidx_struct> m_faidx_list;
    std::vector<char> m_seq_base_list;
};

#endif // __REF_READER_HPP