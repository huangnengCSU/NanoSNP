#include "ref_reader.hpp"
#include "line_reader.hpp"

#include <limits>

using namespace std;

void
load_faidx_list(const char* fasta_path, std::vector<faidx_struct>& faidx_list)
{
    std::string faidx_path = fasta_path;
    faidx_path += ".fai";
    LineReader* line_reader = new LineReader(faidx_path.c_str());
    string line;
    vector<string> columns;
    faidx_struct faidx;
    size_t total_base = 0;
    while (line_reader->getline(line)) {
        columns.clear();
        split_line(line, "\t", columns);
        hbn_assert(columns.size() >= 5);
        faidx.name = columns[0];
        faidx.seq_size = atoll(columns[1].c_str());
        faidx.seq_fasta_file_offset = atoll(columns[2].c_str());
        faidx.seq_base_list_offset = total_base;
        faidx.base_per_line = atoi(columns[3].c_str());
        faidx.byte_per_line = atoi(columns[4].c_str());
        faidx_list.push_back(faidx);
        total_base += faidx.seq_size;
    }
    delete line_reader;
}

reference_struct::reference_struct(const char* fasta_path, bool load_sequence)
{
    load_faidx_list(fasta_path, m_faidx_list);
    m_fasta_path = fasta_path;
    int n_seq = m_faidx_list.size();
    for (int i = 0; i < n_seq; ++i) m_name2id_map[m_faidx_list[i].name] = i;
    if (!load_sequence) return;

    size_t total_base = 0;
    for (int i = 0; i < n_seq; ++i) total_base += m_faidx_list[i].seq_size + 2 * n_seq;
    m_seq_base_list.reserve(total_base);
    string line;
    //fprintf(stderr, "fasta_path: %s, n_seq: %d\n", fasta_path, n_seq);
    LineReader* line_reader = new LineReader(fasta_path);
    for (int i = 0; i < n_seq; ++i) {
        size_t loaded_base = 0;
        //fprintf(stderr, "load seq %d-%d\n", i, m_faidx_list[i].seq_size);
        int line_cnt = 0;
        hbn_assert(line_reader->getline(line));
        hbn_assert(line[0] == '>');
        while (1) {
            hbn_assert(line_reader->getline(line));
            //fprintf(stderr, "%d\t%d\t%s\n", line_cnt, line.size(), line.c_str());
            ++line_cnt;
            m_seq_base_list.insert(m_seq_base_list.end(), line.begin(), line.end());
            loaded_base += line.size();
            if (loaded_base == m_faidx_list[i].seq_size) break;
        }
    }
    delete line_reader;
}

reference_struct::~reference_struct() {}

size_t reference_struct::seq_offset(const std::string& name) const
{
    auto pos = m_name2id_map.find(name);
    hbn_assert(pos != m_name2id_map.end(), "sequence name '%s' does not exist in file '%s'", name.c_str(), m_fasta_path);
    return seq_offset(pos->second);
}

size_t reference_struct::seq_offset(int oid) const
{
    int n_seq = m_faidx_list.size();
    hbn_assert(oid >= 0 && oid < n_seq, "oid = %d, n_seq = %d", oid, n_seq);
    return m_faidx_list[oid].seq_base_list_offset;
}

int reference_struct::seq_size(const std::string& name) const
{
    auto pos = m_name2id_map.find(name);
    hbn_assert(pos != m_name2id_map.end(), "sequence name '%s' does not exist in file '%s'", name.c_str(), m_fasta_path);
    return seq_size(pos->second);    
}

int reference_struct::seq_size(int oid) const
{
    int n_seq = m_faidx_list.size();
    hbn_assert(oid >= 0 && oid < n_seq, "oid = %d, n_seq = %d", oid, n_seq);
    return m_faidx_list[oid].seq_size;    
}

int reference_struct::seq_count() const
{
    return m_faidx_list.size();
}

const char* reference_struct::seq_name(int oid) const
{
    int n_seq = m_faidx_list.size();
    hbn_assert(oid >= 0 && oid < n_seq, "oid = %d, n_seq = %d", oid, n_seq);
    return m_faidx_list[oid].name.c_str();    
}

int reference_struct::seq_id(const std::string& name) const
{
    auto pos = m_name2id_map.find(name);
    hbn_assert(pos != m_name2id_map.end(), "sequence name '%s' does not exist in file '%s'", name.c_str(), m_fasta_path);
    return pos->second;
}

const char* reference_struct::seq(int oid) const
{
    int n_seq = m_faidx_list.size();
    hbn_assert(oid >= 0 && oid < n_seq, "oid = %d, n_seq = %d", oid, n_seq);
    return m_seq_base_list.data() + m_faidx_list[oid].seq_base_list_offset;    
}

const char* reference_struct::seq(const std::string& name) const
{
    auto pos = m_name2id_map.find(name);
    hbn_assert(pos != m_name2id_map.end(), "sequence name '%s' does not exist in file '%s'", name.c_str(), m_fasta_path);
    return seq(pos->second);   
}