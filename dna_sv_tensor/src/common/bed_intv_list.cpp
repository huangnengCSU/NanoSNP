#include "bed_intv_list.hpp"
#include "line_reader.hpp"

using namespace std;

BedIntvList::BedIntvList(reference_struct* reference, const char* bed_path)
{
    m_reference = reference;
    int n_seq = reference->seq_count();
    size_t total_base = 0;
    for (int i = 0; i < n_seq; ++i) total_base += reference->seq_size(i);
    m_intv_endpoint_list.resize(total_base + 2 * n_seq);
    fill(m_intv_endpoint_list.begin(), m_intv_endpoint_list.end(), 0);

    vector<string> columns;
    string line;
    LineReader* line_reader = new LineReader(bed_path);
    size_t bed_intv_base = 0;
    total_base = 0;
    int n_intv = 0;
    string last_name;
    int last_id = -1;
    int seq_size = 0;
    while (line_reader->getline(line)) {
        if (line[0] == '#') continue;
        columns.clear();
        split_line(line, "\t", columns);
        hbn_assert(columns.size() >= 3);
        string& name = columns[0];
        if (name != last_name) {
            last_name = name;
            last_id = reference->seq_id(name);
            total_base += reference->seq_size(last_id);
            seq_size = reference->seq_size(last_id);
        }
        int from = atoi(columns[1].c_str());
        int to = atoi(columns[2].c_str());
        hbn_assert(from < to);
        hbn_assert(to <= seq_size);
        m_intv_list.push_back(pair<int, int>(from, to));
        ++n_intv;
        bed_intv_base += (to - from);

        size_t g_from = reference->seq_offset(last_id) + from;
        size_t g_to = reference->seq_offset(last_id) + to;
        fill(m_intv_endpoint_list.begin() + g_from, m_intv_endpoint_list.begin() + g_to, 1);
    }
    delete line_reader;

    double p = 100.0 * bed_intv_base / total_base;
    fprintf(stderr, "BED file: %s\n", bed_path);
    fprintf(stderr, "Total bases: %zu, bed bases: %zu (%g)\n", total_base, bed_intv_base, p);
}

BedIntvList::~BedIntvList() {}

bool BedIntvList::region_intersect_with_bed_intv(int oid, int from, int to) const
{
    hbn_assert(from >= 0 || to >= 0);
    if (from >= 0 && to >= 0) {
        size_t list_idx_from = m_reference->seq_offset(oid) + from;
        size_t list_idx_to = m_reference->seq_offset(oid) + to;
        for (size_t i = list_idx_from; i < list_idx_to; ++i) {
            if (m_intv_endpoint_list[i]) return true;
        }
    } else if (from >= 0) {
        size_t list_idx = m_reference->seq_offset(oid) + from;
        return m_intv_endpoint_list[list_idx];
    } else if (to >= 0) {
        size_t list_idx = m_reference->seq_offset(oid) + to;
        return m_intv_endpoint_list[list_idx];
    }
    return false;
}

bool BedIntvList::region_intersect_with_bed_intv(const std::string& name, int from, int to) const
{
    int oid = m_reference->seq_id(name);
    return region_intersect_with_bed_intv(oid, from, to);
}