#include "tensor_maker.hpp"

using namespace std;

static const int kMaxIndelSize = 60;

#define extract_ref_base(__chr_seq, __chr_off) ((__chr_seq)[(__chr_off)-1])

struct cstr_eq
{
    bool operator()(const char* s1, const char* s2) const {
        return strcmp(s1, s2) == 0;
    }
};

struct cstr_hash
{
    size_t operator()(const char* str) const {
        size_t result = 0;
        const size_t prime = 31;
        const char* p = str;
        while (*p != '\0') {
            result = *p + (result * prime);
            ++p;
        }
        return result;
    }
};

TensorMaker::TensorMaker()
{
    fill(m_normal_mpileup_base_table, m_normal_mpileup_base_table + 256, 0);
    const char* normal_mpileup_list = "ACGTNacgtn*#";
    const char* p = normal_mpileup_list;
    while (*p) {
        m_normal_mpileup_base_table[int(*p)] = 1;
        ++p;
    }

    fill(m_fwd_normal_mpileup_base_table, m_fwd_normal_mpileup_base_table + 256, 0);
    const char* fwd_normal_mpileup_list = "ACGTN*";
    p = fwd_normal_mpileup_list;
    while (*p) {
        m_fwd_normal_mpileup_base_table[int(*p)] = 1;
        ++p;
    }

    fill(m_mpileup_char_to_channel_index_table, m_mpileup_char_to_channel_index_table + 256, eChannel_Size);
    m_mpileup_char_to_channel_index_table[(int)'A'] = eChannel_A;
    m_mpileup_char_to_channel_index_table[(int)'a'] = eChannel_a;
    m_mpileup_char_to_channel_index_table[(int)'C'] = eChannel_C;
    m_mpileup_char_to_channel_index_table[(int)'c'] = eChannel_c;
    m_mpileup_char_to_channel_index_table[(int)'G'] = eChannel_G;
    m_mpileup_char_to_channel_index_table[(int)'g'] = eChannel_g;
    m_mpileup_char_to_channel_index_table[(int)'T'] = eChannel_T;
    m_mpileup_char_to_channel_index_table[(int)'t'] = eChannel_t;
    m_mpileup_char_to_channel_index_table[(int)'*'] = eChannel_Star;
    m_mpileup_char_to_channel_index_table[(int)'#'] = eChannel_pound;
}

void
TensorMaker::make_tensor(const char* chr_seq,
            const int chr_off,
            const std::string& pileup_bases,
            tensor_cnt_type* pileup_tensor,
            std::map<std::string, int>& alt_dict,
            double& af,
            int& depth,
            bool& pass_af,
            bool& pass_snp_af,
            bool& pass_indel_af,
            const double snp_min_af,
            const double indel_min_af,
            std::vector<std::pair<std::string, int>>& pileup_list,
            int& max_del_length)
{
    char chr_base = evc_base_from(extract_ref_base(chr_seq, chr_off));
    chr_base = toupper(chr_base);
    fill(pileup_tensor, pileup_tensor + eChannel_Size, 0);
    int base_idx = 0, base_cnt = pileup_bases.size();
    map<string, int> cov_stats;
    string cov_key;
    while (base_idx < base_cnt) {
        char base = pileup_bases[base_idx];
        if (base == '+' || base == '-') {
            ++base_idx;
            int advance = 0;
            while (1) {
                char b = pileup_bases[base_idx];
                if (isdigit(b)) {
                    advance = advance * 10 + b - '0';
                    ++base_idx;
                } else {
                    break;
                }
            }
            if (advance <= kMaxIndelSize) {
                cov_key.clear();
                cov_key += base;
                cov_key.append(pileup_bases.c_str() + base_idx, advance);
                ++cov_stats[cov_key];
            }
            base_idx += advance - 1;
        } else if (m_normal_mpileup_base_table[(int)base]) {
            cov_key.clear();
            cov_key += base;
            ++cov_stats[cov_key];
        } else if (base == '^') { // '^' marks the start of a read, the following char is mapping quality
            ++base_idx;
        } else if (base == '$') { // '$' marks the end of a read
            // nothing to be done
        }
        ++base_idx;
    }

    int max_ins_0 = 0;
    int max_del_0 = 0;
    int max_ins_1 = 0;
    int max_del_1 = 0;
    depth = 0;
    max_del_length = 0;
    alt_dict.clear();
    string alt_key;
    map<string, int> pileup_dict;
    string pileup_key;
    string del_base;
    for (auto& cov : cov_stats) {
        const string& key = cov.first;
        const int count = cov.second;
        if (key[0] == '+') {
            alt_key.clear();
            alt_key += 'I';
            alt_key += chr_base;
            size_t key_l = key.size();
            for (size_t p = 1; p < key_l; ++p) alt_key += toupper(key[p]);
            alt_dict[alt_key] += count;
            pileup_key.clear();
            pileup_key += 'I';
            pileup_dict[pileup_key] += count;

            if (m_fwd_normal_mpileup_base_table[(int)key[1]]) { // forward strand
                pileup_tensor[eChannel_I] += count;
                max_ins_0 = max(max_ins_0, count);
            } else {
                pileup_tensor[eChannel_i] += count;
                max_ins_1 = max(max_ins_1, count);
            }
        } else if (key[0] == '-') {
            del_base.clear();
            int key_l = key.size();
            for (int i = 1; i < key_l; ++i) del_base += extract_ref_base(chr_seq, chr_off + i);
            alt_key.clear();
            alt_key = 'D' + del_base;
            alt_dict[alt_key] += count;
            pileup_key.clear();
            pileup_key += 'D';
            pileup_dict[pileup_key] += count;
            max_del_length = max(max_del_length, static_cast<int>(del_base.size()));

            if (m_fwd_normal_mpileup_base_table[(int)key[1]]) { // forward strand
                pileup_tensor[eChannel_D] += count;
                max_del_0 = max(max_del_0, count);
            } else {
                pileup_tensor[eChannel_d] += count;
                max_del_1 = max(max_del_1, count);
            }
        } else {
            if (nst_nt4_table[(int)key[0]] < 4) {
                pileup_key.clear();
                pileup_key += toupper(key[0]);
                pileup_dict[pileup_key] += count;
                depth += count;
                if (toupper(key[0]) != chr_base) {
                    alt_key.clear();
                    alt_key += 'X';
                    alt_key += toupper(key[0]);
                    alt_dict[alt_key] += count;
                }
                pileup_tensor[m_mpileup_char_to_channel_index_table[(int)key[0]]] += count;
            } else if (key[0] == '*') {
                pileup_tensor[eChannel_Star] += count;
                depth += count;
            } else if (key[0] == '#') {
                pileup_tensor[eChannel_pound] += count;
                depth += count;
            }
        }
    }

    pileup_tensor[eChannel_I1] = max_ins_0;
    pileup_tensor[eChannel_i1] = max_ins_1;
    pileup_tensor[eChannel_D1] = max_del_0;
    pileup_tensor[eChannel_d1] = max_del_1;

    int denominator = depth ? depth : 1;
    pileup_list.clear();
    for (auto& p : pileup_dict) pileup_list.push_back(pair<string, int>(p.first, p.second));
    sort(pileup_list.begin(), pileup_list.end(),
        [](const pair<string, int>& a, const pair<string, int>& b)->bool { return a.second > b.second; });
    
    pass_snp_af = false;
    pass_indel_af = false;
    pass_af = pileup_list.size() && pileup_list[0].first[0] != chr_base;

    for (auto& p : pileup_list) {
        const string& item = p.first;
        const int count = p.second;
        if (item.size() == 1 && item[0] == chr_base) {
            continue;
        } else if (item[0] == 'I' || item[0] == 'D') {
            pass_indel_af = pass_indel_af
                            ||
                            (1.0 * count / denominator >= indel_min_af);
            continue;
        }
        pass_snp_af = pass_snp_af
                      ||
                      (1.0 * count / denominator >= snp_min_af);
    }

    af = (pileup_list.size() > 1)
         ?
         (1.0 * pileup_list[1].second / denominator)
         :
         0.0;
    if (pileup_list.size() && pileup_list[0].first[0] != chr_base) {
        af = 1.0 * pileup_list[0].second / denominator;
    }

    int chr_base_pt = 0;
    chr_base_pt += pileup_tensor[eChannel_A];
    chr_base_pt += pileup_tensor[eChannel_C];
    chr_base_pt += pileup_tensor[eChannel_G];
    chr_base_pt += pileup_tensor[eChannel_T];
    int pt_i = m_mpileup_char_to_channel_index_table[(int)chr_base];
    hbn_assert(pt_i >= 0 && pt_i < eChannel_Size);
    pileup_tensor[pt_i] = -chr_base_pt;

    int lower_chr_base_pt = 0;
    lower_chr_base_pt += pileup_tensor[eChannel_a];
    lower_chr_base_pt += pileup_tensor[eChannel_c];
    lower_chr_base_pt += pileup_tensor[eChannel_g];
    lower_chr_base_pt += pileup_tensor[eChannel_t];
    pt_i = m_mpileup_char_to_channel_index_table[tolower(chr_base)];
    hbn_assert(pt_i >= 0 && pt_i < eChannel_Size, "chr_base = %c", chr_base);
    pileup_tensor[pt_i] = -lower_chr_base_pt;

    pass_af = pass_af || pass_snp_af || pass_indel_af;
}