#ifndef __TENSOR_MAKER_HPP
#define __TENSOR_MAKER_HPP

#include "../common/tensor.hpp"
#include "../common/bed_intv_list.hpp"
#include "../common/ref_reader.hpp"

#include <algorithm>
#include <map>
#include <set>
#include <unordered_map>

#include <cstdlib>
#include <cstring>

class TensorMaker
{
public:
    TensorMaker();

    void make_tensor(const char* chr_seq,
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
            int& max_del_length);

private:
    inline char 
    evc_base_from(char base) {
        return (nst_nt4_table[(int)base] < 4)
                ?
                base
                :
                (isupper(base) ? 'A' : 'a');
    }

private:
    char m_normal_mpileup_base_table[256];
    char m_fwd_normal_mpileup_base_table[256];
    int m_mpileup_char_to_channel_index_table[256];
};

#endif // __TENSOR_MAKER_HPP