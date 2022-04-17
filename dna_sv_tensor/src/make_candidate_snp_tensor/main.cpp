#include "tensor_maker.hpp"

#include <algorithm>
#include <map>
#include <set>
#include <sstream>

#include <cstdlib>
#include <cstring>

#include "../common/cpp_aux.hpp"
#include "../common/line_reader.hpp"
#include "../common/kstring.h"
#include "../common/ref_reader.hpp"

using namespace std;

//#define T_AF 0.16
#define T_AF 0.12

double min_af = T_AF;
double snp_min_af = T_AF;
double indel_min_af = T_AF;
int min_coverage = 6;
int flanking_base_num = 16;
int num_threads = 1;

const char* reference_path = nullptr;
const char* chr_pileup_dir = nullptr;
const char* extended_confident_bed_path = nullptr;
const char* confident_bed_path = nullptr;
const char* output_dir = nullptr;

vector<const char*> chr_name_list;
int chr_name_idx = 0;
pthread_mutex_t chr_name_idx_lock;

reference_struct* reference = nullptr;
BedIntvList* extended_confident_bed_list = nullptr;
BedIntvList* confident_bed_list = nullptr;

void
parse_args(int argc, char* argv[])
{
    int i = 1;
    while (i < argc) {
        if (argv[i][0] != '-') break;

        if (strcmp(argv[i], "-reference") == 0) {
            reference_path = argv[i+1];
            i += 2;
            continue;
        }
        if (strcmp(argv[i], "-chr_pileup_dir") == 0) {
            chr_pileup_dir = argv[i + 1];
            i += 2;
            continue;
        }
        if (strcmp(argv[i], "-output_dir") == 0) {
            output_dir = argv[i + 1];
            i += 2;
            continue;
        }
        if (strcmp(argv[i], "-extended_confident_bed") == 0) {
            extended_confident_bed_path = argv[i + 1];
            i += 2;
            continue;
        }
        if (strcmp(argv[i], "-confident_bed") == 0) {
            confident_bed_path = argv[i + 1];
            i += 2;
            continue;
        }
        if (strcmp(argv[i], "-min_af") == 0) {
            min_af = atof(argv[i+1]);
            i += 2;
            continue;
        }
        if (strcmp(argv[i], "-snp_min_af") == 0) {
            snp_min_af = atof(argv[i+1]);
            i += 2;
            continue;
        }
        if (strcmp(argv[i], "-indel_min_af") == 0) {
            indel_min_af = atof(argv[i+1]);
            i += 2;
            continue;
        }  
        if (strcmp(argv[i], "-min_coverage") == 0) {
            min_coverage = atoi(argv[i+1]);
            i += 2;
            continue;
        }    
        if (strcmp(argv[i], "-flanking_base") == 0) {
            flanking_base_num = atoi(argv[i+1]);
            i += 2;
            continue;
        }
        if (strcmp(argv[i], "-num_threads") == 0) {
            num_threads = atoi(argv[i+1]);
            i += 2;
            continue;
        }

        fprintf(stderr, "Unrecognised option '%s'\n", argv[i]);
        exit (1);
    }

    chr_name_list.clear();
    for (; i < argc; ++i) chr_name_list.push_back(argv[i]);
}

void
create_pileup_tensor(reference_struct* reference,
    BedIntvList* extended_bed_list,
    BedIntvList* confident_bed_list,
    const char* chr_name,
    const char* chr_mpileup_path,
    const char* tensor_path,
    const char* alt_info_path)
{
    string cpp_chr_name(chr_name);
    const int chr_id = reference->seq_id(cpp_chr_name);
    const int chr_size = reference->seq_size(chr_id);
    const char* chr_seq = reference->seq(chr_id);
    const int sliding_window_size = 2 * flanking_base_num + 1;
    array<tensor_cnt_type, eChannel_Size>* tensor = new array<tensor_cnt_type, eChannel_Size>[sliding_window_size];
    array<tensor_cnt_type, eChannel_Size> pileup_tensor;
    int pos_offset = 0;
    int pre_ref_off = -1;
    map<string, int> alt_dict;
    map<int, map<string, int>> all_alt_dict;
    double af;
    map<int, double> af_dict;
    int depth;
    map<int, int> depth_dict;
    bool pass_af, pass_snp_af, pass_indel_af;
    vector<pair<string, int>> pileup_list;
    int max_del_length;
    vector<int> candidate_position;
    int num_filled_tensor = 0;
    FILE* out = safe_fopen(tensor_path, "w");
    FILE* alt_info_out = nullptr;
    if (alt_info_path) alt_info_out = safe_fopen(alt_info_path, "w");

    LineReader* pileup_reader = new LineReader(chr_mpileup_path);
    string pileup_line;
    vector<string> pileup_components;
    int line_cnt = 0;
    TensorMaker tensor_maker;
    string ref_subseq;
    string alt_info;
    string tensor_info;
    ks_dinit(tensor_out_s);
    ks_dinit(alt_info_out_s);
    int buf_item = 0;
    ostringstream os;
    while (pileup_reader->getline(pileup_line)) {
        ++line_cnt;
        //if ((line_cnt % 10000) == 0) fprintf(stderr, "%10d (%s) loci processed.\n", line_cnt, chr_name);
        pileup_components.clear();
        split_line(pileup_line, "\t", pileup_components);
        const int ref_off = atoll(pileup_components[1].c_str());
        //if (ref_off != 10270435) continue;
        bool r = (!extended_bed_list) || extended_bed_list->region_intersect_with_bed_intv(chr_id, ref_off - 1, -1);
        if (!r) {
            //HBN_LOG("ref_off = %d out of bed", ref_off);
            continue;
        }
        hbn_assert(ref_off <= chr_size);
        const char ref_base = toupper(chr_seq[ref_off-1]);
        const string& pileup_bases = pileup_components[4];

        if (pre_ref_off + 1 != ref_off) {
            num_filled_tensor = 0;
            pos_offset = 0;
            candidate_position.clear();
        }
        pre_ref_off = ref_off;

        alt_dict.clear();
        pileup_list.clear();
        tensor_maker.make_tensor(chr_seq, ref_off, pileup_bases, pileup_tensor.data(), alt_dict, af, depth,
            pass_af, pass_snp_af, pass_indel_af, snp_min_af, indel_min_af, pileup_list, max_del_length);

#if 0
        fprintf(stderr, "cnt = %d, ref_off = %d, ref_base = %c, %s %g %d %d %d\n", 
            cnt, ref_off, ref_base, pileup_bases.c_str(), af, depth, pass_af, max_del_length);
        fprintf(stderr, "[");
        for (int i = 0; i < eChannel_Size; ++i) fprintf(stderr, "%d ", pileup_tensor[i]);
        fprintf(stderr, "]\n");
#endif

        bool pass_confident_bed = confident_bed_list ? confident_bed_list->region_intersect_with_bed_intv(chr_id, ref_off - 1, ref_off + max_del_length + 1) : true;
        //fprintf(stderr, "pass_confident_bed = %d\n", pass_confident_bed);
        if (pass_confident_bed && nst_nt4_table[(int)ref_base] < 4 && pass_af && depth >= min_coverage) {
            candidate_position.push_back(ref_off);
            all_alt_dict[ref_off] = alt_dict;
            depth_dict[ref_off] = depth;
            af_dict[ref_off] = af;
        }
        tensor[pos_offset] = pileup_tensor;
        ++num_filled_tensor;
        //fprintf(stderr, "ref_off = %d, ref_base = %c, num_filled_tensor = %d\n", ref_off, ref_base, num_filled_tensor);

        //if (candidate_position.size()) HBN_LOG("p0 = %d, num_filled_tensor = %d", candidate_position[0], num_filled_tensor);
        pos_offset = (pos_offset + 1) % sliding_window_size;
        if (candidate_position.size() > 0 && ref_off - candidate_position[0] == flanking_base_num) {
            const int center = candidate_position.front(); 
            //HBN_LOG("center = %d, window_size = %d", center, sliding_window_size);
            if (num_filled_tensor < sliding_window_size) {
                candidate_position.erase(candidate_position.begin());
                all_alt_dict.erase(center);
                depth_dict.erase(center);
                af_dict.erase(center);
                continue;
            }           
            depth = depth_dict[center];

            ref_subseq.clear();
            for (int p = center - flanking_base_num; p < center + flanking_base_num + 1; ++p) {
                ref_subseq += chr_seq[p-1];
            }

            os.str("");
            os << depth << '-';
            auto& center_dict = all_alt_dict[center];
            for (auto& dict_item : center_dict) {
                os << dict_item.first << ' ' << dict_item.second << ' ';
            }
            alt_info = os.str();
            
            os.str("");
            for (int i = pos_offset; i < sliding_window_size; ++i) {
                for (int j = 0; j < eChannel_Size; ++j) {
                    os << tensor[i][j] << ' ';
                }
            }
            for (int i = 0; i < pos_offset; ++i) {
                for (int j = 0; j < eChannel_Size; ++j) {
                    os << tensor[i][j] << ' ';
                }
            }
            tensor_info = os.str();

            ksprintf(&tensor_out_s, "%s\t%d\t%s\t%s\t%s\n",
                pileup_components[0].c_str(),
                center,
                ref_subseq.c_str(),
                tensor_info.c_str(),
                alt_info.c_str());

            if (alt_info_out) {
                os.str("");
                for (auto& dict_item : center_dict) {
                    os << dict_item.first << ' ' << dict_item.second << ' ';
                }
                alt_info = os.str();
                ksprintf(&alt_info_out_s, "%s\t%d\t%d\t%s\t%lf\n", 
                    pileup_components[0].c_str(),
                    center,
                    depth,
                    alt_info.c_str(),
                    af_dict[center]);
            }
            ++buf_item;
            if (buf_item == 2000) {
                size_t l = ks_size(tensor_out_s);
                const char* s = ks_s(tensor_out_s);
                if (ks_back(tensor_out_s) == '\0') --l;
                safe_fwrite(s, 1, l, out);
                if (alt_info_out) {
                    l = ks_size(alt_info_out_s);
                    s = ks_s(alt_info_out_s);
                    if (ks_back(alt_info_out_s) == '\0') --l;
                    safe_fwrite(s, 1, l, alt_info_out);
                }
                ks_clear(tensor_out_s);
                ks_clear(alt_info_out_s);
                buf_item = 0;
            }

            candidate_position.erase(candidate_position.begin());
            all_alt_dict.erase(center);
            depth_dict.erase(center);
            af_dict.erase(center);
        }
    }

    if (buf_item) {
        size_t l = ks_size(tensor_out_s);
        const char* s = ks_s(tensor_out_s);
        if (ks_back(tensor_out_s) == '\0') --l;
        safe_fwrite(s, 1, l, out);
        if (alt_info_out) {
            l = ks_size(alt_info_out_s);
            s = ks_s(alt_info_out_s);
            if (ks_back(alt_info_out_s) == '\0') --l;
            safe_fwrite(s, 1, l, alt_info_out);
        }
        ks_clear(tensor_out_s);
        ks_clear(alt_info_out_s);
        buf_item = 0;
    }

    delete pileup_reader;
    ks_destroy(tensor_out_s);
    ks_destroy(alt_info_out_s);
    delete[] tensor;
    if (out) safe_fclose(out);
    if (alt_info_out) safe_fclose(alt_info_out);
}

void*
create_tensor_thread(void* params)
{
    string chr_fasta_path;
    string chr_extended_bed_path;
    string chr_bed_path;
    string chr_mpileup_path;
    string tensor_path;
    string alt_info_path;

    while (1) {
        int i = -1;
        pthread_mutex_lock(&chr_name_idx_lock);
        i = chr_name_idx++;
        pthread_mutex_unlock(&chr_name_idx_lock);
        if (i >= chr_name_list.size()) break;

        const char* chr_name = chr_name_list[i];

        chr_mpileup_path = chr_pileup_dir;
        chr_mpileup_path += '/';
        chr_mpileup_path += chr_name;
        chr_mpileup_path += ".mpileup";

        tensor_path = output_dir;
        tensor_path += '/';
        tensor_path += chr_name;
        tensor_path += ".tensor";

        alt_info_path = output_dir;
        alt_info_path += '/';
        alt_info_path += chr_name;
        alt_info_path += ".alt_info";

        create_pileup_tensor(reference,
            extended_confident_bed_list,
            confident_bed_list,
            chr_name,
            chr_mpileup_path.c_str(),
            tensor_path.c_str(),
            alt_info_path.c_str());
    }
    return nullptr;
}

int main(int argc, char* argv[])
{
    parse_args(argc, argv);
    fprintf(stderr, "chr: ");
    for (auto chr_name : chr_name_list) fprintf(stderr, "%s ", chr_name);
    fprintf(stderr, "\n");

    create_directory(output_dir);

    chr_name_idx = 0;
    pthread_mutex_init(&chr_name_idx_lock, NULL);
    reference = new reference_struct(reference_path);
    if (extended_confident_bed_path) 
        extended_confident_bed_list = new BedIntvList(reference, extended_confident_bed_path);
    if (confident_bed_path)
        confident_bed_list = new BedIntvList(reference, confident_bed_path);

    pthread_t jobids[num_threads];
    for (int i = 0; i < num_threads; ++i) {
        pthread_create(&jobids[i], NULL, create_tensor_thread, NULL);
    }
    for (int i = 0; i < num_threads; ++i) {
        pthread_join(jobids[i], NULL);
    }

    if (extended_confident_bed_list) delete extended_confident_bed_list;
    if (confident_bed_list) delete confident_bed_list;
    if (reference) delete reference;
    return 0;
}