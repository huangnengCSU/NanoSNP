#include "../common/tensor.hpp"
#include "../common/bed_intv_list.hpp"
#include "../common/ref_reader.hpp"
#include "../common/genotype.hpp"
#include "../common/line_reader.hpp"
#include "../common/cpp_aux.hpp"

#include <algorithm>    // std::shuffle
#include <array>        // std::array
#include <random>       // std::default_random_engine
#include <chrono>       // std::chrono::system_clock
#include <algorithm>
#include <map>
#include <set>
#include <iostream>

#include <cstdlib>
#include <cstring>

using namespace std;

bool shuffle_tensors = true;
bool allow_duplicate_chr_pos = false;
double maximum_non_variant_ratio = 5.0;
int num_flanking_bases = 16;
int num_threads = 1;

const char* chr_tensor_dir = nullptr;
const char* chr_true_var_dir = nullptr;
const char* reference_fasta_path = nullptr;
const char* confident_bed_path = nullptr;
const char* output_dir = nullptr;

vector<const char*> chr_name_list;
int chr_name_idx = 0;
pthread_mutex_t chr_name_idx_lock;
reference_struct* reference = nullptr;
BedIntvList* confident_bed_list = nullptr;

void parse_args(int argc, char* argv[])
{
    int i = 1;
    while (i < argc) {
        if (argv[i][0] != '-') break;

        if (strcmp(argv[i], "-chr_tensor_dir") == 0) {
            chr_tensor_dir = argv[i+1];
            i += 2;
            continue;
        }
        if (strcmp(argv[i], "-chr_true_var_dir") == 0) {
            chr_true_var_dir = argv[i+1];
            i += 2;
            continue;
        }        
        if (strcmp(argv[i], "-reference") == 0) {
            reference_fasta_path = argv[i+1];
            i += 2;
            continue;
        }
        if (strcmp(argv[i], "-confident_bed") == 0) {
            confident_bed_path = argv[i+1];
            i += 2;
            continue;
        }
        if (strcmp(argv[i], "-output_dir") == 0) {
            output_dir = argv[i+1];
            i += 2;
            continue;
        }
        if (strcmp(argv[i], "-shuffle_tensors") == 0) {
            shuffle_tensors = atoi(argv[i+1]);
            i += 2;
            continue;
        }
        if (strcmp(argv[i], "-maxinum_non_variant_ratio") == 0) {
            maximum_non_variant_ratio = atof(argv[i+1]);
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
load_true_variants(const char* var_path,
    map<string, size_t>& Y,
    vector<int>& y_list,
    map<string, string>& tp_var_info)
{
    Y.clear();
    y_list.clear();
    tp_var_info.clear();

    vector<int> label;
    string line;
    vector<string> columns;
    LineReader* line_reader = new LineReader(var_path);
    while (line_reader->getline(line)) {
        columns.clear();
        split_line(line, "\t", columns);
        //cerr << line << '\n';
        const string chr_name = columns[0];
        const string chr_offset = columns[1];
        int n_col = columns.size();
        int gt1 = atoi(columns[n_col - 2].c_str());
        int gt2 = atoi(columns[n_col - 1].c_str());
        string key = chr_name + ':' + chr_offset;
        if (gt1 == -1 || gt2 == -1) continue;
        output_labels_from_vcf_columns(columns, label);
        hbn_assert(Y.find(key) == Y.end());
        Y[key] = y_list.size();
        y_list.insert(y_list.end(), label.begin(), label.end());
        hbn_assert(tp_var_info.find(key) == tp_var_info.end());
        tp_var_info[key] = line;
    }
    delete line_reader;
}

double
calc_non_variant_subsample_ratio(const char* chr_alt_info_path, 
    const map<string, size_t>& Y,
    const double maximum_non_variant_ratio,
    const int chr_id)
{
    double subsample_ratio = 1.0;
    if (!chr_alt_info_path) return subsample_ratio;

    string line;
    vector<string> columns;
    LineReader* line_reader = new LineReader(chr_alt_info_path);
    int variant_cnt = 0, non_variant_cnt = 0;
    int confident_canvar = 0;
    while (line_reader->getline(line)) {
        columns.clear();
        split_line(line, "\t", columns);
        string key = columns[0] + ':' + columns[1];
        auto key_pos = Y.find(key);
        if (key_pos != Y.end()) {
            ++variant_cnt;
        } else {
            ++non_variant_cnt;
        }

        if (confident_bed_list && key_pos != Y.end()) {
            int chr_off = atoi(columns[1].c_str());
            if (confident_bed_list->region_intersect_with_bed_intv(chr_id, chr_off, -1)) ++confident_canvar;
        }
    }
    delete line_reader;

    int max_non_variant_num = variant_cnt * maximum_non_variant_ratio;
    if (max_non_variant_num < non_variant_cnt) {
        subsample_ratio = 1.0 * max_non_variant_num / non_variant_cnt;
    }
    fprintf(stderr, "variants / non_variants / subsample_ratio : %d / %d / %g\n", variant_cnt, non_variant_cnt, subsample_ratio);
    
    int total_variant = Y.size();
    double recall_rate = 1.0 * variant_cnt / total_variant;
    fprintf(stderr, "True variants / recall / recall_ratio : %d / %d / %g\n", total_variant, variant_cnt, recall_rate);

    if (confident_bed_list) {
        int confident_true_var = 0;
        for (auto& y : Y) {
            auto pos = y.first.find(':');
            hbn_assert(pos != string::npos);
            int chr_off = atoi(y.first.c_str() + pos + 1);
            //HBN_LOG("key: %s, pos = %zu, chr_off = %d", y.first.c_str(), pos, chr_off);
            if (confident_bed_list->region_intersect_with_bed_intv(chr_id, chr_off, -1)) ++confident_true_var;
        }
        double r = 1.0 * confident_canvar / confident_true_var;
        fprintf(stderr, "confident_true_variants / confident_candidate_variants / recall: %d / %d /%g\n", confident_true_var, confident_canvar, r);
    }

    return subsample_ratio;
}

struct VariantInfo {
    string key;
    string tensor;
    string ref_seq;
    string alt_info;
};

class TensorReader
{
public:
    TensorReader(const char* tensor_file_path);

    ~TensorReader() {
        if (m_tensor_reader) delete m_tensor_reader;
    }

    bool load_next_batch(map<string, size_t>& Y,
            vector<int>& y_label_list,
            const double non_variant_subsample_ratio,
            map<string, size_t>& X,
            vector<VariantInfo>& var_info_list);

private:
    vector<std::string> m_tensor_file_list;
    int m_tensor_file_idx;
    LineReader* m_tensor_reader;
    string m_tensor_line;
    vector<string> m_tensor_components;
    int m_num_added_tensor;
    int m_dup_id_counter;
    map<char, char> m_iupac_to_acgt_map;
};

TensorReader::TensorReader(const char* tensor_file_path)
{
    m_tensor_file_list.push_back(tensor_file_path);
    m_tensor_file_idx = 0;
    m_tensor_reader = new LineReader(m_tensor_file_list.front().c_str());
    m_dup_id_counter = 0;
    m_num_added_tensor = 0;

/*
A->A
C->C
G->G
T or U->T
R->A or G
Y->C or T
S->G or C
W->A or T
K->G or T
M->A or C
B->C or G or T
D->A or G or T
H->A or C or T
V->A or C or G
*/
    string key =   "ACGTURYSWKMBDHVN";
    string value = "ACGTTACCAGACAAAA";
    for (size_t i = 0; i != key.size(); ++i) {
        m_iupac_to_acgt_map[key[i]] = value[i];
    }
}

bool TensorReader::load_next_batch(map<string, size_t>& Y,
    vector<int>& y_label_list,
    const double non_variant_subsample_ratio,
    map<string, size_t>& X,
    vector<VariantInfo>& var_info_list)
{
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dist(0.0, 1.0);

    X.clear();
    var_info_list.clear();
    int added_tensor = 0;
    vector<int> y_label;
    while (1) {
        if (!m_tensor_reader->getline(m_tensor_line)) {
            ++m_tensor_file_idx;
            if (m_tensor_file_idx >= m_tensor_file_list.size()) break;
            delete m_tensor_reader;
            //HBN_LOG("file_idx = %d, total = %d", m_tensor_file_idx, m_tensor_file_list.size());
            m_tensor_reader = new LineReader(m_tensor_file_list[m_tensor_file_idx].c_str());
            continue;
        }

        m_tensor_components.clear();
        split_line(m_tensor_line, "\t", m_tensor_components);
        string& chrom = m_tensor_components[0];
        string& coord = m_tensor_components[1];
        string& ref_seq = m_tensor_components[2];
        string& tensor = m_tensor_components[3];
        string& alt_info = m_tensor_components[4];
        while ((!alt_info.empty()) && isspace(alt_info.back())) alt_info.pop_back();

        for (auto& c : ref_seq) c = toupper(c);
        if (nst_nt16_table[(int)ref_seq[num_flanking_bases]] > 3) continue;

        string key = chrom + ':' + coord;
        bool is_reference = (Y.find(key) == Y.end());
        bool r = is_reference && (non_variant_subsample_ratio < 1.0) && (dist(gen) >= non_variant_subsample_ratio);
        if (r) continue;

        VariantInfo vi;
        vi.key = key;
        vi.tensor = tensor;
        vi.alt_info = alt_info;
        vi.ref_seq = ref_seq;
        if (X.find(key) == X.end()) {
            X[key] = var_info_list.size();
            var_info_list.push_back(vi);
        } else if (allow_duplicate_chr_pos) {
            ++m_dup_id_counter;
            char s_buf[64];
            sprintf(s_buf, "%d", m_dup_id_counter);
            string new_key = s_buf;
            new_key += '_';
            new_key += key;
            X[new_key] = var_info_list.size();
            vi.key = new_key;
            var_info_list.push_back(vi);
        }

        if (is_reference) {
            char ref_base = m_iupac_to_acgt_map[ref_seq[num_flanking_bases]];
            output_labels_from_reference(ref_base, y_label);
            Y[key] = y_label_list.size();
            y_label_list.insert(y_label_list.end(), y_label.begin(), y_label.end());
        }

        ++m_num_added_tensor;
        //if ((m_num_added_tensor % 10000) == 0) fprintf(stderr, "%10d tensors processed.\n", m_num_added_tensor);
        ++added_tensor;
        if (added_tensor == 10000) break;
    }

    return !X.empty();
}

void 
make_training_array(const char* chr_tensor_path,
    const char* chr_alt_info_path,
    const char* chr_true_var_path,
    const char* output,
    const char* chr_name)
{
    string cpp_chr_name = chr_name;
    int chr_id = reference ? reference->seq_id(cpp_chr_name) : -1;
    map<string, size_t> Y;
    set<string> miss_variant_set;
    vector<int> y_label_list;
    const int y_label_size = Y_label_size();
    map<string, string> tp_var_info;
    load_true_variants(chr_true_var_path, Y, y_label_list, tp_var_info);
    double non_variant_subsample_ratio = calc_non_variant_subsample_ratio(chr_alt_info_path, Y, maximum_non_variant_ratio, chr_id);

    TensorReader reader(chr_tensor_path);
    map<string, size_t> X;
    vector<VariantInfo> var_info_list;
    vector<size_t> X_list;
    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    FILE* out = safe_fopen(output, "w");
    while (reader.load_next_batch(Y, y_label_list, non_variant_subsample_ratio, X, var_info_list)) {
        X_list.clear();
        for (auto& x : X) X_list.push_back(x.second);
        sort(X_list.begin(), X_list.end());
        if (shuffle_tensors) std::shuffle(X_list.begin(), X_list.end(), std::default_random_engine(seed));
        for (auto x : X_list) {
            string& tensor = var_info_list[x].tensor;
            string& alt_info = var_info_list[x].alt_info;
            string& ref_seq = var_info_list[x].ref_seq;
            string& key = var_info_list[x].key;
            auto iter = Y.find(key);
            string pos;
            int* label;
            if (iter != Y.end()) {
                label = y_label_list.data() + iter->second;
                pos = key + ':' + ref_seq;
            } 

            fprintf(out, "%s\t", tensor.c_str());
            for (int i = 0; i < y_label_size - 1; ++i) {
                fprintf(out, "%d ", label[i]);
            }
            fprintf(out, "%d\t", label[y_label_size - 1]);
            fprintf(out, "%s\t", pos.c_str());
            fprintf(out, "%s", alt_info.c_str());
            auto tp_iter = tp_var_info.find(key);
            if (tp_iter != tp_var_info.end()) {
                fprintf(out, "\t%s", tp_iter->second.c_str());
                //is_deletion_true_variant(tp_iter->second);
                //is_insertion_true_variant(tp_iter->second);
            }
            fprintf(out, "\n"); 
        }    
    }
    safe_fclose(out);
}

void*
make_train_data_thread(void* params)
{
    string chr_tensor_path;
    string chr_alt_info_path;
    string chr_true_var_path;
    string output;

    while (1) {
        int i = -1;
        pthread_mutex_lock(&chr_name_idx_lock);
        i = chr_name_idx++;
        pthread_mutex_unlock(&chr_name_idx_lock);
        if (i >= chr_name_list.size()) break;
        const char* chr_name = chr_name_list[i];

        chr_tensor_path = chr_tensor_dir;
        chr_tensor_path += '/';
        chr_tensor_path += chr_name;
        chr_tensor_path += ".tensor";

        chr_alt_info_path = chr_tensor_dir;
        chr_alt_info_path += '/';
        chr_alt_info_path += chr_name;
        chr_alt_info_path += ".alt_info";

        chr_true_var_path = chr_true_var_dir;
        chr_true_var_path += '/';
        chr_true_var_path += chr_name;
        chr_true_var_path += ".true_var";

        output = output_dir;
        output += '/';
        output += chr_name;
        output += ".td";

        make_training_array(chr_tensor_path.c_str(),
            chr_alt_info_path.c_str(),
            chr_true_var_path.c_str(),
            output.c_str(),
            chr_name);
    }

    return nullptr;
}

int main(int argc, char* argv[])
{
    parse_args(argc, argv);
    fprintf(stderr, "chr: ");
    for (auto chr_name: chr_name_list) fprintf(stderr, "%s ", chr_name);
    fprintf(stderr, "\n");

    chr_name_idx = 0;
    pthread_mutex_init(&chr_name_idx_lock, NULL);
    if (reference_fasta_path) reference = new reference_struct(reference_fasta_path, false);
    if (confident_bed_path) confident_bed_list = new BedIntvList(reference, confident_bed_path);
    init_gt21_labels_map();
    create_directory(output_dir);

    pthread_t jobids[num_threads];
    for (int i = 0; i < num_threads; ++i) {
        pthread_create(&jobids[i], NULL, make_train_data_thread, NULL);
    }
    for (int i = 0; i < num_threads; ++i) {
        pthread_join(jobids[i], NULL);
    }

    if (confident_bed_list) delete confident_bed_list;
    if (reference) delete reference;
    return 0;
}