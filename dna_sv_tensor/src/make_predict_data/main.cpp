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

bool allow_duplicate_chr_pos = false;
int num_flanking_bases = 16;
int num_threads = 1;

const char* chr_tensor_dir = nullptr;
const char* reference_fasta_path = nullptr;
const char* output_dir = nullptr;

vector<const char*> chr_name_list;
int chr_name_idx = 0;
pthread_mutex_t chr_name_idx_lock;
reference_struct* reference = nullptr;

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
        if (strcmp(argv[i], "-reference") == 0) {
            reference_fasta_path = argv[i+1];
            i += 2;
            continue;
        }
        if (strcmp(argv[i], "-output_dir") == 0) {
            output_dir = argv[i+1];
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

struct VariantInfo {
    string key;
    string tensor;
    string ref_seq;
    string alt_info;
};

bool 
s_load_next_vaf_info(LineReader* line_reader, string& line, vector<string> columnes, VariantInfo& var_info)
{
    bool ret = false;
    while (1) {
        if (!line_reader->getline(line)) break;
        columnes.clear();
        split_line(line, "\t", columnes);
        string& chrom = columnes[0];
        string& coord = columnes[1];
        string& ref_seq = columnes[2];
        string& tensor = columnes[3];
        string& alt_info = columnes[4];
        while ((!alt_info.empty()) && isspace(alt_info.back())) alt_info.pop_back();

        for (auto& c : ref_seq) c = toupper(c);
        if (nst_nt16_table[(int)ref_seq[num_flanking_bases]] > 3) continue;

        var_info.key = chrom + ':' + coord;
        var_info.tensor = tensor;
        var_info.alt_info = alt_info;
        var_info.ref_seq = ref_seq;
        ret = true;
        break;
    }
    
    return ret;
}

void 
make_predict_array(const char* chr_tensor_path,
    const char* output,
    const char* chr_name)
{
    string line;
    vector<string> columns;
    VariantInfo var_info;
    LineReader* line_reader = new LineReader(chr_tensor_path);
    FILE* out = safe_fopen(output, "w");
    while (s_load_next_vaf_info(line_reader, line, columns, var_info)) {
        string& tensor = var_info.tensor;
        string& alt_info = var_info.alt_info;
        string& ref_seq = var_info.ref_seq;
        string& key = var_info.key;
        string pos = key + ':' + ref_seq;
        fprintf(out, "%s\t", tensor.c_str());
        fprintf(out, "%s\t", pos.c_str());
        fprintf(out, "%s\n", alt_info.c_str());
    }
    safe_fclose(out);
    delete line_reader;
}

void*
make_train_data_thread(void* params)
{
    string chr_tensor_path;
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

        output = output_dir;
        output += '/';
        output += chr_name;
        output += ".pd";

        make_predict_array(chr_tensor_path.c_str(), output.c_str(), chr_name);
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
    init_gt21_labels_map();
    create_directory(output_dir);

    pthread_t jobids[num_threads];
    for (int i = 0; i < num_threads; ++i) {
        pthread_create(&jobids[i], NULL, make_train_data_thread, NULL);
    }
    for (int i = 0; i < num_threads; ++i) {
        pthread_join(jobids[i], NULL);
    }

    if (reference) delete reference;
    return 0;
}