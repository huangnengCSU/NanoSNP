#include "genotype.hpp"

#include "line_reader.hpp"

#include <cstring>

#include <algorithm>
#include <unordered_map>

using namespace std;

const char* GT21_LABLES[] = {
    "AA",
    "AC",
    "AG",
    "AT",
    "CC",
    "CG",
    "CT",
    "GG",
    "GT",
    "TT",
    "DelDel",
    "ADel",
    "CDel",
    "GDel",
    "TDel",
    "InsIns",
    "AIns",
    "CIns",
    "GIns",
    "TIns",
    "InsDel"
};

const int variant_length_index_offset = 16;

const VariantLengthNamedTuple VariantLength = {
    .index_offset = variant_length_index_offset,
    .min = variant_length_index_offset,
    .max = variant_length_index_offset,
    .output_label_count = variant_length_index_offset * 2 + 1
};

const OutputLabelNamedTuple GT21 = {
    .output_label_count = 21,
    .y_start_index = 0,
    .y_end_index = 21
};

const OutputLabelNamedTuple GENOTYPE = {
    .output_label_count = 3,
    .y_start_index = GT21.y_end_index,
    .y_end_index = GT21.y_end_index + 3
};

const OutputLabelNamedTuple VARIANT_LENGTH_1 = {
    .output_label_count = VariantLength.output_label_count,
    .y_start_index = GENOTYPE.y_end_index,
    .y_end_index = GENOTYPE.y_end_index + VariantLength.output_label_count
};

const OutputLabelNamedTuple VARIANT_LENGTH_2 = {
    .output_label_count = VariantLength.output_label_count,
    .y_start_index = VARIANT_LENGTH_1.y_end_index,
    .y_end_index = VARIANT_LENGTH_1.y_end_index + VariantLength.output_label_count
};

const char* GENOTYPES[] = {
    "0/0",
    "1/1",
    "0/1",
    "1/2"
};

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

static unordered_map<const char*, EGT21Type, cstr_hash, cstr_eq> GT21_LABELS_MAP;
static int gt21_labels_map_has_been_initialized = 0;

void
init_gt21_labels_map()
{
    if (gt21_labels_map_has_been_initialized) return;

    for (int i = 0; i < eGT21_Size; ++i) {
        GT21_LABELS_MAP[GT21_LABLES[i]] = (EGT21Type)(i);
    }   
    for (int i = 0; i < eGT21_Size; ++i) {
        EGT21Type type = GT21_LABELS_MAP[GT21_LABLES[i]];
        hbn_assert(i == type);
    }
    gt21_labels_map_has_been_initialized = 1;      
}

//////////////////////

const char*
genotype_string_from(EGenoType type)
{
    return (type < e_geno_type_cnt) ? GENOTYPES[type] : "";
}

EGenoType
genotype_enum_from(int genotype_1, int genotype_2)
{
    if (genotype_1 == 0 && genotype_2 == 0) {
        return e_homo_reference;
    }

    if (genotype_1 == genotype_2) {
        return e_homo_variant;
    }

    if (genotype_1 != 0 && genotype_2 != 0) {
        return e_hetero_variant_multi;
    }

    return e_hetero_variant;
}

EGenoType
genotype_enum_for_task(EGenoType type)
{
    return (type == e_hetero_variant_multi) ? e_hetero_variant : type;
}

const char*
gt21_label_from_enum(EGT21Type gt21_enum)
{
    return (gt21_enum < eGT21_Size) ? GT21_LABLES[gt21_enum] : "";
}

EGT21Type
gt21_enum_from_label(const char* gt21_label)
{
#if 0
    static unordered_map<const char*, EGT21Type, cstr_hash, cstr_eq> GT21_LABELS_MAP;
    static int map_has_been_initialized = 0;
    if (!map_has_been_initialized) {
        for (int i = 0; i < eGT21_Size; ++i) {
            GT21_LABELS_MAP[GT21_LABLES[i]] = (EGT21Type)(i);
        }   
        for (int i = 0; i < eGT21_Size; ++i) {
            EGT21Type type = GT21_LABELS_MAP[GT21_LABLES[i]];
            hbn_assert(i == type);
        }
        map_has_been_initialized = 1;     
    }
#endif
    hbn_assert(gt21_labels_map_has_been_initialized);
    auto pos = GT21_LABELS_MAP.find(gt21_label);
    if (pos == GT21_LABELS_MAP.end()) {
        fprintf(stderr, "Invalid gt21 label %s", gt21_label);
        abort();
    }
    return pos->second;
}

string
partial_label_from(const string& ref, const string& alt)
{
    string result;
    if (ref.size() > alt.size()) {
        result = "Del";
    } else if (ref.size() < alt.size()) {
        result = "Ins";
    } else {
        result.push_back(alt[0]);
    }
    return result;
}

string 
mix_two_partial_labels(string& label1, string& label2)
{
    // AA, AC, AG, AT, CC, CG, CT, GG, GT, TT
    if (label1.size() == 1 && label2.size() == 1) {
        string result;
        if (label1 <= label2) {
            result += label1;
            result += label2;
        } else {
            result += label2;
            result += label1;
        }
        return result;
    }

    // ADel, CDel, GDel, TDel, AIns, CIns, GIns, TIns
    string tlb1 = label1;
    string tlb2 = label2;
    if (label1.size() > 1 && label2.size() == 1) {
        tlb1 = label2;
        tlb2 = label1;
    }
    if (tlb2.size() > 1 && tlb1.size() == 1) {
        return tlb1 + tlb2;
    }

    // InsIns, DelDel
    if (label1.size() > 0 && label2.size() > 0 && label1 == label2) {
        return label1 + label2;
    }

    // InsDel
    return gt21_label_from_enum(eGT21_InsDel);
}

EGT21Type
gt21_enum_from(const std::string& reference, 
    const std::string& alternate,
    const int genotype_1,
    const int genotype_2,
    std::vector<std::string>& alternate_arr)
{
    if (!alternate_arr.empty()) {
        vector<string> partial_labels;
        for (auto& alt : alternate_arr) {
            partial_labels.push_back(partial_label_from(reference, alt));
        }
        string gt21_label = mix_two_partial_labels(partial_labels[0], partial_labels[1]);
        return gt21_enum_from_label(gt21_label.c_str());
    }

    split_line(alternate, ",", alternate_arr);
    if (alternate_arr.size() == 1) {
        alternate_arr.clear();
        if (genotype_1 == 0 || genotype_2 == 0) {
            alternate_arr.push_back(reference);
            alternate_arr.push_back(alternate);
        } else {
            alternate_arr.push_back(alternate);
            alternate_arr.push_back(alternate);
        }
    }

    vector<string> partial_labels;
    for (auto& alt : alternate_arr) partial_labels.push_back(partial_label_from(reference, alt));
    string gt21_label = mix_two_partial_labels(partial_labels[0], partial_labels[1]);

    return gt21_enum_from_label(gt21_label.c_str());
}

int
Y_label_size()
{
    return GT21.output_label_count
           +
           GENOTYPE.output_label_count
           +
           VARIANT_LENGTH_1.output_label_count
           +
           VARIANT_LENGTH_2.output_label_count;
}

int
min_max(int value, int minimum, int maximum)
{
    return max(min(value, maximum), minimum);
}

void
output_labels_from_reference(const char reference_base, vector<int>& y_label)
{
    vector<int> gt21_vec(GT21.output_label_count, 0);
    string label;
    label += reference_base;
    label += reference_base;
    gt21_vec[gt21_enum_from_label(label.c_str())] = 1;

    vector<int> genotype_vec(GENOTYPE.output_label_count, 0);
    genotype_vec[e_homo_reference] = 1;

    vector<int> variant_length_vec_1(VARIANT_LENGTH_1.output_label_count, 0);
    vector<int> variant_length_vec_2(VARIANT_LENGTH_2.output_label_count, 0);
    variant_length_vec_1[0 + VariantLength.index_offset] = 1;
    variant_length_vec_2[0 + VariantLength.index_offset] = 1;

    y_label.clear();
    y_label.insert(y_label.end(), gt21_vec.begin(), gt21_vec.end());
    y_label.insert(y_label.end(), genotype_vec.begin(), genotype_vec.end());
    y_label.insert(y_label.end(), variant_length_vec_1.begin(), variant_length_vec_1.end());
    y_label.insert(y_label.end(), variant_length_vec_2.begin(), variant_length_vec_2.end());    
}

void
output_labels_from_vcf_columns(std::vector<std::string>& columns, vector<int>& y_label)
{
    const string& reference = columns[2];
    const string& alternate = columns[3];
    const int genotype_1 = atoi(columns[4].c_str());
    const int genotype_2 = atoi(columns[5].c_str());
    vector<string> alternate_arr;
    split_line(alternate, ",", alternate_arr);
    if (alternate_arr.size() == 1) {
        alternate_arr.clear();
        if (genotype_1 == 0 || genotype_2 == 0) {
            alternate_arr.push_back(reference);
            alternate_arr.push_back(alternate);
        } else {
            alternate_arr.push_back(alternate);
            alternate_arr.push_back(alternate);
        }
    }

    EGT21Type gt21 = gt21_enum_from(reference, alternate, genotype_1, genotype_2, alternate_arr);
    vector<int> gt21_vec(GT21.output_label_count, 0);
    gt21_vec[gt21] = 1;

    EGenoType genotype = genotype_enum_from(genotype_1, genotype_2);
    EGenoType genotype_for_task = genotype_enum_for_task(genotype);
    vector<int> genotype_vec(GENOTYPE.output_label_count, 0);
    genotype_vec[genotype_for_task] = 1;

    vector<int> variant_lengths;
    for (auto& alt : alternate_arr) {
        int alt_l = alt.size();
        int ref_l = reference.size();
        int len = min_max(alt_l - ref_l, VariantLength.min, VariantLength.max);
        variant_lengths.push_back(len);
    }
    sort(variant_lengths.begin(), variant_lengths.end());
    vector<int> variant_length_vec_1(VARIANT_LENGTH_1.output_label_count, 0);
    vector<int> variant_length_vec_2(VARIANT_LENGTH_2.output_label_count, 0);
    variant_length_vec_1[ variant_lengths[0] + VariantLength.index_offset ] = 1;
    variant_length_vec_2[ variant_lengths[1] + VariantLength.index_offset ] = 1;

    y_label.clear();
    y_label.insert(y_label.end(), gt21_vec.begin(), gt21_vec.end());
    y_label.insert(y_label.end(), genotype_vec.begin(), genotype_vec.end());
    y_label.insert(y_label.end(), variant_length_vec_1.begin(), variant_length_vec_1.end());
    y_label.insert(y_label.end(), variant_length_vec_2.begin(), variant_length_vec_2.end());
}

void
variant_map_from(const char* var_file_path,
    BedIntvList* tree, 
    std::map<std::string, size_t>& Y, 
    std::vector<int>& y_label_list, 
    std::set<std::string>& miss_variant_set)
{
    Y.clear();
    y_label_list.clear();
    miss_variant_set.clear();
    if (!var_file_path) return;

    LineReader* line_reader = new LineReader(var_file_path);
    string var_line;
    vector<string> var_components;
    vector<int> label;
    while (line_reader->getline(var_line)) {
        //cerr << var_line << endl;
        var_components.clear();
        split_line(var_line, "\t", var_components);
        const string& ctg_name = var_components[0];
        const string& position = var_components[1];
        //if (position != "5293389") continue;
        int n_comp = var_components.size();
        int genotype1 = atoi(var_components[n_comp - 2].c_str());
        int genotype2 = atoi(var_components[n_comp - 1].c_str());
        string key = ctg_name + ':' + position;
        //HBN_LOG("gt1 = %d, gt2 = %d, key = %s", genotype1, genotype2, key.c_str());
        if (genotype1 == -1 || genotype2 == -1) {
            miss_variant_set.insert(key);
            continue;
        }
        bool r = (!tree) || tree->region_intersect_with_bed_intv(ctg_name, atoi(position.c_str()), -1);
        if (!r) continue;
        output_labels_from_vcf_columns(var_components, label);
        Y[key] = y_label_list.size();
        y_label_list.insert(y_label_list.end(), label.begin(), label.end());
    }
    delete line_reader;
}