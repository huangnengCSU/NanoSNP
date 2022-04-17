#ifndef __GENOTYPE_HPP
#define __GENOTYPE_HPP

#include <map>
#include <set>
#include <string>
#include <vector>

#include "bed_intv_list.hpp"

extern 
const char* GT21_LABLES[];

typedef enum {
    eGT21_AA = 0,
    eGT21_AC,
    eGT21_AG,
    eGT21_AT,
    eGT21_CC,
    eGT21_CG,
    eGT21_CT,
    eGT21_GG,
    eGT21_GT,
    eGT21_TT,
    eGT21_DelDel,
    eGT21_ADel,
    eGT21_CDel,
    eGT21_GDel,
    eGT21_TDel,
    eGT21_InsIns,
    eGT21_AIns,
    eGT21_CIns,
    eGT21_GIns,
    eGT21_TIns,
    eGT21_InsDel,
    eGT21_Size
} EGT21Type;

typedef struct {
    int index_offset;
    int min;
    int max;
    int output_label_count;
} VariantLengthNamedTuple;

extern
const int variant_length_index_offset;

extern
const VariantLengthNamedTuple VariantLength;

typedef struct {
    int output_label_count;
    int y_start_index;
    int y_end_index;
} OutputLabelNamedTuple;

extern 
const OutputLabelNamedTuple GT21;

extern 
const OutputLabelNamedTuple GENOTYPE;

extern 
const OutputLabelNamedTuple VARIANT_LENGTH_1;

extern 
const OutputLabelNamedTuple VARIANT_LENGTH_2;

typedef enum {
    e_homo_reference = 0, // 0/0
    e_homo_variant,       // 1/1
    e_hetero_variant,     // 0/1 (or 1/2 for genotype task)
    e_hetero_variant_multi, // 1/2
    e_geno_type_cnt
} EGenoType;

extern
const char* GENOTYPES[];

int Y_label_size();

void
output_labels_from_reference(const char reference_base, std::vector<int>& y_label);

void
output_labels_from_vcf_columns(std::vector<std::string>& columns, std::vector<int>& y_label);

void
variant_map_from(const char* var_file_path,
    BedIntvList* tree, 
    std::map<std::string, size_t>& Y, 
    std::vector<int>& y_label_list, 
    std::set<std::string>& miss_variant_set);

void
init_gt21_labels_map();

#endif // __GENOTYPE_HPP