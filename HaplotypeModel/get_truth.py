from argparse import ArgumentParser
from enum import Enum
import numpy as np


class EGT21Type(Enum):
    eGT21_AA = 0
    eGT21_AC = 1
    eGT21_AG = 2
    eGT21_AT = 3
    eGT21_CC = 4
    eGT21_CG = 5
    eGT21_CT = 6
    eGT21_GG = 7
    eGT21_GT = 8
    eGT21_TT = 9
    eGT21_DelDel = 10
    eGT21_ADel = 11
    eGT21_CDel = 12
    eGT21_GDel = 13
    eGT21_TDel = 14
    eGT21_InsIns = 15
    eGT21_AIns = 16
    eGT21_CIns = 17
    eGT21_GIns = 18
    eGT21_TIns = 19
    eGT21_InsDel = 20
    eGT21_Size = 21


class EGenoType(Enum):
    e_homo_reference = 0  # 0/0
    e_homo_variant = 1  # 1/1
    e_hetero_variant = 2  # 0/1
    e_hetero_variant_multi = 3
    e_geno_type_cnt = 4


GT21_LABLES = [
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
    "--",
    "A-",
    "C-",
    "G-",
    "T-",
    "++",
    "A+",
    "C+",
    "G+",
    "T+",
    "+-"
]

GT21_LABELS_MAP = {
    "AA": 0,
    "AC": 1,
    "AG": 2,
    "AT": 3,
    "CC": 4,
    "CG": 5,
    "CT": 6,
    "GG": 7,
    "GT": 8,
    "TT": 9,
    "DelDel": 10,
    "ADel": 11,
    "CDel": 12,
    "GDel": 13,
    "TDel": 14,
    "InsIns": 15,
    "AIns": 16,
    "CIns": 17,
    "GIns": 18,
    "TIns": 19,
    "InsDel": 20
}


def load_reference_file(ref_fn):
    seq = ""
    ctg = ""
    references = {}
    with open(ref_fn, 'r') as fin:
        for line in fin:
            if line.startswith('>'):
                if len(seq) > 0:
                    references[ctg] = seq
                    seq = ""
                    ctg = ""
                ctg = line.strip().split(' ')[0].replace('>', '')
            else:
                seq += line.strip()
        if len(seq) > 0:
            references[ctg] = seq
    return references


def load_faidx(fasta_path):
    faidx_path = fasta_path
    faidx_path += '.fai'
    line_reader = open(faidx_path, 'r')
    faidx_dict = {}
    for line in line_reader:
        ctgname, seq_size = line.strip().split('\t')[:2]
        faidx_dict[ctgname] = int(seq_size)
    return faidx_dict


def load_confident_bed(bed_file, high_confident_variants):
    line_reader = open(bed_file, 'r')
    for line in line_reader:
        ctg, start_point, end_point = line.strip().split('\t')
        start_point = int(start_point)
        end_point = int(end_point)
        for i in range(start_point, end_point):
            high_confident_variants[ctg][i - 1][0] = 1


def partial_label_from(ref, alt):
    if len(ref) > len(alt):
        result = "Del"
    elif len(ref) < len(alt):
        result = "Ins"
    else:
        result = alt[0]
    return result


def gt21_label_from_enum(gt_21_enum):
    if gt_21_enum.value < EGT21Type.eGT21_Size.value:
        return GT21_LABLES[gt_21_enum.value]
    else:
        return ""


def mix_two_partial_labels(label1, label2):
    if len(label1) == 1 and len(label2) == 1:
        # A, C, G, T
        result = ""
        if label1 <= label2:
            result += label1
            result += label2
        else:
            result += label2
            result += label1
        return result
    tlb1 = label1  # Ins
    tlb2 = label2  # A
    if len(label1) > 1 and len(label2) == 1:
        # Ins, A
        tlb1 = label2  # A
        tlb2 = label1  # Ins
    if len(tlb2) > 1 and len(tlb1) == 1:
        return tlb1 + tlb2
    if len(label1) > 0 and len(label2) > 0 and label1 == label2:
        # InsIns, DelDel
        return label1 + label2
    return "InsDel"


def gt21_enum_from_label(gt21_label):
    return GT21_LABELS_MAP[gt21_label]


def gt21_enum_from(reference, alternate, genotype_1, genotype_2, alternate_arr):
    if len(alternate_arr) > 0:
        partial_labels = []
        for alt in alternate_arr:
            partial_labels.append(partial_label_from(reference, alt))
        gt21_label = mix_two_partial_labels(partial_labels[0], partial_labels[1])  # 'Ins, Del, A, C, G, T'
        return GT21_LABELS_MAP[gt21_label]

    alternate_arr = alternate.split(',')
    if len(alternate_arr) == 1:
        alternate_arr.clear()
        if genotype_1 == 0 or genotype_2 == 0:
            alternate_arr.append(reference)
            alternate_arr.append(alternate)
        else:
            alternate_arr.append(alternate)
            alternate_arr.append(alternate)
    partial_labels = []
    for alt in alternate_arr:
        partial_labels.append(partial_label_from(reference, alt))
    gt21_label = mix_two_partial_labels(partial_labels[0], partial_labels[1])
    return GT21_LABELS_MAP[gt21_label]


def genotype_enum_from(genotype_1, genotype_2):
    if genotype_1 == 0 and genotype_2 == 0:
        return EGenoType.e_homo_reference.value
    if genotype_1 == genotype_2:
        return EGenoType.e_homo_variant.value
    if genotype_1 != 0 and genotype_2 != 0:
        return EGenoType.e_hetero_variant_multi.value
    return EGenoType.e_hetero_variant.value


def genotype_enum_for_task(type):
    if type == EGenoType.e_hetero_variant_multi.value:
        return EGenoType.e_hetero_variant.value
    else:
        return type


def min_max(value, minimum, maximum):
    return max(min(value, maximum), minimum)


def output_labels_from_vcf_columns(columns):
    reference = columns[3]
    alternate = columns[4]
    zygosity = columns[-1].split(':')[0]
    zygosity = zygosity.replace('/', '|')
    genotype_1, genotype_2 = [int(v) for v in zygosity.split('|')]
    alternate_arr = alternate.split(',')
    if (len(alternate_arr) == 1):
        alternate_arr.clear()
        if genotype_1 == 0 or genotype_2 == 0:
            alternate_arr.append(reference)
            alternate_arr.append(alternate)
        else:
            alternate_arr.append(alternate)
            alternate_arr.append(alternate)
    gt21 = gt21_enum_from(reference, alternate, genotype_1, genotype_2, alternate_arr)

    genotype = genotype_enum_from(genotype_1, genotype_2)
    genotype_for_task = genotype_enum_for_task(genotype)

    return gt21, genotype_for_task


def load_truth_vcf(vcf_file, high_confident_variants):
    line_reader = open(vcf_file, 'r')
    for line in line_reader:
        if line.startswith('#'):
            continue
        else:
            fields = line.strip().split('\t')
            ctgname = fields[0]
            chr_offset = int(fields[1])
            if high_confident_variants[ctgname][chr_offset - 1][0] == 0:
                continue
            else:
                gt21, genotype_for_task = output_labels_from_vcf_columns(fields)
                high_confident_variants[ctgname][chr_offset - 1][1:3] = [gt21, genotype_for_task]


def get_truth_variants(args):
    fasta_path = args.reference
    confident_bed_path = args.confident_bed
    references = load_reference_file(fasta_path)
    faidx_dict = load_faidx(fasta_path)
    truth_vcf_path = args.truth_vcf
    high_confident_variants = {}
    for ctg in faidx_dict.keys():
        ref = np.fromstring(references[ctg], dtype=np.uint8)
        ref[np.where((ref == 65) | (ref == 97))] = 0
        ref[np.where((ref == 67) | (ref == 99))] = 4
        ref[np.where((ref == 71) | (ref == 103))] = 7
        ref[np.where((ref == 84) | (ref == 116))] = 9
        assert faidx_dict[ctg] == len(ref)
        high_confident_variants[ctg] = np.zeros(shape=(faidx_dict[ctg], 3))  # first:confid bed, second:gt21, third:genotype
        high_confident_variants[ctg][:, 1] = ref   # record the reference sequence in gt21, A/a:0, C/c:4, G/g:7, T/t:9, others base:ascii
        high_confident_variants[ctg][:, 2] -= 1  # ambiguities gt21: zero is AA, genotype: zero is homo_ref
    load_confident_bed(confident_bed_path, high_confident_variants)
    load_truth_vcf(truth_vcf_path, high_confident_variants)
    return high_confident_variants


if __name__ == '__main__':
    parser = ArgumentParser(description="Get Truth Variants")
    parser.add_argument("--truth_vcf", type=str, required=True,
                        help="Input the truth variants with vcf format, required.")
    parser.add_argument("--confident_bed", type=str, required=True, help="Input the confident bed file, required.")
    parser.add_argument("--reference", type=str, required=True, help="Input the reference file, required.")
    args = parser.parse_args()
    high_confident_variants = get_truth_variants(args)
