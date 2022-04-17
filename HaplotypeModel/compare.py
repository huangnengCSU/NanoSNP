import sys
import numpy as np
from get_truth import *

vcf_file = sys.argv[1]
bed_file = sys.argv[2]
ref_file = sys.argv[3]
failed_file = sys.argv[4]
fn_file = sys.argv[5]

faidx_dict = load_faidx(ref_file)
high_confident_variants = {}
for ctg in faidx_dict.keys():
    high_confident_variants[ctg] = np.zeros(shape=(faidx_dict[ctg], 3))  # first:confid bed, second:gt21, third:genotype
    high_confident_variants[ctg][:, 1:] -= 1  # ambiguities gt21: zero is AA, genotype: zero is homo_ref
load_confident_bed(bed_file, high_confident_variants)
load_truth_vcf(vcf_file, high_confident_variants)
fn_out = open(fn_file, 'w')
with open(failed_file, 'r') as fin:
    for line in fin:
        fields = line.strip().split('\t')
        ctgname, pos = fields[:2]
        pos = int(pos)
        if high_confident_variants[ctgname][pos - 1][0] > 0:
            if high_confident_variants[ctgname][pos - 1][2] == 2:
                fn_out.write(line)
fn_out.close()