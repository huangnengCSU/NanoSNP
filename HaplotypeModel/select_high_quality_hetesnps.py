import argparse
from collections import defaultdict
import concurrent.futures
import math
import sys
from datetime import datetime
import os


major_contigs_order = ["chr" + str(a) for a in list(range(1, 23)) + ["X", "Y"]] + [
    str(a) for a in list(range(1, 23)) + ["X", "Y"]]


class SNPItem:
    def __init__(self, ctgname, pos, homo_hete, info=None):
        self.ctgname = ctgname
        self.position = pos
        self.homo_hete = homo_hete
        self.info = info

    def is_heterozygous(self):
        if self.homo_hete == '0/1' or self.homo_hete == '0|1' or self.homo_hete == '1/2' or self.homo_hete == '1|2':
            return True
        else:
            return False

def select_high_quality_hetesnps(vcf_file, out_dir, support_quality=15):
    header = []
    contig_dict = {}
    row_count = 0
    with open(vcf_file, 'r') as fin:
        for row in fin:
            row_count += 1
            if row[0] == '#':
                if row not in header:
                    header.append(row)
                continue
            columns = row.strip().split()
            ctg_name = columns[0]
            pos = int(columns[1])
            ref_base = columns[3]
            alt_base = columns[4]
            quality = float(columns[5])
            genotype = columns[9].split(':')[0].replace('|', '/')  # 0/0, 1/1, 0/1
            if (genotype == '0/0') or (genotype == '1/1'):
                continue
            if quality >= support_quality:
                contig_dict[ctg_name] = contig_dict.get(ctg_name, [])
                contig_dict[ctg_name].append(row)
    for ctg in contig_dict.keys():
        # chr20.splited.vcf
        with open(out_dir+'/'+ctg+'.splited.vcf','w') as fout:
            for row in header:
                fout.write(row)
            for row in contig_dict[ctg]:
                fout.write(row)

def main():
    parse = argparse.ArgumentParser()
    parse.add_argument('--pileup_vcf', help='Input pileup vcf file', required=True)
    parse.add_argument('--support_quality', help='Min quality of hetezygous SNP used for phasing. (default: %(default)f)', default=16, type=float)
    parse.add_argument('--output_dir',required=True)
    args = parse.parse_args()
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)
    select_high_quality_hetesnps(vcf_file=args.pileup_vcf, support_quality=args.support_quality, out_dir=args.output_dir)


if __name__ == '__main__':
    main()
