#!/usr/bin/env python
import os
import sys
import time
import numpy as np
import datetime
from tqdm import tqdm
from argparse import ArgumentParser
from collections import defaultdict

major_contigs_order = ["chr" + str(a) for a in list(range(1, 23)) + ["X", "Y"]] + [
    str(a) for a in list(range(1, 23)) + ["X", "Y"]]


def Run(args):
    cat_file = args.cat_predict
    output_path = args.output
    pileup_vcf = args.pileup_vcf
    quality_threshold = args.quality

    cat_dict = defaultdict(defaultdict)
    row_count = 0
    no_vcf_output = True
    with open(cat_file, 'r') as fin:
        for row in fin:
            row_count += 1
            ctgname, pos, gt, qual = row.strip().split('\t')
            cat_dict[ctgname][pos] = (gt, qual)
            no_vcf_output = False
        if row_count == 0:
            print("[WARNING] No dp file found, please check the setting")
        if no_vcf_output:
            print("[WARNING] No dp results found, please check the setting")

    fout = open(output_path, 'w')

    modify_count = 0
    insert_HP=True
    pileup_ref_out = open("pileup_refcall.txt",'w')
    pileup_low_qual_out = open("pileup_lowqual.txt",'w')
    haplotype_ref_out = open("haplotype_refcall.txt",'w')
    with open(pileup_vcf, 'r') as fin:
        for line in fin:
            if line.startswith('#'):
                fout.write(line)
                if insert_HP:
                    fout.write('##INFO=<ID=P,Number=0,Type=Flag,Description="Result from pileup model">\n')
                    fout.write('##INFO=<ID=H,Number=0,Type=Flag,Description="Result from haplotype model">\n')
                    insert_HP = False
                continue
            fields = line.strip().split('\t')
            ref = fields[3]
            alt = fields[4]
            quality = float(fields[5])
            filt = fields[6]
            ctgname = fields[0]
            chr_offset = int(fields[1])
            zygosity = fields[-1].split(':')[0]
            depth, af = fields[-1].split(':')[-2:]
            depth = int(depth)
            af = float(af)
            zygosity = zygosity.replace('/', '|')
            if quality <= quality_threshold:
                try:
                    gt, qual = cat_dict[ctgname][str(chr_offset)]
                    qual = float(qual)
                    if qual < 13:
                        if filt != "RefCall" and quality>=13:
                            fields = line.strip().split('\t')
                            fields[7] = 'P'
                            line = '\t'.join(fields)
                            fout.write(line+'\n')
                        elif quality>=13 and filt=="RefCall":
                            pileup_ref_out.write(ctgname+'\t'+str(chr_offset)+'\t'+str(quality)+'\n')
                        else:
                            pileup_low_qual_out.write(ctgname+'\t'+str(chr_offset)+'\t'+str(quality)+'\n')
                        continue
                    if ref in gt:
                        # ref: A , alt: AA
                        # ref: A , alt: AC
                        if gt[0] == gt[1]:
                            haplotype_ref_out.write(ctgname+'\t'+str(chr_offset)+'\n')
                            continue
                        elif gt[0] != gt[1]:
                            new_gt = gt.replace(ref, '')
                            new_zy = '0/1'
                            quality = qual
                    else:
                        # ref: A , alt: CC
                        # ref: A , alt: CG
                        if gt[0] == gt[1]:
                            new_gt = gt[0]
                            new_zy = '1/1'
                            quality = qual
                        elif gt[0] != gt[1]:
                            new_gt = ','.join(sorted(gt))
                            new_zy = '1/2'
                            quality = qual
                    if 'D' in new_gt:
                        if new_zy == '0/1' or new_zy == '1/1':
                            continue
                        elif new_zy == '1/2':
                            new_gt = gt.replace('D', '')
                            new_zy = '0/1'
                    elif 'I' in new_gt:
                        if new_zy == '0/1' or new_zy == '1/1':
                            continue
                        elif new_zy == '1/2':
                            new_gt = gt.replace('I', '')
                            new_zy = '0/1'
                    fout.write("{0}\t{1}\t.\t{2}\t{3}\t{4}\t{5}\t{6}\t{7}\t{8}\n".format(ctgname, chr_offset, \
                                                                                         ref, new_gt, str(quality),
                                                                                         'PASS', 'H', 'GT:GQ:DP:AF',
                                                                                         new_zy + ":%s:%d:%f" % \
                                                                                         (
                                                                                             str(int(quality)), depth,
                                                                                             af)))
                    modify_count += 1
                except KeyError:
                    if filt != "RefCall" and quality>=13:
                        fields = line.strip().split('\t')
                        fields[7] = 'P'
                        line = '\t'.join(fields)
                        fout.write(line+'\n')
                    elif quality>=13 and filt=="RefCall":
                        pileup_ref_out.write(ctgname+'\t'+str(chr_offset)+'\t'+str(quality)+'\n')
                    else:
                        pileup_low_qual_out.write(ctgname+'\t'+str(chr_offset)+'\t'+str(quality)+'\n')
            else:
                if filt != "RefCall":
                    fields = line.strip().split('\t')
                    fields[7] = 'P'
                    line = '\t'.join(fields)
                    fout.write(line+'\n')
                else:
                    pileup_ref_out.write(ctgname+'\t'+str(chr_offset)+'\t'+str(quality)+'\n')
    fout.close()
    print('modify count:', modify_count)


def main():
    parser = ArgumentParser()
    parser.add_argument("--pileup_vcf", type=str, required=True,
                        help="Input the variants called by pileup model, required.")
    parser.add_argument("--cat_predict", type=str, required=True,
                        help="Input the predict results from CatModel, required.")
    parser.add_argument("--quality", type=float, default=15,
                        help="Input the quality whether the site will be filtered by edge model.")
    parser.add_argument("--output", type=str, required=True,
                        help="Input the output variants file, required.")
    args = parser.parse_args()
    Run(args)


if __name__ == '__main__':
    main()
