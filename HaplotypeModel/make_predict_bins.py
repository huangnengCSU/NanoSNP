from ast import arg
import multiprocessing
import os
import sys
import time
import math
import tables
import numpy as np
import pysam
import datetime
import concurrent.futures
from multiprocessing import Pool
from tqdm import tqdm
from argparse import ArgumentParser
# from extract_adjacent_pileup import extract_pileups, extract_pileups_batch
from select_hetesnp_homosnp import select_snp, select_snp_multiprocess
from create_pileup_haplotype import single_group_pileup_haplotype_feature
from write_to_bins import write_to_bins


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


class GroupQueue:
    def __init__(self, maxsize):
        self.maxsize = maxsize
        self.__queue = []

    def full(self):
        return len(self.__queue) == self.maxsize

    def empty(self):
        return len(self.__queue) == 0

    def qsize(self):
        return len(self.__queue)

    def get(self):
        tmp = self.__queue[0]
        self.__queue = self.__queue[1:]
        return tmp

    def put(self, item):
        if self.full():
            self.get()
        self.__queue.append(item)

    def show_middle(self):
        if self.full():
            return self.__queue[self.maxsize // 2]
        else:
            return None

    def show_last(self):
        if not self.empty():
            return self.__queue[self.qsize() - 1]
        else:
            return None

    def show_queue(self):
        q = self.__queue
        return q

    def empty_queue(self):
        self.__queue = []


def multigroups_pileup_haplotype_feature(bam, groups, max_coverage, adjacent_size, pileup_flanking_size):
    ctgname = groups[0][0].ctgname
    out_candidate_positions = []
    out_haplotype_hap, out_pileup_hap = [], []
    out_haplotype_positions, out_haplotype_sequences, out_haplotype_baseq, out_haplotype_mapq = [], [], [], []
    out_pileup_sequences, out_pileup_baseq, out_pileup_mapq = [], [], []
    max_haplotype_depth, max_pileup_depth = 0, 0
    samfile = pysam.AlignmentFile(bam, 'r')
    # 对groups再进行分组sub_group，每个sub_group包含100个group，如果某个group与前一个group相距远，则直接中断，新起sub_group
    dt = 0
    step = 100
    # sub_groups = [groups[dt:dt + step] for dt in range(0, len(groups), step)]  # each time pysam pileup processes multiple groups for acceleration
    sub_groups = []
    num_groups = len(groups)
    while dt < num_groups:
        for i in range(1, step):
            if dt + i >= num_groups:
                sub_groups.append(groups[dt:dt + i])
                dt = dt + i
                break
            else:
                if int(groups[dt + i][0].position) - int(groups[dt + i - 1][-1].position) > 1000:
                    sub_groups.append(groups[dt:dt + i])
                    dt = dt + i
                    break
                elif i == step - 1:
                    sub_groups.append(groups[dt:dt + i + 1])
                    dt = dt + i + 1
                    break
                else:
                    pass
    for sub_sub_groups in sub_groups:
        candidate_positions, haplotype_positions, haplotype_sequences, haplotype_baseq, haplotype_mapq, haplotype_hap, haplotype_depth, pileup_sequences, pileup_baseq, \
        pileup_mapq, pileup_hap, pileup_depth = single_group_pileup_haplotype_feature(samfile, sub_sub_groups, max_coverage, adjacent_size, pileup_flanking_size)
        if all( len(rtv)>0 for rtv in [candidate_positions, haplotype_positions, haplotype_sequences, haplotype_baseq, haplotype_mapq, haplotype_hap, pileup_sequences, pileup_baseq, pileup_mapq, pileup_hap]):
            out_candidate_positions.extend(candidate_positions)
            out_haplotype_positions.extend(haplotype_positions)
            out_haplotype_sequences.extend(haplotype_sequences)
            out_haplotype_hap.extend(haplotype_hap)
            out_haplotype_baseq.extend(haplotype_baseq)
            out_haplotype_mapq.extend(haplotype_mapq)
            max_haplotype_depth = haplotype_depth if haplotype_depth > max_haplotype_depth else max_haplotype_depth
            out_pileup_sequences.extend(pileup_sequences)
            out_pileup_hap.extend(pileup_hap)
            out_pileup_baseq.extend(pileup_baseq)
            out_pileup_mapq.extend(pileup_mapq)
            max_pileup_depth = pileup_depth if pileup_depth > max_pileup_depth else max_pileup_depth
        else:
            print("single_group_pileup_haplotype_feature output is None")
    return out_candidate_positions, out_haplotype_positions, out_haplotype_sequences, out_haplotype_baseq, \
           out_haplotype_mapq, out_haplotype_hap, max_haplotype_depth, out_pileup_sequences, out_pileup_baseq, out_pileup_mapq, out_pileup_hap, max_pileup_depth


def Run(args):
    pileup_vcf = args.pileup_vcf
    low_quality_threshold = args.low_quality_threshold
    adjacent_size = args.adjacent_size
    pileup_flanking_size = args.pileup_flanking_size
    hete_support_quality = args.hete_support_quality

    groups_dict = select_snp_multiprocess(vcf_file=pileup_vcf, quality_threshold=low_quality_threshold, adjacent_size=adjacent_size, support_quality=hete_support_quality, nthreads=args.threads)
    # 把groups先按照染色体进行划分
    for k in groups_dict.keys():
        ## k is contig name
        # out_candidate_positions = []
        # out_haplotype_hap, out_pileup_hap = [], []
        # out_haplotype_positions, out_haplotype_sequences, out_haplotype_baseq, out_haplotype_mapq = [], [], [], []
        # out_pileup_sequences, out_pileup_baseq, out_pileup_mapq = [], [], []
        max_pileup_depth, max_haplotype_depth = 0, 0
        chromosome_groups = groups_dict[k]
        total_threads = args.threads
        step = math.ceil(len(chromosome_groups) / total_threads)
        if step == 0:
            continue
        divided_groups = [chromosome_groups[dt:dt + step] for dt in range(0, len(chromosome_groups), step)]  # divided for multiple threads
        samfile = args.bams + '/' + k + '.bam'
        assert os.path.exists(samfile)

        with Pool(processes=total_threads) as pool:
            signals = [pool.apply_async(multigroups_pileup_haplotype_feature, args=(samfile, groups, args.max_coverage, adjacent_size, pileup_flanking_size)) for groups in divided_groups]
            if len(signals)==0:
                print("No signals, continue!")
                continue
            if len(signals) > 0:
                for sig in signals:
                    [candidate_positions, haplotype_positions, haplotype_sequences, haplotype_baseq, haplotype_mapq, haplotype_hap, haplotype_depth, pileup_sequences, pileup_baseq, pileup_mapq, pileup_hap, pileup_depth] = sig.get()
                    if all( len(rtv)>0 for rtv in [candidate_positions, haplotype_positions, haplotype_sequences, haplotype_baseq, haplotype_mapq, haplotype_hap, pileup_sequences, pileup_baseq, pileup_mapq, pileup_hap]):
                        # max_haplotype_depth = haplotype_depth if haplotype_depth > max_haplotype_depth else max_haplotype_depth
                        # max_pileup_depth = pileup_depth if pileup_depth > max_pileup_depth else max_pileup_depth
                        write_to_bins(args=args,
                                      contig_name=k,
                                      adjacent_size=adjacent_size,
                                      pileup_flanking_size=pileup_flanking_size,
                                      out_candidate_positions=candidate_positions,
                                      out_haplotype_positions=haplotype_positions,
                                      out_haplotype_sequences=haplotype_sequences,
                                      out_haplotype_hap=haplotype_hap,
                                      out_haplotype_baseq=haplotype_baseq,
                                      out_haplotype_mapq=haplotype_mapq,
                                      out_pileup_sequences=pileup_sequences,
                                      out_pileup_hap=pileup_hap,
                                      out_pileup_baseq=pileup_baseq,
                                      out_pileup_mapq=pileup_mapq,
                                      max_haplotype_depth=haplotype_depth,
                                      max_pileup_depth=pileup_depth)
                    else:
                        print("multicandidates_pileup_haplotype_feature output is empty")
    print(time.strftime("[%a %b %d %H:%M:%S %Y] Done.", time.localtime()))


def main():
    parser = ArgumentParser(description="Create pileup and haplotype feature of candidate SNPs for predicting")
    parser.add_argument("--pileup_vcf", type=str, required=True,
                        help="Input the variants called by pileup model, required.")
    parser.add_argument("--bams", type=str, required=True,
                        help="Directory to phased bams, required.")
    # parser.add_argument("--reference", type=str, required=True,
    #                     help="Input the reference file, required.")
    parser.add_argument("--output", type=str, required=True,
                        help="Input the output directory, required.")
    parser.add_argument("--pileup_flanking_size", type=int, default=5,
                        help="The flanking size used to construct the pileup feature of the candidate SNP , default: %(default)d")                  
    parser.add_argument("--adjacent_size", type=int, default=5,
                        help="The range of SNPS around the candidate sites to be considered as groups, default: %(default)d")
    parser.add_argument("--low_quality_threshold", type=int, default=19,
                        help="The low quality threshold of snp site for haplotype model.")
    parser.add_argument('--hete_support_quality', default=14, type=float,
                        help='Min quality of hetezygous SNP used for making up of group. (default: %(default)f)')
    parser.add_argument("--max_coverage", type=int, default=150,
                        help="The maximum coverage of each position, default: %(default)d")
    parser.add_argument("--max_pileup_depth", type=int, default=None,help="The maximum depth of pileup feature, default: %(default)d")
    parser.add_argument("--max_haplotype_depth", type=int, default=None,help="The maximum depth of haplotype feature, default: %(default)d")
    parser.add_argument("--threads", '-t', type=int, default=1,
                        help="The number of threads used for computing: %(default)d")
    args = parser.parse_args()
    if len(sys.argv[1:]) == 0:
        parser.print_help()
        sys.exit(1)
    if not os.path.exists(args.output):
        os.mkdir(args.output)
    Run(args)


if __name__ == "__main__":
    main()
