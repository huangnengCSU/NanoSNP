import os
import sys
import time
import math
import tables
import numpy as np
import pysam
import datetime
import concurrent.futures
from tqdm import tqdm
from argparse import ArgumentParser
# from extract_adjacent_pileup import extract_pileups, extract_pileups_batch
from get_truth import get_truth_variants
from select_hetesnp_homosnp import select_snp, select_snp_multiprocess
from create_pileup_haplotype import single_group_pileup_haplotype_feature


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


def multigroups_pileup_haplotype_feature(bam, groups, chromosome_high_confident_variants, max_coverage, adjacent_size, pileup_flanking_size):
    ctgname = groups[0][0].ctgname
    out_candidate_positions, out_candidate_labels = [], []
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
        if all(len(rtv)>0 for rtv in [candidate_positions, haplotype_positions, haplotype_sequences, haplotype_baseq, haplotype_mapq, pileup_sequences, pileup_baseq, pileup_mapq, haplotype_hap, pileup_hap]):
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
            for pos in candidate_positions:
                ctg, p = pos.split(':')
                p = int(p)
                lbl = chromosome_high_confident_variants[p - 1]  # [1x3]
                out_candidate_labels.append(lbl)
        else:
            print("single_group_pileup_haplotype_feature output is None")
    return out_candidate_positions, out_candidate_labels, out_haplotype_positions, out_haplotype_sequences, out_haplotype_baseq, \
           out_haplotype_mapq, out_haplotype_hap, max_haplotype_depth, out_pileup_sequences, out_pileup_baseq, out_pileup_mapq, out_pileup_hap, max_pileup_depth


def Run(args):
    print(time.strftime(
        "[%a %b %d %H:%M:%S %Y] ============> Step 1/2: Get Truth", time.localtime()))
    high_confident_variants = get_truth_variants(args)
    print(time.strftime(
        "[%a %b %d %H:%M:%S %Y] ============> Step 2/2: Make Groups", time.localtime()))
    pileup_vcf = args.pileup_vcf
    low_quality_threshold = args.low_quality_threshold
    adjacent_size = args.adjacent_size
    pileup_flanking_size = args.pileup_flanking_size
    hete_support_quality = args.hete_support_quality

    groups_dict = select_snp_multiprocess(vcf_file=pileup_vcf, quality_threshold=low_quality_threshold, adjacent_size=adjacent_size, support_quality=hete_support_quality, nthreads=args.threads)
    # 把groups先按照染色体进行划分
    for k in groups_dict.keys():
        ## k is contig name
        out_candidate_positions, out_candidate_labels = [], []
        out_haplotype_hap, out_pileup_hap = [], []
        out_haplotype_positions, out_haplotype_sequences, out_haplotype_baseq, out_haplotype_mapq = [], [], [], []
        out_pileup_sequences, out_pileup_baseq, out_pileup_mapq = [], [], []
        max_pileup_depth, max_haplotype_depth = 0, 0
        chromosome_high_confident_variants = high_confident_variants[k]
        chromosome_groups = groups_dict[k]
        total_threads = args.threads
        step = math.ceil(len(chromosome_groups) / total_threads)
        divided_groups = [chromosome_groups[dt:dt + step] for dt in range(0, len(chromosome_groups), step)]  # divided for multiple threads
        samfile = args.bams + '/' + k + '.bam'
        assert os.path.exists(samfile)

        with concurrent.futures.ProcessPoolExecutor(max_workers=total_threads) as executor:
            signals = [executor.submit(multigroups_pileup_haplotype_feature, samfile, groups, chromosome_high_confident_variants,
                            args.max_coverage, adjacent_size, pileup_flanking_size) for groups in divided_groups]
            for sig in concurrent.futures.as_completed(signals):
                if sig.exception() is None:
                    # get the results
                    [candidate_positions, candidate_labels, haplotype_positions, haplotype_sequences, haplotype_baseq, haplotype_mapq, haplotype_hap, haplotype_depth,
                     pileup_sequences, pileup_baseq, pileup_mapq, pileup_hap, pileup_depth] = sig.result()
                    if all( len(rtv)>0 for rtv in [candidate_positions, candidate_labels, haplotype_positions, haplotype_sequences, haplotype_baseq, haplotype_mapq, haplotype_hap,
                     pileup_sequences, pileup_baseq, pileup_mapq, pileup_hap]):
                        out_candidate_positions.extend(candidate_positions)
                        out_candidate_labels.extend(candidate_labels)
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
                        print("multicandidates_pileup_haplotype_feature output is empty")
                else:
                    sys.stderr.write("ERROR: " + str(sig.exception()) + "\n")
                sig._result = None  # python issue 27144

            # 排序
        out_candidate_positions = np.array(out_candidate_positions)
        new_candidate_positions = [int(v.split(':')[1]) for v in out_candidate_positions]
        new_index = np.argsort(new_candidate_positions)
        out_candidate_positions = out_candidate_positions[new_index]
        out_haplotype_positions = np.array(out_haplotype_positions)[new_index]
        out_candidate_labels = np.array(out_candidate_labels)[new_index]

        out_haplotype_sequences = [np.expand_dims(np.pad(a, ((0, max_haplotype_depth - a.shape[0]), (0, 0)), 'constant', constant_values=-2), 0) for a in out_haplotype_sequences]
        out_haplotype_sequences = np.concatenate(out_haplotype_sequences)[new_index]

        out_haplotype_hap = [np.expand_dims(np.pad(a, ((0, max_haplotype_depth - a.shape[0]), (0, 0)), 'constant', constant_values=-2), 0) for a in out_haplotype_hap]
        out_haplotype_hap = np.concatenate(out_haplotype_hap)[new_index]

        out_haplotype_baseq = [np.expand_dims(np.pad(a, ((0, max_haplotype_depth - a.shape[0]), (0, 0)), 'constant', constant_values=-2), 0) for a in out_haplotype_baseq]
        out_haplotype_baseq = np.concatenate(out_haplotype_baseq)[new_index]

        out_haplotype_mapq = [np.expand_dims(np.pad(a, ((0, max_haplotype_depth - a.shape[0]), (0, 0)), 'constant', constant_values=-2), 0) for a in out_haplotype_mapq]
        out_haplotype_mapq = np.concatenate(out_haplotype_mapq)[new_index]

        out_pileup_sequences = [np.expand_dims(np.pad(a, ((0, max_pileup_depth - a.shape[0]), (0, 0)), 'constant', constant_values=-2), 0) for a in out_pileup_sequences]
        out_pileup_sequences = np.concatenate(out_pileup_sequences)[new_index]

        out_pileup_hap = [np.expand_dims(np.pad(a, ((0, max_pileup_depth - a.shape[0]), (0, 0)), 'constant', constant_values=-2), 0) for a in out_pileup_hap]
        out_pileup_hap = np.concatenate(out_pileup_hap)[new_index]

        out_pileup_baseq = [np.expand_dims(np.pad(a, ((0, max_pileup_depth - a.shape[0]), (0, 0)), 'constant', constant_values=-2), 0) for a in out_pileup_baseq]
        out_pileup_baseq = np.concatenate(out_pileup_baseq)[new_index]

        out_pileup_mapq = [np.expand_dims(np.pad(a, ((0, max_pileup_depth - a.shape[0]), (0, 0)), 'constant', constant_values=-2), 0) for a in out_pileup_mapq]
        out_pileup_mapq = np.concatenate(out_pileup_mapq)[new_index]

        TABLE_FILTERS = tables.Filters(complib='blosc:lz4hc', complevel=5)
        output = args.output + '/' + k + '.bin'
        table_file = tables.open_file(output, mode='w')
        int_atom = tables.Atom.from_dtype(np.dtype('int32'))
        string_atom = tables.StringAtom(itemsize=30 * (2 * adjacent_size))
        
        if args.max_pileup_depth is not None and args.max_pileup_depth < max_pileup_depth:
            max_pileup_depth = args.max_pileup_depth
        
        if args.max_haplotype_depth is not None and args.max_haplotype_depth < max_haplotype_depth:
            max_haplotype_depth = args.max_haplotype_depth

        table_file.create_earray(where='/', name='haplotype_sequences', atom=int_atom, shape=[0, max_haplotype_depth, 2 * adjacent_size + 1])
        table_file.create_earray(where='/', name='haplotype_hap', atom=int_atom, shape=[0, max_haplotype_depth, 2 * adjacent_size + 1])
        table_file.create_earray(where='/', name='haplotype_baseq', atom=int_atom, shape=[0, max_haplotype_depth, 2 * adjacent_size + 1])
        table_file.create_earray(where='/', name='haplotype_mapq', atom=int_atom, shape=[0, max_haplotype_depth, 2 * adjacent_size + 1])
        table_file.create_earray(where='/', name='pileup_sequences', atom=int_atom, shape=[0, max_pileup_depth, 2 * pileup_flanking_size + 1])
        table_file.create_earray(where='/', name='pileup_hap', atom=int_atom, shape=[0, max_pileup_depth, 2 * pileup_flanking_size + 1])
        table_file.create_earray(where='/', name='pileup_baseq', atom=int_atom, shape=[0, max_pileup_depth, 2 * pileup_flanking_size + 1])
        table_file.create_earray(where='/', name='pileup_mapq', atom=int_atom, shape=[0, max_pileup_depth, 2 * pileup_flanking_size + 1])
        table_file.create_earray(where='/', name='candidate_positions', atom=string_atom, shape=(0, 1), filters=TABLE_FILTERS)
        table_file.create_earray(where='/', name='haplotype_positions', atom=string_atom, shape=(0, adjacent_size * 2 + 1),filters=TABLE_FILTERS)
        table_file.create_earray(where='/', name='candidate_labels', atom=int_atom, shape=[0, 3])


        table_file.root.haplotype_sequences.append(out_haplotype_sequences[:, :max_haplotype_depth, :])
        table_file.root.haplotype_hap.append(out_haplotype_hap[:, :max_haplotype_depth, :])
        table_file.root.haplotype_baseq.append(out_haplotype_baseq[:, :max_haplotype_depth, :])
        table_file.root.haplotype_mapq.append(out_haplotype_mapq[:, :max_haplotype_depth, :])
        table_file.root.pileup_sequences.append(out_pileup_sequences[:, :max_pileup_depth, :])
        table_file.root.pileup_hap.append(out_pileup_hap[:, :max_pileup_depth, :])
        table_file.root.pileup_baseq.append(out_pileup_baseq[:, :max_pileup_depth, :])
        table_file.root.pileup_mapq.append(out_pileup_mapq[:, :max_pileup_depth, :])
        table_file.root.candidate_positions.append(np.array(out_candidate_positions).reshape(-1, 1))
        table_file.root.candidate_labels.append(np.array(out_candidate_labels).reshape(-1, 3))
        table_file.root.haplotype_positions.append(np.array(out_haplotype_positions).reshape(-1, adjacent_size * 2 + 1))
        table_file.close()
    print(time.strftime("[%a %b %d %H:%M:%S %Y] Done.", time.localtime()))


def main():
    parser = ArgumentParser(description="Create pileup and haplotype feature of candidate SNPs for training")
    parser.add_argument("--pileup_vcf", type=str, required=True,
                        help="Input the variants called by pileup model, required.")
    parser.add_argument("--bams", type=str, required=True,
                        help="Directory to phased bams, required.")
    parser.add_argument("--truth_vcf", type=str, required=True,
                        help="Input the truth variants with vcf format, required.")
    parser.add_argument("--confident_bed", type=str, required=True,
                        help="Input the confident bed file, required.")
    parser.add_argument("--reference", type=str, required=True,
                        help="Input the reference file, required.")
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
