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
from extract_adjacent_pileup import extract_pileups, extract_pileups_batch
from select_hetesnp_homosnp import select_snp


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


def extract_pileups_group(bam, groups, max_coverage, pileup_length):
    ctgname = groups[0][0].ctgname
    out_cand_pos, out_cand_edge_columns, out_cand_edge_matrix, out_cand_pair_columns, out_cand_pair_route = [], [], [], [], []
    out_cand_group_pos, out_cand_read_matrix, out_cand_baseq_matrix, out_cand_mapq_matrix = [], [], [], []
    out_cand_surrounding_read_matrix, out_cand_surrounding_baseq_matrix, out_cand_surrounding_mapq_matrix = [], [], []
    max_depth = 0
    max_surrounding_depth = 0
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
        cand_pos, cand_edge_columns, cand_edge_matrix, cand_pair_columns, cand_pair_route, cand_group_pos, \
        cand_read_matrix, cand_baseq_matrix, cand_mapq_matrix, depth, \
        cand_surrounding_read_matrix, cand_surrounding_baseq_matrix, \
        cand_surrounding_mapq_matrix, surrounding_depth = extract_pileups_batch(
            samfile, sub_sub_groups, max_coverage, pileup_length)
        if cand_edge_matrix is not None:
            out_cand_pos.extend(cand_pos)
            out_cand_edge_columns.extend(cand_edge_columns)
            out_cand_edge_matrix.extend(cand_edge_matrix)
            out_cand_pair_columns.extend(cand_pair_columns)
            out_cand_pair_route.extend(cand_pair_route)
            out_cand_group_pos.extend(cand_group_pos)
            out_cand_read_matrix.extend(cand_read_matrix)
            out_cand_baseq_matrix.extend(cand_baseq_matrix)
            out_cand_mapq_matrix.extend(cand_mapq_matrix)
            max_depth = depth if depth > max_depth else max_depth
            out_cand_surrounding_read_matrix.extend(cand_surrounding_read_matrix)
            out_cand_surrounding_baseq_matrix.extend(cand_surrounding_baseq_matrix)
            out_cand_surrounding_mapq_matrix.extend(cand_surrounding_mapq_matrix)
            max_surrounding_depth = surrounding_depth if surrounding_depth > max_surrounding_depth else max_surrounding_depth
    return out_cand_pos, out_cand_edge_columns, out_cand_edge_matrix, out_cand_pair_columns, out_cand_pair_route, \
           out_cand_group_pos, out_cand_read_matrix, out_cand_baseq_matrix, out_cand_mapq_matrix, max_depth, \
           out_cand_surrounding_read_matrix, out_cand_surrounding_baseq_matrix, \
           out_cand_surrounding_mapq_matrix, max_surrounding_depth


def Run(args):
    pileup_vcf = args.pileup_vcf
    quality_threshold = args.min_quality
    adjacent_size = args.adjacent_size
    pileup_length = args.pileup_length
    support_quality = args.support_quality

    assert pileup_length % 2 == 1  # 必须为奇数

    groups_dict = select_snp(
        vcf_file=pileup_vcf, quality_threshold=quality_threshold, adjacent_size=adjacent_size,
        support_quality=support_quality)
    for k in groups_dict.keys():
        out_cand_pos, out_cand_edge_columns, out_cand_edge_matrix, out_cand_pair_columns, out_cand_pair_route = [], [], [], [], []
        out_cand_group_pos, out_cand_read_matrix, out_cand_baseq_matrix, out_cand_mapq_matrix = [], [], [], []
        out_cand_surrounding_read_matrix, out_cand_surrounding_baseq_matrix, out_cand_surrounding_mapq_matrix = [], [], []
        max_depth = 0
        max_surrounding_depth = 0
        chromosome_groups = groups_dict[k]
        total_threads = args.threads
        step = math.ceil(len(chromosome_groups) / total_threads)
        divided_groups = [chromosome_groups[dt:dt + step] for dt in
                          range(0, len(chromosome_groups), step)]  # divided for multiple threads

        with concurrent.futures.ProcessPoolExecutor(max_workers=total_threads) as executor:
            if len(divided_groups) == total_threads:
                signals = [
                    executor.submit(extract_pileups_group, args.bam, divided_groups[thread_id], args.max_coverage,
                                    pileup_length) for thread_id in range(0, total_threads)]
            else:
                # 如果chromosome_groups恰好被total_threads整除，则divided_groups长度只有total_threads-1，会出现数组越界错误。
                signals = [
                    executor.submit(extract_pileups_group, args.bam, divided_groups[thread_id], args.max_coverage,
                                    pileup_length) for thread_id in range(0, len(divided_groups))]
            for sig in concurrent.futures.as_completed(signals):
                if sig.exception() is None:
                    # get the results
                    [cand_pos, cand_edge_columns, cand_edge_matrix, cand_pair_columns, cand_pair_route, cand_group_pos,
                     cand_read_matrix, cand_baseq_matrix, cand_mapq_matrix, depth,
                     cand_surrounding_read_matrix, cand_surrounding_baseq_matrix,
                     cand_surrounding_mapq_matrix, surrounding_depth] = sig.result()
                    if len(cand_edge_matrix) > 0:
                        out_cand_pos.extend(cand_pos)
                        out_cand_edge_columns.extend(cand_edge_columns)
                        out_cand_edge_matrix.extend(cand_edge_matrix)
                        out_cand_pair_columns.extend(cand_pair_columns)
                        out_cand_pair_route.extend(cand_pair_route)
                        out_cand_group_pos.extend(cand_group_pos)
                        out_cand_read_matrix.extend(cand_read_matrix)
                        out_cand_baseq_matrix.extend(cand_baseq_matrix)
                        out_cand_mapq_matrix.extend(cand_mapq_matrix)
                        max_depth = depth if depth > max_depth else max_depth
                        out_cand_surrounding_read_matrix.extend(cand_surrounding_read_matrix)
                        out_cand_surrounding_baseq_matrix.extend(cand_surrounding_baseq_matrix)
                        out_cand_surrounding_mapq_matrix.extend(cand_surrounding_mapq_matrix)
                        max_surrounding_depth = surrounding_depth if surrounding_depth > max_surrounding_depth else max_surrounding_depth
                else:
                    sys.stderr.write("ERROR: " + str(sig.exception()) + "\n")
                sig._result = None  # python issue 27144

        # 排序
        out_cand_pos = np.array(out_cand_pos)  # [N,]
        new_cand_pos = [int(v.split(':')[1]) for v in out_cand_pos]
        new_index = np.argsort(new_cand_pos)
        out_cand_pos = out_cand_pos[new_index]
        out_cand_edge_columns = np.array(out_cand_edge_columns)[new_index]
        out_cand_pair_columns = np.array(out_cand_pair_columns)[new_index]
        out_cand_group_pos = np.array(out_cand_group_pos)[new_index]

        out_cand_edge_matrix = np.concatenate(out_cand_edge_matrix)[new_index]
        out_cand_pair_route = np.concatenate(out_cand_pair_route)[new_index]

        out_cand_read_matrix = [
            np.expand_dims(np.pad(a, ((0, max_depth - a.shape[0]), (0, 0)), 'constant', constant_values=-2), 0) for a in
            out_cand_read_matrix]
        out_cand_read_matrix = np.concatenate(out_cand_read_matrix)[new_index]

        out_cand_baseq_matrix = [
            np.expand_dims(np.pad(a, ((0, max_depth - a.shape[0]), (0, 0)), 'constant', constant_values=-2), 0) for a in
            out_cand_baseq_matrix]
        out_cand_baseq_matrix = np.concatenate(out_cand_baseq_matrix)[new_index]

        out_cand_mapq_matrix = [
            np.expand_dims(np.pad(a, ((0, max_depth - a.shape[0]), (0, 0)), 'constant', constant_values=-2), 0) for a in
            out_cand_mapq_matrix]
        out_cand_mapq_matrix = np.concatenate(out_cand_mapq_matrix)[new_index]

        out_cand_surrounding_read_matrix = [
            np.expand_dims(np.pad(a, ((0, max_surrounding_depth - a.shape[0]), (0, 0)), 'constant', constant_values=-2),
                           0) for a in
            out_cand_surrounding_read_matrix]
        out_cand_surrounding_read_matrix = np.concatenate(out_cand_surrounding_read_matrix)[new_index]

        out_cand_surrounding_baseq_matrix = [
            np.expand_dims(np.pad(a, ((0, max_surrounding_depth - a.shape[0]), (0, 0)), 'constant', constant_values=-2),
                           0) for a in
            out_cand_surrounding_baseq_matrix]
        out_cand_surrounding_baseq_matrix = np.concatenate(out_cand_surrounding_baseq_matrix)[new_index]

        out_cand_surrounding_mapq_matrix = [
            np.expand_dims(np.pad(a, ((0, max_surrounding_depth - a.shape[0]), (0, 0)), 'constant', constant_values=-2),
                           0) for a in
            out_cand_surrounding_mapq_matrix]
        out_cand_surrounding_mapq_matrix = np.concatenate(out_cand_surrounding_mapq_matrix)[new_index]

        TABLE_FILTERS = tables.Filters(complib='blosc:lz4hc', complevel=5)
        output = args.output + '/' + k + '.bin'
        table_file = tables.open_file(output, mode='w')
        int_atom = tables.Atom.from_dtype(np.dtype('int32'))
        string_atom = tables.StringAtom(itemsize=30 * (2 * adjacent_size))
        table_file.create_earray(
            where='/', name='edge_matrix', atom=int_atom, shape=[0, 25, 2 * adjacent_size])
        table_file.create_earray(
            where='/', name='pair_route', atom=int_atom, shape=[0, 25, 2 * adjacent_size])
        table_file.create_earray(
            where='/', name='read_matrix', atom=int_atom, shape=[0, max_depth, 2 * adjacent_size + 1])
        table_file.create_earray(
            where='/', name='base_quality_matrix', atom=int_atom, shape=[0, max_depth, 2 * adjacent_size + 1])
        table_file.create_earray(
            where='/', name='mapping_quality_matrix', atom=int_atom, shape=[0, max_depth, 2 * adjacent_size + 1])
        table_file.create_earray(
            where='/', name='position', atom=string_atom, shape=(0, 1), filters=TABLE_FILTERS)
        # TODO: 可修改pileup特征的长度
        table_file.create_earray(
            where='/', name='surrounding_read_matrix', atom=int_atom, shape=[0, max_surrounding_depth, pileup_length])
        table_file.create_earray(
            where='/', name='surrounding_base_quality_matrix', atom=int_atom,
            shape=[0, max_surrounding_depth, pileup_length])
        table_file.create_earray(
            where='/', name='surrounding_mapping_quality_matrix', atom=int_atom,
            shape=[0, max_surrounding_depth, pileup_length])

        table_file.create_earray(where='/', name='edge_columns', atom=string_atom, shape=(0, adjacent_size * 2),
                                 filters=TABLE_FILTERS)
        table_file.create_earray(where='/', name='pair_columns', atom=string_atom, shape=(0, adjacent_size * 2),
                                 filters=TABLE_FILTERS)
        table_file.create_earray(where='/', name='group_positions', atom=string_atom, shape=(0, adjacent_size * 2 + 1),
                                 filters=TABLE_FILTERS)

        table_file.root.edge_matrix.append(out_cand_edge_matrix)
        table_file.root.pair_route.append(out_cand_pair_route)
        table_file.root.read_matrix.append(out_cand_read_matrix)
        table_file.root.base_quality_matrix.append(out_cand_baseq_matrix)
        table_file.root.mapping_quality_matrix.append(out_cand_mapq_matrix)
        table_file.root.surrounding_read_matrix.append(out_cand_surrounding_read_matrix)
        table_file.root.surrounding_base_quality_matrix.append(out_cand_surrounding_baseq_matrix)
        table_file.root.surrounding_mapping_quality_matrix.append(out_cand_surrounding_mapq_matrix)

        table_file.root.position.append(np.array(out_cand_pos).reshape(-1, 1))
        table_file.root.edge_columns.append(
            np.array(out_cand_edge_columns).reshape(-1, adjacent_size * 2))
        table_file.root.pair_columns.append(
            np.array(out_cand_pair_columns).reshape(-1, adjacent_size * 2))
        table_file.root.group_positions.append(
            np.array(out_cand_group_pos).reshape(-1, adjacent_size * 2 + 1))
        table_file.close()
    print(time.strftime("[%a %b %d %H:%M:%S %Y] Done.", time.localtime()))


def main():
    parser = ArgumentParser(description="Group adjacent SNPs")
    parser.add_argument("--pileup_vcf", type=str, required=True,
                        help="Input the variants called by pileup model, required.")
    parser.add_argument("--bam", type=str, required=True,
                        help="Input the bam file, required.")
    parser.add_argument("--output", type=str, required=True,
                        help="Input the output directory, required.")
    parser.add_argument("--adjacent_size", type=int, default=5,
                        help="The range of SNPS around the candidate sites to be considered as groups, default: %(default)d")
    parser.add_argument("--pileup_length", type=int, default=11,
                        help="The length of surrounding pileup feature in haplotype model, default: %(default)d")
    parser.add_argument("--min_quality", type=int, default=15,
                        help="The minimum quality of hetezygous site used as edge model.")
    parser.add_argument('--support_quality', default=19, type=float,
                        help='Min quality of hetezygous SNP used for making up of group. (default: %(default)f)')
    parser.add_argument("--max_coverage", type=int, default=150,
                        help="The maximum coverage of each position, default: %(default)d")
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
