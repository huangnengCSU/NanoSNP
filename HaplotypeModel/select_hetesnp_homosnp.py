import argparse
from collections import defaultdict

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


def select_snp(vcf_file, quality_threshold, adjacent_size, support_quality=15):
    # 候选位点的质量低于quality_threshold, 用于构造group的杂合位点只要质量超过support_quality才行
    header = []
    num_groups = 0
    groups_dict = {}
    contig_dict = defaultdict(defaultdict)
    row_count = 0
    no_vcf_output = True
    success_failed_cnt_dict = {}
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
            genotype = columns[9].split(':')[0].replace(
                '|', '/')  # 0/0, 1/1, 0/1

            # 保留杂合位点和低质量的纯合位点
            if (genotype == '0/0' and quality >= quality_threshold) or (
                    genotype == '1/1' and quality >= quality_threshold):
                continue

            contig_dict[ctg_name][pos] = (genotype, quality)  # GT, Q
            no_vcf_output = False
        if row_count == 0:
            print("[WARNING] No vcf file found, please check the setting")
        if no_vcf_output:
            print("[WARNING] No variant found, please check the setting")
        contigs_order = major_contigs_order + list(contig_dict.keys())
        contigs_order_list = sorted(
            contig_dict.keys(), key=lambda x: contigs_order.index(x))
        for contig in contigs_order_list:
            all_pos = sorted(contig_dict[contig].keys())
            for pos_i in range(len(all_pos)):
                pos = all_pos[pos_i]
                if contig_dict[contig][pos][1] < quality_threshold:
                    adjacent_pos = []
                    left_adjacent_pos, right_adjacent_pos = [], []
                    l_count, r_count = adjacent_size, adjacent_size
                    pos_il = pos_i - 1
                    pos_ir = pos_i + 1
                    while pos_il >= 0:
                        if contig_dict[contig][all_pos[pos_il]][1] >= support_quality and \
                                contig_dict[contig][all_pos[pos_il]][0] == '0/1':
                            left_adjacent_pos.append(all_pos[pos_il])
                            l_count -= 1
                        if l_count <= 0:
                            break
                        pos_il -= 1
                    left_adjacent_pos = left_adjacent_pos[::-1]
                    if len(left_adjacent_pos) != adjacent_size:
                        continue
                    while pos_ir < len(all_pos):
                        if contig_dict[contig][all_pos[pos_ir]][1] >= support_quality and \
                                contig_dict[contig][all_pos[pos_ir]][0] == '0/1':
                            right_adjacent_pos.append(all_pos[pos_ir])
                            r_count -= 1
                        if r_count <= 0:
                            break
                        pos_ir += 1
                    if len(right_adjacent_pos) != adjacent_size:
                        success_failed_cnt_dict[contig] = success_failed_cnt_dict.get(contig, [0, 0])  # [success, fail]
                        success_failed_cnt_dict[contig][1] += 1
                        continue
                    adjacent_pos = left_adjacent_pos + [pos] + right_adjacent_pos
                    queue = []
                    assert len(adjacent_pos) == adjacent_size * 2 + 1
                    for p in adjacent_pos:
                        dict_item = contig_dict[contig][p]
                        queue.append(
                            SNPItem(contig, p, dict_item[0], dict_item[1]))
                    if contig not in groups_dict.keys():
                        groups_dict[contig] = []
                    groups_dict[contig].append(queue)
                    success_failed_cnt_dict[contig] = success_failed_cnt_dict.get(contig, [0, 0])  # [success, fail]
                    success_failed_cnt_dict[contig][0] += 1
                    num_groups += 1
                else:
                    continue
    # print('total groups:', num_groups)
    # for contig in contigs_order_list:
    #     print("contig:%s\tsuccess cnt:%d\tfailed cnt:%d\tratio=%f\n" % (
    #         contig, success_failed_cnt_dict[contig][0], success_failed_cnt_dict[contig][1],
    #         success_failed_cnt_dict[contig][1] / (
    #                     success_failed_cnt_dict[contig][0] + success_failed_cnt_dict[contig][1]+1e-9)))
    return groups_dict


def main():
    parse = argparse.ArgumentParser()
    parse.add_argument(
        '--pileup_vcf', help='Input pileup vcf file', required=True)
    parse.add_argument(
        '--quality', help='Quality threshold for snp selection. (default: %(default)f)', default=15, type=float)
    parse.add_argument('--support_quality',
                       help='Min quality of hetezygous SNP used for making up of group. (default: %(default)f)',
                       default=19, type=float)
    parse.add_argument(
        '--adjacent_size',
        help='The number of selected heterozygous sites around the candidate site. (default: %(default)s)', default=5,
        type=int)
    args = parse.parse_args()
    groups_dict = select_snp(vcf_file=args.pileup_vcf, quality_threshold=args.quality, adjacent_size=args.adjacent_size,
                             support_quality=args.support_quality)


if __name__ == '__main__':
    main()
