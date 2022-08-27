import numpy as np
import pysam
import sys
import pandas as pd
from itertools import product

base_to_int = {'A': 1, 'C': 2, 'G': 3, 'T': 4}
int_to_base = {1: 'A', 2: 'C', 3: 'G', 4: 'T', -1: 'D', 0: 'B'}
edges_label = list(map(''.join, product(
    ['A', 'C', 'G', 'T', 'D'], ['A', 'C', 'G', 'T', 'D'])))


class Edge:
    def __init__(self, s, t):
        self.source = s
        self.target = t
        self.weight = 1

    def add_weight(self, w):
        self.weight += w


def single_group_pileup_haplotype_feature(samfile, groups, max_coverage, adjacent_size, pileup_flanking_size):
    candidate_positions = []
    haplotype_positions, haplotype_sequences, haplotype_baseq, haplotype_mapq = [], [], [], []
    haplotype_hap, pileup_hap = [], []
    pileup_sequences, pileup_baseq, pileup_mapq = [], [], []
    max_haplotype_depth, max_pileup_depth = 0, 0
    haplotype_depths = []
    pileup_depths = []
    positions = []
    ctgname = groups[0][0].ctgname
    for g in groups:
        assert g[0].ctgname == ctgname
        for snpitem in g:
            positions.append(int(snpitem.position))
    positions = sorted(list(set(positions)))
    failed_positions = []
    for pileupcolumn in samfile.pileup(ctgname, positions[0], positions[-1], min_base_quality=0, min_mapping_quality=0):
        if pileupcolumn.pos + 1 < positions[0]:
            continue
        if pileupcolumn.pos + 1 > positions[-1]:
            break
        if pileupcolumn.pos + 1 not in positions:
            continue
        if pileupcolumn.n > max_coverage:
            failed_positions.append(pileupcolumn.pos + 1)

    # 存在某些位点覆盖度超过阈值，则过滤掉这些group
    if len(failed_positions) > 0:
        new_groups = []
        for g in groups:
            failed = False
            for snpitem in g:
                if snpitem.position in failed_positions:
                    failed = True
                    break
            if not failed:
                new_groups.append(g)
        groups = new_groups
    
    ## TODO: 由于覆盖度过高的位点，直接被过滤掉，真实突变位点会变成FN，如何把这部分位点进行召回？

    # TODO: 实际数据中，会存在过滤后的group为空的情况，如何处理？
    if len(groups) == 0:
        print("No group is left after filtering")
        return candidate_positions,  haplotype_positions, haplotype_sequences, haplotype_baseq, haplotype_mapq, haplotype_hap, \
               max_haplotype_depth, pileup_sequences, pileup_baseq, pileup_mapq, pileup_hap, max_pileup_depth

    try:
        """
        extend positions 存储了每个候选位点周围11mer的position，以及候选位点周围高质量Hete位点的position
        """
        extend_positions = []
        for g in groups:
            for tidx in range(len(g)):
                snpitem = g[tidx]
                if tidx == len(g) // 2:
                    ## pileup window size = 2 * pileup_flanking_size + 1
                    extend_positions.extend(
                        list(range(int(snpitem.position) - pileup_flanking_size, int(snpitem.position) + pileup_flanking_size+1)))  # pileup的position
                else:
                    extend_positions.append(int(snpitem.position))  # haplotype的position
        extend_positions = sorted(list(set(extend_positions)))

        SNP_mat = {}
        hap_mat = {}
        base_quality_mat = {}
        mapping_quality_mat = {}
        for pileupcolumn in samfile.pileup(ctgname, extend_positions[0], extend_positions[-1], min_base_quality=0,
                                           min_mapping_quality=0):
            if pileupcolumn.pos + 1 < extend_positions[0]:
                continue
            if pileupcolumn.pos + 1 > extend_positions[-1]:
                break
            if pileupcolumn.pos + 1 not in extend_positions:
                continue
            assert pileupcolumn.n <= max_coverage
            i = extend_positions.index(pileupcolumn.pos + 1)
            # print("coverage at base %s = %s" % (pileupcolumn.pos, pileupcolumn.n))
            for pileupread in pileupcolumn.pileups:
                qname = pileupread.alignment.query_name
                tag = 3 # unphased tag
                if pileupread.alignment.has_tag("HP"):
                    tag = pileupread.alignment.get_tag("HP")    # if tag exists, it equals to 1 or 2.
                assert tag == 1 or tag == 2 or tag == 3
                if qname not in SNP_mat.keys():
                    SNP_mat[qname] = [0] * len(extend_positions)
                    hap_mat[qname] = [0] * len(extend_positions)
                    base_quality_mat[qname] = [0] * len(extend_positions)
                    mapping_quality_mat[qname] = [0] * len(extend_positions)
                    row = SNP_mat[qname]
                    hp = hap_mat[qname]
                    bq_row = base_quality_mat[qname]
                    mq_row = mapping_quality_mat[qname]
                else:
                    row = SNP_mat[qname]
                    hap = hap_mat[qname]
                    bq_row = base_quality_mat[qname]
                    mq_row = mapping_quality_mat[qname]
                if not pileupread.is_del and not pileupread.is_refskip:
                    qbase = pileupread.alignment.query_sequence[pileupread.query_position]
                    row[i] = base_to_int[str.upper(qbase)]
                    hp[i] = tag
                    base_q = pileupread.alignment.query_qualities[pileupread.query_position]
                    bq_row[i] = base_q
                    map_q = pileupread.alignment.mapping_quality
                    mq_row[i] = map_q
                elif pileupread.is_del:
                    # base is -1, base_q is 0, and map_q is normal.
                    row[i] = -1  # deletion
                    hp[i] = tag
                    map_q = pileupread.alignment.mapping_quality
                    mq_row[i] = map_q
        df = pd.DataFrame(SNP_mat, index=extend_positions).T
        hap_df = pd.DataFrame(hap_mat, index=extend_positions).T
        baseq_df = pd.DataFrame(base_quality_mat, index=extend_positions).T
        mapq_df = pd.DataFrame(mapping_quality_mat, index=extend_positions).T

        for g in groups:
            ctg = g[0].ctgname
            gpos = [int(snpitem.position) for snpitem in g]
            gdf = df[gpos]
            # filter gdf: 每条reads必须跨过group的中的中间位置即可。如果要求reads跨过所有点，则筛选出来的覆盖度很低
            filter_gdf_lines = []
            for i in range(gdf.shape[0]):
                mid_gpos = gpos[len(gpos) // 2]
                if gdf.iloc[i][mid_gpos] != 0:
                    filter_gdf_lines.append(i)
                # if gdf.iloc[i][gpos[0]] != 0 and gdf.iloc[i][gpos[-1]] != 0:
                #     filter_gdf_lines.append(i)

            haplotype_read_df_gpos = df[gpos].iloc[filter_gdf_lines]  # [depth, adjacent_size*2+1]
            haplotype_hap_df_gpos = hap_df[gpos].iloc[filter_gdf_lines]   # [depth, adjacent_size*2+1]
            haplotype_baseq_df_gpos = baseq_df[gpos].iloc[filter_gdf_lines]  # [depth, adjacent_size*2+1]
            haplotype_mapq_df_gpos = mapq_df[gpos].iloc[filter_gdf_lines]  # [depth, adjacent_size*2+1]

            ###: 增加四个df根据hap排序,根据中间位置的hap值进行排序
            mid_col_name = haplotype_hap_df_gpos.columns[len(haplotype_hap_df_gpos.columns) // 2]
            sort_haplotype_hap_df_gpos = haplotype_hap_df_gpos.sort_values(by=mid_col_name)
            rearrange_index = sort_haplotype_hap_df_gpos.index
            haplotype_read_df_gpos = haplotype_read_df_gpos.loc[rearrange_index]
            haplotype_hap_df_gpos = haplotype_hap_df_gpos.loc[rearrange_index]
            haplotype_baseq_df_gpos = haplotype_baseq_df_gpos.loc[rearrange_index]
            haplotype_mapq_df_gpos = haplotype_mapq_df_gpos.loc[rearrange_index]


            candidate_positions.append(ctg + ":" + str(gpos[len(gpos) // 2]))
            haplotype_positions.append([ctg + ":" + str(gpos_v) for gpos_v in gpos])
            haplotype_sequences.append(haplotype_read_df_gpos.values)
            haplotype_hap.append(haplotype_hap_df_gpos.values)
            haplotype_baseq.append(haplotype_baseq_df_gpos.values)
            haplotype_mapq.append(haplotype_mapq_df_gpos.values)
            haplotype_depths.append(haplotype_read_df_gpos.shape[0])

            # 候选位点周围11-mer的reads比对信息
            gpos = list(range(int(g[len(g) // 2].position) - pileup_flanking_size, int(g[len(g) // 2].position) + pileup_flanking_size + 1))
            gdf = df[gpos]
            # filter gdf: 每条reads必须跨过group的中的中间位置即可。如果要求reads跨过所有点，则筛选出来的覆盖度很低
            ### 此处的gdf与haplotype的gdf不同了
            filter_gdf_lines = []
            for i in range(gdf.shape[0]):
                mid_gpos = gpos[len(gpos) // 2]
                if gdf.iloc[i][mid_gpos] != 0:
                    filter_gdf_lines.append(i)
                # if gdf.iloc[i][gpos[0]] != 0 and gdf.iloc[i][gpos[-1]] != 0:
                #     filter_gdf_lines.append(i)
            pileup_read_df_gpos = df[gpos].iloc[filter_gdf_lines]  # [depth, 11]
            pileup_hap_df_gpos = hap_df[gpos].iloc[filter_gdf_lines]    # [depth, 11]
            pileup_baseq_df_gpos = baseq_df[gpos].iloc[filter_gdf_lines]  # [depth, 11]
            pileup_mapq_df_gpos = mapq_df[gpos].iloc[filter_gdf_lines]  # [depth, 11]

            ###: 增加四个df根据hap排序,根据中间位置的hap值进行排序
            mid_col_name = pileup_hap_df_gpos.columns[len(pileup_hap_df_gpos.columns) // 2]
            sort_pileup_hap_df_gpos = pileup_hap_df_gpos.sort_values(by=mid_col_name)
            rearrange_index = sort_pileup_hap_df_gpos.index
            pileup_read_df_gpos = pileup_read_df_gpos.loc[rearrange_index]
            pileup_hap_df_gpos = pileup_hap_df_gpos.loc[rearrange_index]
            pileup_baseq_df_gpos = pileup_baseq_df_gpos.loc[rearrange_index]
            pileup_mapq_df_gpos = pileup_mapq_df_gpos.loc[rearrange_index]


            pileup_sequences.append(pileup_read_df_gpos.values)
            pileup_hap.append(pileup_hap_df_gpos.values)
            pileup_baseq.append(pileup_baseq_df_gpos.values)
            pileup_mapq.append(pileup_mapq_df_gpos.values)
            pileup_depths.append(pileup_read_df_gpos.shape[0])

        max_haplotype_depth = max(haplotype_depths)
        max_pileup_depth = max(pileup_depths)
        return candidate_positions,  haplotype_positions, haplotype_sequences, haplotype_baseq, haplotype_mapq, haplotype_hap, \
               max_haplotype_depth, pileup_sequences, pileup_baseq, pileup_mapq, pileup_hap, max_pileup_depth
    except:
        print("try..except.. break")
        return candidate_positions,  haplotype_positions, haplotype_sequences, haplotype_baseq, haplotype_mapq, haplotype_hap, \
               max_haplotype_depth, pileup_sequences, pileup_baseq, pileup_mapq, pileup_hap, max_pileup_depth


if __name__ == "__main__":
    bamfile_path = sys.argv[1]
    ctg = input("please input the contig name:\n")
    positions = input("please input the positions on the contig:\n")
    positions = [int(v) for v in positions.strip().split(' ')]
    samfile = pysam.AlignmentFile(bamfile_path, 'r')
    # edge_dict = extract_pileups(samfile, ctg, positions)
    # with open('graph.csv', 'w') as fout:
    #     fout.write('source,target,weight\n')
    #     for k in edge_dict.keys():
    #         edge = edge_dict[k]
    #         fout.write('%s,%s,%d\n' % (edge.source, edge.target, edge.weight))
