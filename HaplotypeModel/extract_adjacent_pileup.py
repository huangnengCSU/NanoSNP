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


def extract_pileups(samfile, ctg, positions):
    SNP_mat = {}
    for pileupcolumn in samfile.pileup(ctg, positions[0], positions[-1], min_base_quality=0, min_mapping_quality=0):
        if pileupcolumn.pos + 1 < positions[0]:
            continue
        if pileupcolumn.pos + 1 > positions[-1]:
            break
        if pileupcolumn.pos + 1 not in positions:
            continue
        i = positions.index(pileupcolumn.pos + 1)
        # print("coverage at base %s = %s" % (pileupcolumn.pos, pileupcolumn.n))
        for pileupread in pileupcolumn.pileups:
            qname = pileupread.alignment.query_name
            if qname not in SNP_mat.keys():
                SNP_mat[qname] = [0] * len(positions)
                row = SNP_mat[qname]
            else:
                row = SNP_mat[qname]
            if not pileupread.is_del and not pileupread.is_refskip:
                qbase = pileupread.alignment.query_sequence[pileupread.query_position]
                row[i] = base_to_int[str.upper(qbase)]
            elif pileupread.is_del:
                row[i] = -1  # deletion
    df = pd.DataFrame(SNP_mat, index=positions).T
    # print(df)

    # convert to a graph according to SNP_mat

    edge_dict = {}  # key is source-target, value is Edge

    # edges: {A,C,G,T,D}x{A,C,G,T,D}
    edge_columns = [ctg + ':' + str(positions[ti]) + '-' + str(positions[ti + 1])
                    for ti in range(len(positions) - 1)]
    edge_df = pd.DataFrame(np.zeros(
        shape=(25, len(positions) - 1)), index=edges_label, columns=edge_columns)
    for i in range(df.shape[0]):
        row = df.iloc[i]
        for j in range(len(row) - 1):
            s_pos_idx = df.columns[j]
            t_pos_idx = df.columns[j + 1]
            s = row[s_pos_idx]
            t = row[t_pos_idx]
            if int(s) == 0 or int(t) == 0:
                continue
            source_node_name = int_to_base[int(s)]
            target_node_name = int_to_base[int(t)]
            edge_df.loc[source_node_name + target_node_name][j] += 1
    # print(edge_df)
    '''
    ### used for Gephi ###
    for i in range(df.shape[0]):
        row = df.iloc[i]
        for j in range(len(row) - 1):
            s_pos_idx = df.columns[j]
            t_pos_idx = df.columns[j + 1]
            s = row[s_pos_idx]
            t = row[t_pos_idx]
            if int(s) == 0 or int(s) == -1 or int(t) == 0 or int(t) == -1:
                continue
            source_node_name = str(s_pos_idx) + '_' + int_to_base[int(s)]
            target_node_name = str(t_pos_idx) + '_' + int_to_base[int(t)]
            if (source_node_name, target_node_name) in edge_dict.keys():
                edge_dict[(source_node_name, target_node_name)].add_weight(1)
            else:
                edge = Edge(source_node_name, target_node_name)
                edge_dict[(source_node_name, target_node_name)] = edge
    '''
    return edge_df


def extract_pileups_batch(samfile, groups, max_coverage):
    cand_pos, cand_edge_columns, cand_edge_matrix, cand_pair_columns, cand_pair_route = [], [], [], [], []
    cand_group_pos, cand_read_matrix, cand_baseq_matrix, cand_mapq_matrix = [], [], [], []
    cand_surrounding_read_matrix, cand_surrounding_baseq_matrix, cand_surrounding_mapq_matrix = [], [], []
    max_depth = 0
    max_surrounding_depth = 0
    depth = []
    surrounding_depth = []
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
        # TODO: 可优化，减少一轮扫描，在前一轮时就将positions存储
        positions = []
        for g in new_groups:
            for snpitem in g:
                positions.append(int(snpitem.position))
        positions = sorted(list(set(positions)))
        groups = new_groups
    

    # TODO: 实际数据中，会存在过滤后的group为空的情况，如何处理？
    if len(groups)==0:
        print("No group is left after filtering")
        return cand_pos, cand_edge_columns, cand_edge_matrix, cand_pair_columns, cand_pair_route, \
           cand_group_pos, cand_read_matrix, cand_baseq_matrix, cand_mapq_matrix, max_depth, \
           cand_surrounding_read_matrix, cand_surrounding_baseq_matrix, \
           cand_surrounding_mapq_matrix, max_surrounding_depth
    
    try:
        """
        extend positions 存储了每个候选位点周围11mer的position，以及候选位点周围高质量Hete位点的position
        """
        extend_positions = []
        for g in groups:
            for tidx in range(len(g)):
                snpitem = g[tidx]
                if tidx == len(g) // 2:
                    # TODO: 目前写死了11个长度，后期增加参数设置
                    extend_positions.extend(list(range(int(snpitem.position) - 5, int(snpitem.position) + 6)))  # pileup的position
                else:
                    extend_positions.append(int(snpitem.position))  # haplotype的position
        extend_positions = sorted(list(set(extend_positions)))

        SNP_mat = {}
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
                if qname not in SNP_mat.keys():
                    SNP_mat[qname] = [0] * len(extend_positions)
                    base_quality_mat[qname] = [0] * len(extend_positions)
                    mapping_quality_mat[qname] = [0] * len(extend_positions)
                    row = SNP_mat[qname]
                    bq_row = base_quality_mat[qname]
                    mq_row = mapping_quality_mat[qname]
                else:
                    row = SNP_mat[qname]
                    bq_row = base_quality_mat[qname]
                    mq_row = mapping_quality_mat[qname]
                if not pileupread.is_del and not pileupread.is_refskip:
                    qbase = pileupread.alignment.query_sequence[pileupread.query_position]
                    row[i] = base_to_int[str.upper(qbase)]
                    base_q = pileupread.alignment.query_qualities[pileupread.query_position]
                    bq_row[i] = base_q
                    map_q = pileupread.alignment.mapping_quality
                    mq_row[i] = map_q
                elif pileupread.is_del:
                    # base is -1, base_q is 0, and map_q is normal.
                    row[i] = -1  # deletion
                    map_q = pileupread.alignment.mapping_quality
                    mq_row[i] = map_q
        df = pd.DataFrame(SNP_mat, index=extend_positions).T
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
            gdf = gdf.iloc[filter_gdf_lines]
            # TODO: edge_columns,edge_df可以删掉，没用到该特征
            edge_columns = [ctg + ':' + str(gpos[ti]) + '-' + str(gpos[ti + 1])
                            for ti in range(len(gpos) - 1)]
            edge_df = pd.DataFrame(
                np.zeros(shape=(25, len(gpos) - 1)), index=edges_label, columns=edge_columns)
            for i in range(gdf.shape[0]):
                row = gdf.iloc[i]
                for j in range(len(row) - 1):
                    s_pos_idx = gdf.columns[j]
                    t_pos_idx = gdf.columns[j + 1]
                    s = row[s_pos_idx]
                    t = row[t_pos_idx]
                    if int(s) == 0 or int(t) == 0:
                        continue
                    source_node_name = int_to_base[int(s)]
                    target_node_name = int_to_base[int(t)]
                    edge_df.loc[source_node_name + target_node_name][j] += 1
            
            # TODO: pair_columns,pair_route 可以删掉，没用到该特征    
            pair_columns = []
            for ti in range(len(gpos)):
                if ti != len(gpos) // 2:
                    pair_columns.append(
                        ctg + ':' + str(gpos[ti]) + '-' + str(gpos[len(gpos) // 2]))
            pair_route = pd.DataFrame(
                np.zeros(shape=(25, len(gpos) - 1)), index=edges_label, columns=pair_columns)
            for i in range(gdf.shape[0]):
                row = gdf.iloc[i]
                for j in range(len(row)):
                    if j != len(gpos) // 2:
                        s_pos_idx = gdf.columns[j]
                        t_pos_idx = gdf.columns[len(gpos) // 2]
                        s = row[s_pos_idx]
                        t = row[t_pos_idx]
                        if int(s) == 0 or int(t) == 0:
                            continue
                        source_node_name = int_to_base[int(s)]
                        target_node_name = int_to_base[int(t)]
                        col = ctg + ':' + str(s_pos_idx) + '-' + str(t_pos_idx)
                        pair_route.loc[source_node_name +
                                    target_node_name][col] += 1

            read_df_gpos = df[gpos].iloc[filter_gdf_lines]  # [depth, adjacent_size*2+1]
            mapq_df_gpos = mapq_df[gpos].iloc[filter_gdf_lines]  # [depth, adjacent_size*2+1]
            baseq_df_gpos = baseq_df[gpos].iloc[filter_gdf_lines]  # [depth, adjacent_size*2+1]

            cand_pos.append(ctg + ":" + str(gpos[len(gpos) // 2]))
            cand_edge_columns.append(edge_columns)
            cand_pair_columns.append(pair_columns)
            cand_edge_matrix.append(np.expand_dims(edge_df.values, 0))
            cand_pair_route.append(np.expand_dims(pair_route.values, 0))
            cand_group_pos.append([ctg + ":" + str(gpos_v) for gpos_v in gpos])
            cand_read_matrix.append(read_df_gpos.values)
            cand_mapq_matrix.append(mapq_df_gpos.values)
            cand_baseq_matrix.append(baseq_df_gpos.values)
            depth.append(read_df_gpos.shape[0])

            # 候选位点周围11-mer的reads比对信息
            gpos = list(range(int(g[len(g) // 2].position) - 5, int(g[len(g) // 2].position) + 6))
            gdf = df[gpos]
            # filter gdf: 每条reads必须跨过group的中的中间位置即可。如果要求reads跨过所有点，则筛选出来的覆盖度很低
            filter_gdf_lines = []
            for i in range(gdf.shape[0]):
                mid_gpos = gpos[len(gpos) // 2]
                if gdf.iloc[i][mid_gpos] != 0:
                    filter_gdf_lines.append(i)
                # if gdf.iloc[i][gpos[0]] != 0 and gdf.iloc[i][gpos[-1]] != 0:
                #     filter_gdf_lines.append(i)
            surrounding_read_df_gpos = df[gpos].iloc[filter_gdf_lines]  # [depth, 11]
            surrounding_mapq_df_gpos = mapq_df[gpos].iloc[filter_gdf_lines]  # [depth, 11]
            surrounding_baseq_df_gpos = baseq_df[gpos].iloc[filter_gdf_lines]  # [depth, 11]

            cand_surrounding_read_matrix.append(surrounding_read_df_gpos.values)
            cand_surrounding_baseq_matrix.append(surrounding_baseq_df_gpos.values)
            cand_surrounding_mapq_matrix.append(surrounding_mapq_df_gpos.values)
            surrounding_depth.append(surrounding_read_df_gpos.shape[0])

        max_depth = max(depth)
        max_surrounding_depth = max(surrounding_depth)
        return cand_pos, cand_edge_columns, cand_edge_matrix, cand_pair_columns, cand_pair_route, \
            cand_group_pos, cand_read_matrix, cand_baseq_matrix, cand_mapq_matrix, max_depth, \
            cand_surrounding_read_matrix, cand_surrounding_baseq_matrix, \
            cand_surrounding_mapq_matrix, max_surrounding_depth
    except:
        print("try..except.. break")
        return cand_pos, cand_edge_columns, cand_edge_matrix, cand_pair_columns, cand_pair_route, \
            cand_group_pos, cand_read_matrix, cand_baseq_matrix, cand_mapq_matrix, max_depth, \
            cand_surrounding_read_matrix, cand_surrounding_baseq_matrix, \
            cand_surrounding_mapq_matrix, max_surrounding_depth


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
