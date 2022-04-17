import os
import argparse
import tables
import numpy as np
import pandas as pd
from itertools import product
import concurrent.futures
import math
import sys
from tqdm import tqdm

indexes_with_deletion = list(map(''.join, product(
    ['A', 'C', 'G', 'T', 'D'], ['A', 'C', 'G', 'T', 'D'])))
indexes_without_deletion = list(
    map(''.join, product(['A', 'C', 'G', 'T'], ['A', 'C', 'G', 'T'])))


def find_path(data_frame, node0, node1, col):
    if node0 == node1:
        # 找node0为起点的两条权重最大的边
        idx = list(map(''.join, product([node0], ['A', 'C', 'G', 'T'])))
        edges = data_frame.loc[idx, col].sort_values(ascending=False)[:2]
        edge_idx = edges.index
        edge_0 = edges[0]
        edge_1 = edges[1]
        return edge_idx[0][1], edge_idx[1][1]
    else:
        idx0 = list(map(''.join, product([node0], ['A', 'C', 'G', 'T'])))
        edges = data_frame.loc[idx0, col].sort_values(ascending=False)[:1]
        edge_0 = edges[0]
        edge_idx0 = edges.index[0][1]

        idx1 = list(map(''.join, product([node1], ['A', 'C', 'G', 'T'])))
        edges = data_frame.loc[idx1, col].sort_values(ascending=False)[:1]
        edge_1 = edges[0]
        edge_idx1 = edges.index[0][1]
        return edge_idx0, edge_idx1


def parse_graph(data_frame):
    columns = data_frame.columns
    indexes = data_frame.index
    remove_deletion_indexes = [v for v in indexes if 'D' not in v]
    path = []
    for i in range(len(columns)):
        if i == 0:
            # 找到两条权值最大的边
            edge_column = columns[i]
            edges = data_frame.loc[remove_deletion_indexes,
                                   columns[i]].sort_values(ascending=False)[:2]
            edge_idx = edges.index
            edge_0 = edges[0]
            edge_1 = edges[1]
            if edge_1 == 0:
                # 第二大权值的边为0
                source_node_0 = edge_idx[0][0]
                source_node_1 = edge_idx[0][0]
                out_node_0 = edge_idx[0][1]
                out_node_1 = edge_idx[0][1]
            else:
                ratio = edge_0 / edge_1
                # TODO: 判断两条边的权重相差是否很大，确定edge_1是否是可取的
                source_node_0 = edge_idx[0][0]
                source_node_1 = edge_idx[1][0]
                out_node_0 = edge_idx[0][1]
                out_node_1 = edge_idx[1][1]
            path.append((source_node_0, source_node_1))
            path.append((out_node_0, out_node_1))
        else:
            out_node_0, out_node_1 = find_path(
                data_frame, out_node_0, out_node_1, columns[i])
            path.append((out_node_0, out_node_1))
    return path


def run(edge_matrix, edge_columns, positions, interval):
    homo_pos = []
    ctgname = positions[0].split(':')[0]
    for i in tqdm(interval, ncols=100, desc="%s" % ctgname):
        edge_col = [str(v, encoding='utf-8').split(':')[1]
                    for v in edge_columns[i]]
        df = pd.DataFrame(
            edge_matrix[i], columns=edge_col, index=indexes_with_deletion)
        position = positions[i]
        path = parse_graph(df)  # 正向
        cols_t = df.columns[::-1]
        df2 = df.loc[:,cols_t]
        cols_t = ['-'.join(v.split('-')[::-1]) for v in cols_t] # columns每条边的起点和终点需要交换
        idx_t = [v[::-1] for v in df2.index]  # edge需要反向，例如CA需要变成AC
        df2.columns = cols_t
        df2.index = idx_t
        path2 = parse_graph(df2)    # 反向
        if len(path) == 11 and len(path2) == 11:
            if path[5][0] == path[5][1] or path2[5][0] == path2[5][1]:
                homo_pos.append(position)
        # if path[5][0] == path[5][1]:
        #     homo_pos.append(position)
    return homo_pos


def Run(args):
    total_threads = args.threads
    bin_dir = args.data
    output = args.output
    fout = open(output, 'w')
    data_files = []
    for name in os.listdir(bin_dir):
        data_files.append(bin_dir + os.sep + name)
    for bin_path in data_files:
        print(os.path.basename(bin_path))
        table_file = tables.open_file(bin_path, 'r')
        edge_matrix = np.array(table_file.root.edge_matrix)  # [N,25,10]
        edge_columns = np.array(table_file.root.edge_columns)
        positions = [str(v[0], encoding='utf-8')
                     for v in np.array(table_file.root.position).tolist()]

        step = math.ceil(edge_matrix.shape[0] / total_threads)
        divided_intervals = []
        for dt in range(0, edge_matrix.shape[0], step):
            if dt + step < edge_matrix.shape[0]:
                divided_intervals.append(list(range(dt, dt + step)))
            else:
                divided_intervals.append(list(range(dt, edge_matrix.shape[0])))
        output_homo = []
        with concurrent.futures.ProcessPoolExecutor(max_workers=total_threads) as executor:
            signals = [executor.submit(run, edge_matrix, edge_columns, positions, divided_intervals[thread_id]) for
                       thread_id in range(0, total_threads)]
            for sig in concurrent.futures.as_completed(signals):
                if sig.exception() is None:
                    # get the results
                    homos = sig.result()
                    if len(homos) > 0:
                        output_homo.extend(homos)
                else:
                    sys.stderr.write("ERROR: " + str(sig.exception()) + "\n")
                sig._result = None  # python issue 27144
        for item in output_homo:
            fout.write(item + '\n')
        table_file.close()
    fout.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-data', required=True, help='directory of bin files')
    parser.add_argument('-output', required=True, help='output vcf file')
    parser.add_argument("--threads", '-t', type=int, default=20,
                        help="The number of threads: %(default)d")
    args = parser.parse_args()
    Run(args)


if __name__ == '__main__':
    main()
