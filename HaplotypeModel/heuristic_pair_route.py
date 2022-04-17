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

def run(pair_route, pair_columns, positions, interval):
    homo_pos = []
    ctgname = positions[0].split(':')[0]
    for i in tqdm(interval, ncols=100, desc="%s" % ctgname):
        # edge_col = [str(v, encoding='utf-8').split(':')[1]
        #             for v in edge_columns[i]]
        # df = pd.DataFrame(
        #     edge_matrix[i], columns=edge_col, index=indexes_with_deletion)
        position = positions[i]
        p = pair_route[i]
        pcol = [str(v,encoding='utf-8').split(':')[1] for v in pair_columns[i]]
        homo_cnt = 0
        hete_cnt = 0
        p = pd.DataFrame(p,index=indexes_with_deletion,columns=pcol)
        p = p.loc[indexes_without_deletion]
        A_freq = p.loc[['AA', 'AC', 'AG', 'AT']].values
        C_freq = p.loc[['CA', 'CC', 'CG', 'CT']].values
        G_freq = p.loc[['GA', 'GC', 'GG', 'GT']].values
        T_freq = p.loc[['TA', 'TC', 'TG', 'TT']].values

        A_weights = A_freq.max(0).reshape((1,-1))
        A_idx = np.array(['AA', 'AC', 'AG', 'AT'])[A_freq.argmax(0)].reshape((1,-1))
        C_weights = C_freq.max(0).reshape((1,-1))
        C_idx = np.array(['CA', 'CC', 'CG', 'CT'])[C_freq.argmax(0)].reshape((1,-1))
        G_weights = G_freq.max(0).reshape((1,-1))
        G_idx = np.array(['GA', 'GC', 'GG', 'GT'])[G_freq.argmax(0)].reshape((1,-1))
        T_weights = T_freq.max(0).reshape((1,-1))
        T_idx = np.array(['TA', 'TC', 'TG', 'TT'])[T_freq.argmax(0)].reshape((1,-1))
        weights = np.concatenate([A_weights,C_weights,G_weights,T_weights],axis=0)

        idxes = np.concatenate([A_idx,C_idx,G_idx,T_idx],axis=0)

        sortidx = np.argsort(-weights,axis=0)

        for j in range(sortidx.shape[1]):
            allele1 = idxes[sortidx[0][j],j][1]
            allele2 = idxes[sortidx[1][j],j][1]
            # print(allele1,allele2)
            if allele1==allele2:
                homo_cnt+=1
            else:
                hete_cnt+=1
        if homo_cnt >= hete_cnt:
            homo_pos.append(position)
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
        pair_route = np.array(table_file.root.pair_route)  # [N,25,10]
        pair_columns = np.array(table_file.root.pair_columns)
        positions = [str(v[0], encoding='utf-8')
                     for v in np.array(table_file.root.position).tolist()]

        step = math.ceil(pair_route.shape[0] / total_threads)
        divided_intervals = []
        for dt in range(0, pair_route.shape[0], step):
            if dt + step < pair_route.shape[0]:
                divided_intervals.append(list(range(dt, dt + step)))
            else:
                divided_intervals.append(list(range(dt, pair_route.shape[0])))
        output_homo = []
        with concurrent.futures.ProcessPoolExecutor(max_workers=total_threads) as executor:
            signals = [executor.submit(run, pair_route, pair_columns, positions, divided_intervals[thread_id]) for
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
