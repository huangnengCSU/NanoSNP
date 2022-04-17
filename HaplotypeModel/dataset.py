from numpy.core.fromnumeric import choose
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import torch
from tqdm import tqdm
import h5py
import numpy as np
import pandas as pd
import argparse
import os
import sys
import tables
from get_truth import GT21_LABLES

TABLE_FILTERS = tables.Filters(complib='blosc:lz4hc', complevel=5)


def calculate_percentage(ts):
    # ts: L, N, C
    # return: L, N, 5
    ts_A = np.expand_dims(((ts == 1).sum(2) / ((ts != -2).sum(2) + 1e-9)), 2)
    ts_C = np.expand_dims(((ts == 2).sum(2) / ((ts != -2).sum(2) + 1e-9)), 2)
    ts_G = np.expand_dims(((ts == 3).sum(2) / ((ts != -2).sum(2) + 1e-9)), 2)
    ts_T = np.expand_dims(((ts == 4).sum(2) / ((ts != -2).sum(2) + 1e-9)), 2)
    ts_D = np.expand_dims(((ts == -1).sum(2) / ((ts != -2).sum(2) + 1e-9)), 2)
    return np.concatenate((ts_A, ts_C, ts_G, ts_T, ts_D), axis=2)


def cal_label(v1, v2):
    if v1 == 0 and v2 == 0:
        return 0
    elif (v1 == 0 and v2 == 1) or (v1 == 1 and v2 == 0):
        return 1
    elif (v1 == 0 and v2 == 2) or (v1 == 2 and v2 == 0):
        return 2
    elif (v1 == 0 and v2 == 3) or (v1 == 3 and v2 == 0):
        return 3
    elif v1 == 1 and v2 == 1:
        return 4
    elif (v1 == 1 and v2 == 2) or (v1 == 2 and v2 == 1):
        return 5
    elif (v1 == 1 and v2 == 3) or (v1 == 3 and v2 == 1):
        return 6
    elif v1 == 2 and v2 == 2:
        return 7
    elif (v1 == 2 and v2 == 3) or (v1 == 3 and v2 == 2):
        return 8
    elif v1 == 3 and v2 == 3:
        return 9
    elif v1 == 4 and v2 == 4:
        return 10
    elif (v1 == 0 and v2 == 4) or (v1 == 4 and v2 == 0):
        return 11
    elif (v1 == 1 and v2 == 4) or (v1 == 4 and v2 == 1):
        return 12
    elif (v1 == 2 and v2 == 4) or (v1 == 4 and v2 == 2):
        return 13
    elif (v1 == 3 and v2 == 4) or (v1 == 4 and v2 == 3):
        return 14


class TrainDataset(Dataset):
    def __init__(self, data_dir1, data_dir2, max_depth=20, min_depth=5):
        bin_names1 = os.listdir(data_dir1)
        bin_paths1 = []
        for name in bin_names1:
            bin_paths1.append(data_dir1 + os.sep + name)
        self.bin_paths1 = bin_paths1

        bin_names2 = os.listdir(data_dir2)
        bin_paths2 = []
        for name in bin_names2:
            bin_paths2.append(data_dir2 + os.sep + name)
        self.bin_paths2 = bin_paths2
        self.max_depth = max_depth
        self.min_depth = min_depth

    def __getitem__(self, i):
        tables.set_blosc_max_threads(16)

        table_file1 = tables.open_file(self.bin_paths1[i], 'r')
        # positions1 = np.array(table_file1.root.position).tolist()
        positions1 = [str(v, encoding='utf-8') for v in np.array(table_file1.root.position).squeeze(1).tolist()]
        surrounding_read_matrix1 = np.array(table_file1.root.surrounding_read_matrix)
        surrounding_mask_matrix1 = (surrounding_read_matrix1 != -2).astype(int)  # [N,depth,11]
        surrounding_depth1 = (surrounding_mask_matrix1.sum(2) > 0).sum(1)

        assert len(surrounding_depth1) == len(positions1)

        table_file2 = tables.open_file(self.bin_paths2[i], 'r')
        # positions2 = np.array(table_file2.root.position).tolist()
        positions2 = [str(v, encoding='utf-8') for v in np.array(table_file2.root.position).squeeze(1).tolist()]
        surrounding_read_matrix2 = np.array(table_file2.root.surrounding_read_matrix)
        surrounding_mask_matrix2 = (surrounding_read_matrix2 != -2).astype(int)  # [N,depth,11]
        surrounding_depth2 = (surrounding_mask_matrix2.sum(2) > 0).sum(1)

        idx1, idx2 = [], []
        k, j = 0, 0
        while k < len(positions1) and j < len(positions2):
            try:
                if positions1[k] == positions2[j]:
                    if surrounding_depth1[k] < self.min_depth or surrounding_depth2[j] < self.min_depth:
                        # reads覆盖度过低，则不作为训练数据
                        k += 1
                        j += 1
                        continue
                    idx1.append(k)
                    idx2.append(j)
                    k += 1
                    j += 1
                elif int(positions1[k].split(':')[1]) < int(positions2[j].split(':')[1]):
                    while (int(positions1[k].split(':')[1]) < int(positions2[j].split(':')[1])):
                        k += 1

                elif int(positions1[k].split(':')[1]) > int(positions2[j].split(':')[1]):
                    while (int(positions1[k].split(':')[1]) > int(positions2[j].split(':')[1])):
                        j += 1
            except IndexError:
                break

        if len(idx1) == 0 or len(idx2) == 0:
            return None, None, None, None, None

        edge_matrix1 = np.array(table_file1.root.edge_matrix).transpose((0, 2, 1))[idx1]  # [N,adjacent*2,25]
        # edge_matrix1 = edge_matrix1/(np.expand_dims(edge_matrix1.sum(2) + 1e-6, axis=2))
        pair_route1 = np.array(table_file1.root.pair_route).transpose((0, 2, 1))[idx1]  # [N,adjacent*2,25]
        read_matrix1 = np.array(table_file1.root.read_matrix)[idx1, :self.max_depth, :]  # [N,max_depth,adjacent*2+1]
        base_quality_matrix1 = np.array(table_file1.root.base_quality_matrix)[idx1, :self.max_depth,
                               :]  # [N,max_depth,adjacent*2+1]
        mapping_quality_matrix1 = np.array(table_file1.root.mapping_quality_matrix)[idx1, :self.max_depth,
                                  :]  # [N,max_depth,adjacent*2+1]
        mask_matrix1 = (read_matrix1 != -2).astype(int)  # [N,depth,adjacent*2+1]
        phase_matrix1 = np.ones_like(read_matrix1)  # [N,depth,adjacent*2+1]

        surrounding_read_matrix1 = np.array(table_file1.root.surrounding_read_matrix)[idx1, :self.max_depth, :]
        surrounding_base_quality_matrix1 = np.array(table_file1.root.surrounding_base_quality_matrix)[idx1,
                                           :self.max_depth, :]
        surrounding_mapping_quality_matrix1 = np.array(table_file1.root.surrounding_mapping_quality_matrix)[idx1,
                                              :self.max_depth, :]

        surrounding_mask_matrix1 = (surrounding_read_matrix1 != -2).astype(int)  # [N,depth,11]
        surrounding_phase_matrix1 = np.ones_like(surrounding_read_matrix1)  # [N,depth,11]

        edge_matrix2 = np.array(table_file2.root.edge_matrix).transpose((0, 2, 1))[idx2]  # [N,adjacent*2,25]
        pair_route2 = np.array(table_file2.root.pair_route).transpose((0, 2, 1))[idx2]  # [N,adjacent*2,25]
        read_matrix2 = np.array(table_file2.root.read_matrix)[idx2, :self.max_depth, :]  # [N,max_depth,adjacent*2+1]
        base_quality_matrix2 = np.array(table_file2.root.base_quality_matrix)[idx2, :self.max_depth,
                               :]  # [N,max_depth,adjacent*2+1]
        mapping_quality_matrix2 = np.array(table_file2.root.mapping_quality_matrix)[idx2, :self.max_depth,
                                  :]  # [N,depth,adjacent*2+1]
        mask_matrix2 = (read_matrix2 != -2).astype(int)  # [N,max_depth,adjacent*2+1]
        phase_matrix2 = np.ones_like(read_matrix2) + 1  # [N,max_depth,adjacent*2+1]

        surrounding_read_matrix2 = np.array(table_file2.root.surrounding_read_matrix)[idx2, :self.max_depth, :]
        surrounding_base_quality_matrix2 = np.array(table_file2.root.surrounding_base_quality_matrix)[idx2,
                                           :self.max_depth, :]
        surrounding_mapping_quality_matrix2 = np.array(table_file2.root.surrounding_mapping_quality_matrix)[idx2,
                                              :self.max_depth, :]
        surrounding_mask_matrix2 = (surrounding_read_matrix2 != -2).astype(int)  # [N,depth,11]
        surrounding_phase_matrix2 = np.ones_like(surrounding_read_matrix2) + 1  # [N,depth,11]

        cat_surrounding_read_matrix = np.expand_dims(
            np.concatenate([surrounding_read_matrix1, surrounding_read_matrix2], axis=1), axis=3)
        cat_surrounding_base_quality_matrix = np.expand_dims(
            np.concatenate([surrounding_base_quality_matrix1, surrounding_base_quality_matrix2], axis=1), axis=3)
        cat_surrounding_mapping_quality_matrix = np.expand_dims(
            np.concatenate([surrounding_mapping_quality_matrix1, surrounding_mapping_quality_matrix2], axis=1), axis=3)
        cat_surrounding_mask_matrix = np.expand_dims(
            np.concatenate([surrounding_mask_matrix1, surrounding_mask_matrix2], axis=1), axis=3)
        cat_surrounding_phase_matrix = np.expand_dims(
            np.concatenate([surrounding_phase_matrix1, surrounding_phase_matrix2], axis=1), axis=3)
        g0 = np.concatenate([cat_surrounding_read_matrix, cat_surrounding_base_quality_matrix, \
                             cat_surrounding_mapping_quality_matrix, cat_surrounding_mask_matrix, \
                             cat_surrounding_phase_matrix], axis=3)  # [N,depth,11,5]

        cat_read_matrix = np.expand_dims(np.concatenate([read_matrix1, read_matrix2], axis=1), axis=3)
        cat_base_quality_matrix = np.expand_dims(np.concatenate([base_quality_matrix1, base_quality_matrix2], axis=1),
                                                 axis=3)
        cat_mapping_quality_matrix = np.expand_dims(
            np.concatenate([mapping_quality_matrix1, mapping_quality_matrix2], axis=1), axis=3)
        cat_mask_matrix = np.expand_dims(np.concatenate([mask_matrix1, mask_matrix2], axis=1), axis=3)
        cat_phase_matrix = np.expand_dims(np.concatenate([phase_matrix1, phase_matrix2], axis=1), axis=3)
        g1 = np.concatenate(
            [cat_read_matrix, cat_base_quality_matrix, cat_mapping_quality_matrix, cat_mask_matrix, cat_phase_matrix],
            axis=3)  # [N,depth,adjacent*2+1,5]

        g2 = np.concatenate([np.expand_dims(edge_matrix1, axis=3), np.expand_dims(edge_matrix2, axis=3)], axis=3)
        g3 = np.concatenate([np.expand_dims(pair_route1, axis=3), np.expand_dims(pair_route2, axis=3)], axis=3)

        labels = np.array(table_file1.root.labels)[idx1]  # [N,3]
        high_condfident_label = labels[:, 0]
        gt_label = labels[:, 1]
        zy_label = labels[:, 2]

        # filter with high confident region and shuffle
        variants_idx = np.where((high_condfident_label > 0) & (zy_label >= 0) & (gt_label >= 0) & (gt_label < 10))[0]
        non_variants_idx = np.where((high_condfident_label > 0) & (zy_label == -1))[0]
        if len(variants_idx) > len(non_variants_idx):
            filtered_idx = np.concatenate((variants_idx, non_variants_idx))
        else:
            filtered_idx = np.concatenate(
                (variants_idx, np.random.choice(non_variants_idx, size=len(variants_idx), replace=False)))
        # filtered_idx = np.where((high_condfident_label > 0) & (gt_label >= 0) & (gt_label < 10))[0]
        np.random.shuffle(filtered_idx)
        filtered_idx = filtered_idx.tolist()

        table_file1.close()
        table_file2.close()

        return g0[filtered_idx], g1[filtered_idx], g2[filtered_idx], g3[filtered_idx], gt_label[filtered_idx]

    def __len__(self):
        return len(self.bin_paths1)


###
# 把所有的数据文件在初始化时全部读入内存
###

class TrainDatasetPreLoad(Dataset):
    def __init__(self, data_dir1, data_dir2, pileup_length, haplotype_length, max_depth=20, min_depth=5):
        assert pileup_length % 2 == 1
        assert haplotype_length % 2 == 1
        assert pileup_length == haplotype_length
        bin_names1 = os.listdir(data_dir1)
        bin_paths1 = []
        for name in bin_names1:
            bin_paths1.append(data_dir1 + os.sep + name)
        self.bin_paths1 = bin_paths1

        bin_names2 = os.listdir(data_dir2)
        bin_paths2 = []
        for name in bin_names2:
            bin_paths2.append(data_dir2 + os.sep + name)
        self.bin_paths2 = bin_paths2
        self.max_depth = max_depth
        self.min_depth = min_depth
        bins = []
        assert len(bin_paths1) == len(bin_paths2)
        for i in range(len(bin_paths1)):
            table_file1 = tables.open_file(self.bin_paths1[i], 'r')
            positions1 = [str(v, encoding='utf-8') for v in np.array(table_file1.root.position).squeeze(1).tolist()]
            surrounding_read_matrix1 = np.array(table_file1.root.surrounding_read_matrix)
            surrounding_mask_matrix1 = (surrounding_read_matrix1 != -2).astype(int)  # [N,depth,11]
            surrounding_depth1 = (surrounding_mask_matrix1.sum(2) > 0).sum(1)
            assert len(surrounding_depth1) == len(positions1)

            table_file2 = tables.open_file(self.bin_paths2[i], 'r')
            positions2 = [str(v, encoding='utf-8') for v in np.array(table_file2.root.position).squeeze(1).tolist()]
            surrounding_read_matrix2 = np.array(table_file2.root.surrounding_read_matrix)
            surrounding_mask_matrix2 = (surrounding_read_matrix2 != -2).astype(int)  # [N,depth,11]
            surrounding_depth2 = (surrounding_mask_matrix2.sum(2) > 0).sum(1)

            idx1, idx2 = [], []
            k, j = 0, 0
            while k < len(positions1) and j < len(positions2):
                try:
                    if positions1[k] == positions2[j]:
                        if surrounding_depth1[k] < self.min_depth or surrounding_depth2[j] < self.min_depth:
                            # reads覆盖度过低，则不作为训练数据
                            k += 1
                            j += 1
                            continue
                        idx1.append(k)
                        idx2.append(j)
                        k += 1
                        j += 1
                    elif int(positions1[k].split(':')[1]) < int(positions2[j].split(':')[1]):
                        while (int(positions1[k].split(':')[1]) < int(positions2[j].split(':')[1])):
                            k += 1

                    elif int(positions1[k].split(':')[1]) > int(positions2[j].split(':')[1]):
                        while (int(positions1[k].split(':')[1]) > int(positions2[j].split(':')[1])):
                            j += 1
                except IndexError:
                    break

            if len(idx1) == 0 or len(idx2) == 0:
                bins.append([None, None, None, None, None])

            edge_matrix1 = np.array(table_file1.root.edge_matrix).transpose((0, 2, 1))[idx1]  # [N,adjacent*2,25]
            # edge_matrix1 = edge_matrix1/(np.expand_dims(edge_matrix1.sum(2) + 1e-6, axis=2))
            pair_route1 = np.array(table_file1.root.pair_route).transpose((0, 2, 1))[idx1]  # [N,adjacent*2,25]
            read_matrix1 = np.array(table_file1.root.read_matrix)[idx1, :self.max_depth,
                           :]  # [N,max_depth,adjacent*2+1]
            base_quality_matrix1 = np.array(table_file1.root.base_quality_matrix)[idx1, :self.max_depth,
                                   :]  # [N,max_depth,adjacent*2+1]
            mapping_quality_matrix1 = np.array(table_file1.root.mapping_quality_matrix)[idx1, :self.max_depth,
                                      :]  # [N,max_depth,adjacent*2+1]
            mask_matrix1 = (read_matrix1 != -2).astype(int)  # [N,depth,adjacent*2+1]
            phase_matrix1 = np.ones_like(read_matrix1)  # [N,depth,adjacent*2+1]

            surrounding_read_matrix1 = np.array(table_file1.root.surrounding_read_matrix)[idx1, :self.max_depth, :]
            surrounding_base_quality_matrix1 = np.array(table_file1.root.surrounding_base_quality_matrix)[idx1,
                                               :self.max_depth, :]
            surrounding_mapping_quality_matrix1 = np.array(table_file1.root.surrounding_mapping_quality_matrix)[idx1,
                                                  :self.max_depth, :]

            surrounding_mask_matrix1 = (surrounding_read_matrix1 != -2).astype(int)  # [N,depth,11]
            surrounding_phase_matrix1 = np.ones_like(surrounding_read_matrix1)  # [N,depth,11]

            edge_matrix2 = np.array(table_file2.root.edge_matrix).transpose((0, 2, 1))[idx2]  # [N,adjacent*2,25]
            pair_route2 = np.array(table_file2.root.pair_route).transpose((0, 2, 1))[idx2]  # [N,adjacent*2,25]
            read_matrix2 = np.array(table_file2.root.read_matrix)[idx2, :self.max_depth,
                           :]  # [N,max_depth,adjacent*2+1]
            base_quality_matrix2 = np.array(table_file2.root.base_quality_matrix)[idx2, :self.max_depth,
                                   :]  # [N,max_depth,adjacent*2+1]
            mapping_quality_matrix2 = np.array(table_file2.root.mapping_quality_matrix)[idx2, :self.max_depth,
                                      :]  # [N,depth,adjacent*2+1]
            mask_matrix2 = (read_matrix2 != -2).astype(int)  # [N,max_depth,adjacent*2+1]
            phase_matrix2 = np.ones_like(read_matrix2) + 1  # [N,max_depth,adjacent*2+1]

            surrounding_read_matrix2 = np.array(table_file2.root.surrounding_read_matrix)[idx2, :self.max_depth, :]
            surrounding_base_quality_matrix2 = np.array(table_file2.root.surrounding_base_quality_matrix)[idx2,
                                               :self.max_depth, :]
            surrounding_mapping_quality_matrix2 = np.array(table_file2.root.surrounding_mapping_quality_matrix)[idx2,
                                                  :self.max_depth, :]
            surrounding_mask_matrix2 = (surrounding_read_matrix2 != -2).astype(int)  # [N,depth,11]
            surrounding_phase_matrix2 = np.ones_like(surrounding_read_matrix2) + 1  # [N,depth,11]

            cat_surrounding_read_matrix = np.expand_dims(
                np.concatenate([surrounding_read_matrix1, surrounding_read_matrix2], axis=1), axis=3)
            cat_surrounding_base_quality_matrix = np.expand_dims(
                np.concatenate([surrounding_base_quality_matrix1, surrounding_base_quality_matrix2], axis=1), axis=3)
            cat_surrounding_mapping_quality_matrix = np.expand_dims(
                np.concatenate([surrounding_mapping_quality_matrix1, surrounding_mapping_quality_matrix2], axis=1),
                axis=3)
            cat_surrounding_mask_matrix = np.expand_dims(
                np.concatenate([surrounding_mask_matrix1, surrounding_mask_matrix2], axis=1), axis=3)
            cat_surrounding_phase_matrix = np.expand_dims(
                np.concatenate([surrounding_phase_matrix1, surrounding_phase_matrix2], axis=1), axis=3)
            g0 = np.concatenate([cat_surrounding_read_matrix, cat_surrounding_base_quality_matrix, \
                                 cat_surrounding_mapping_quality_matrix, cat_surrounding_mask_matrix, \
                                 cat_surrounding_phase_matrix], axis=3)  # [N,depth,11,5]

            cat_read_matrix = np.expand_dims(np.concatenate([read_matrix1, read_matrix2], axis=1), axis=3)
            cat_base_quality_matrix = np.expand_dims(
                np.concatenate([base_quality_matrix1, base_quality_matrix2], axis=1),
                axis=3)
            cat_mapping_quality_matrix = np.expand_dims(
                np.concatenate([mapping_quality_matrix1, mapping_quality_matrix2], axis=1), axis=3)
            cat_mask_matrix = np.expand_dims(np.concatenate([mask_matrix1, mask_matrix2], axis=1), axis=3)
            cat_phase_matrix = np.expand_dims(np.concatenate([phase_matrix1, phase_matrix2], axis=1), axis=3)
            g1 = np.concatenate(
                [cat_read_matrix, cat_base_quality_matrix, cat_mapping_quality_matrix, cat_mask_matrix,
                 cat_phase_matrix],
                axis=3)  # [N,depth,adjacent*2+1,5]

            g2 = np.concatenate([np.expand_dims(edge_matrix1, axis=3), np.expand_dims(edge_matrix2, axis=3)], axis=3)
            g3 = np.concatenate([np.expand_dims(pair_route1, axis=3), np.expand_dims(pair_route2, axis=3)], axis=3)

            labels = np.array(table_file1.root.labels)[idx1]  # [N,3]
            high_condfident_label = labels[:, 0]
            gt_label = labels[:, 1]
            zy_label = labels[:, 2]

            # filter with high confident region and shuffle
            variants_idx = np.where((high_condfident_label > 0) & (zy_label >= 0) & (gt_label >= 0) & (gt_label < 10))[
                0]
            non_variants_idx = np.where((high_condfident_label > 0) & (zy_label == -1))[0]
            if len(variants_idx) > len(non_variants_idx):
                filtered_idx = np.concatenate((variants_idx, non_variants_idx))
            else:
                filtered_idx = np.concatenate(
                    (variants_idx, np.random.choice(non_variants_idx, size=len(variants_idx), replace=False)))
            # filtered_idx = np.where((high_condfident_label > 0) & (gt_label >= 0) & (gt_label < 10))[0]
            np.random.shuffle(filtered_idx)
            filtered_idx = filtered_idx.tolist()

            # # filter with high confident region and shuffle
            # filtered_idx = np.where((high_condfident_label > 0) & (gt_label >= 0) & (gt_label < 10))[0]
            # np.random.shuffle(filtered_idx)
            # filtered_idx = filtered_idx.tolist()

            g0 = g0[filtered_idx]
            g1 = g1[filtered_idx]  # [N,40,L,C]
            g2 = g2[filtered_idx]
            g3 = g3[filtered_idx]
            gt_label = gt_label[filtered_idx]

            # 过滤掉标签错误的位点
            g1_base = np.transpose(g1[:, :, :, 0], (2, 0, 1))
            # g1_base = g1[:, :, :, 0].permute(2, 0, 1)  # [5, N, 40]
            g1_tag1_base = g1_base[:, :, :20]
            g1_tag2_base = g1_base[:, :, 20:]

            g1_tag1_base_percentage = calculate_percentage(g1_tag1_base)  # [L,N,5]
            g1_tag2_base_percentage = calculate_percentage(g1_tag2_base)

            # TODO: 可修改haplotype特征的长度
            g1_tag1_max = np.max(g1_tag1_base_percentage[(haplotype_length - 1) // 2], axis=1)  # [N]
            g1_tag1_argmax = np.argmax(g1_tag1_base_percentage[(haplotype_length - 1) // 2], axis=1)  # [A,C,G,T,D]

            g1_tag2_max = np.max(g1_tag2_base_percentage[(haplotype_length - 1) // 2], axis=1)
            g1_tag2_argmax = np.argmax(g1_tag2_base_percentage[(haplotype_length - 1) // 2], axis=1)

            ignore_idx = []
            sel_idx = np.where((g1_tag1_max >= 0.70) & (g1_tag2_max >= 0.70))[0]
            for ti in sel_idx:
                if cal_label(g1_tag1_argmax[ti], g1_tag2_argmax[ti]) != gt_label[ti]:
                    ignore_idx.append(ti)

            mask = np.ones(len(g1_tag1_max), np.bool)
            mask[ignore_idx] = False

            g0 = g0[mask]
            g1 = g1[mask]
            g2 = g2[mask]
            g3 = g3[mask]
            gt_label = gt_label[mask]
            print(self.bin_paths1[i], ':')
            print(len(ignore_idx), len(mask) - len(ignore_idx))

            bins.append([g0, g1, g2, g3, gt_label])
            table_file1.close()
            table_file2.close()

        self.bins = bins

    def __getitem__(self, i):
        g0, g1, g2, g3, gt_label = self.bins[i]
        idx = list(range(len(g1)))
        np.random.shuffle(idx)
        return g0[idx], g1[idx], g2[idx], g3[idx], gt_label[idx]

    def __len__(self):
        return len(self.bin_paths1)


class EvaluateDataset(Dataset):
    def __init__(self, data_dir1, data_dir2, max_depth=20, min_depth=5):
        bin_names1 = os.listdir(data_dir1)
        bin_paths1 = []
        for name in bin_names1:
            bin_paths1.append(data_dir1 + os.sep + name)
        self.bin_paths1 = bin_paths1

        bin_names2 = os.listdir(data_dir2)
        bin_paths2 = []
        for name in bin_names2:
            bin_paths2.append(data_dir2 + os.sep + name)
        self.bin_paths2 = bin_paths2
        self.max_depth = max_depth
        self.min_depth = min_depth

    def __getitem__(self, i):
        tables.set_blosc_max_threads(16)

        table_file1 = tables.open_file(self.bin_paths1[i], 'r')
        # positions1 = np.array(table_file1.root.position).tolist()
        positions1 = [str(v, encoding='utf-8') for v in np.array(table_file1.root.position).squeeze(1).tolist()]
        surrounding_read_matrix1 = np.array(table_file1.root.surrounding_read_matrix)
        surrounding_mask_matrix1 = (surrounding_read_matrix1 != -2).astype(int)  # [N,depth,11]
        surrounding_depth1 = (surrounding_mask_matrix1.sum(2) > 0).sum(1)
        assert len(surrounding_depth1) == len(positions1)

        table_file2 = tables.open_file(self.bin_paths2[i], 'r')
        # positions2 = np.array(table_file2.root.position).tolist()
        positions2 = [str(v, encoding='utf-8') for v in np.array(table_file2.root.position).squeeze(1).tolist()]
        surrounding_read_matrix2 = np.array(table_file2.root.surrounding_read_matrix)
        surrounding_mask_matrix2 = (surrounding_read_matrix2 != -2).astype(int)  # [N,depth,11]
        surrounding_depth2 = (surrounding_mask_matrix2.sum(2) > 0).sum(1)

        idx1, idx2 = [], []
        k, j = 0, 0
        while k < len(positions1) and j < len(positions2):
            try:
                if positions1[k] == positions2[j]:
                    if surrounding_depth1[k] < self.min_depth or surrounding_depth2[j] < self.min_depth:
                        # reads覆盖度过低，则不作为训练数据
                        k += 1
                        j += 1
                        continue
                    idx1.append(k)
                    idx2.append(j)
                    k += 1
                    j += 1
                elif int(positions1[k].split(':')[1]) < int(positions2[j].split(':')[1]):
                    while (int(positions1[k].split(':')[1]) < int(positions2[j].split(':')[1])):
                        k += 1

                elif int(positions1[k].split(':')[1]) > int(positions2[j].split(':')[1]):
                    while (int(positions1[k].split(':')[1]) > int(positions2[j].split(':')[1])):
                        j += 1
            except IndexError:
                break

        if len(idx1) == 0 or len(idx2) == 0:
            return None, None, None, None, None, None

        position = np.array(positions1)[idx1]

        edge_matrix1 = np.array(table_file1.root.edge_matrix).transpose((0, 2, 1))[idx1]  # [N,adjacent*2,25]
        # edge_matrix1 = edge_matrix1/(np.expand_dims(edge_matrix1.sum(2) + 1e-6, axis=2))
        pair_route1 = np.array(table_file1.root.pair_route).transpose((0, 2, 1))[idx1]  # [N,adjacent*2,25]
        read_matrix1 = np.array(table_file1.root.read_matrix)[idx1, :self.max_depth, :]  # [N,depth,adjacent*2+1]
        base_quality_matrix1 = np.array(table_file1.root.base_quality_matrix)[idx1, :self.max_depth,
                               :]  # [N,depth,adjacent*2+1]
        mapping_quality_matrix1 = np.array(table_file1.root.mapping_quality_matrix)[idx1, :self.max_depth,
                                  :]  # [N,depth,adjacent*2+1]
        mask_matrix1 = (read_matrix1 != -2).astype(int)  # [N,depth,adjacent*2+1]
        phase_matrix1 = np.ones_like(read_matrix1)  # [N,depth,adjacent*2+1]

        surrounding_read_matrix1 = np.array(table_file1.root.surrounding_read_matrix)[idx1, :self.max_depth, :]
        surrounding_base_quality_matrix1 = np.array(table_file1.root.surrounding_base_quality_matrix)[idx1,
                                           :self.max_depth, :]
        surrounding_mapping_quality_matrix1 = np.array(table_file1.root.surrounding_mapping_quality_matrix)[idx1,
                                              :self.max_depth, :]

        surrounding_mask_matrix1 = (surrounding_read_matrix1 != -2).astype(int)  # [N,depth,11]
        surrounding_phase_matrix1 = np.ones_like(surrounding_read_matrix1)  # [N,depth,11]

        edge_matrix2 = np.array(table_file2.root.edge_matrix).transpose((0, 2, 1))[idx2]  # [N,adjacent*2,25]
        pair_route2 = np.array(table_file2.root.pair_route).transpose((0, 2, 1))[idx2]  # [N,adjacent*2,25]
        read_matrix2 = np.array(table_file2.root.read_matrix)[idx2, :self.max_depth, :]  # [N,depth,adjacent*2+1]
        base_quality_matrix2 = np.array(table_file2.root.base_quality_matrix)[idx2, :self.max_depth,
                               :]  # [N,depth,adjacent*2+1]
        mapping_quality_matrix2 = np.array(table_file2.root.mapping_quality_matrix)[idx2, :self.max_depth,
                                  :]  # [N,depth,adjacent*2+1]
        mask_matrix2 = (read_matrix2 != -2).astype(int)  # [N,depth,adjacent*2+1]
        phase_matrix2 = np.ones_like(read_matrix2) + 1  # [N,depth,adjacent*2+1]

        surrounding_read_matrix2 = np.array(table_file2.root.surrounding_read_matrix)[idx2, :self.max_depth, :]
        surrounding_base_quality_matrix2 = np.array(table_file2.root.surrounding_base_quality_matrix)[idx2,
                                           :self.max_depth, :]
        surrounding_mapping_quality_matrix2 = np.array(table_file2.root.surrounding_mapping_quality_matrix)[idx2,
                                              :self.max_depth, :]
        surrounding_mask_matrix2 = (surrounding_read_matrix2 != -2).astype(int)  # [N,depth,11]
        surrounding_phase_matrix2 = np.ones_like(surrounding_read_matrix2) + 1  # [N,depth,11]

        cat_surrounding_read_matrix = np.expand_dims(
            np.concatenate([surrounding_read_matrix1, surrounding_read_matrix2], axis=1), axis=3)
        cat_surrounding_base_quality_matrix = np.expand_dims(
            np.concatenate([surrounding_base_quality_matrix1, surrounding_base_quality_matrix2], axis=1), axis=3)
        cat_surrounding_mapping_quality_matrix = np.expand_dims(
            np.concatenate([surrounding_mapping_quality_matrix1, surrounding_mapping_quality_matrix2], axis=1), axis=3)
        cat_surrounding_mask_matrix = np.expand_dims(
            np.concatenate([surrounding_mask_matrix1, surrounding_mask_matrix2], axis=1), axis=3)
        cat_surrounding_phase_matrix = np.expand_dims(
            np.concatenate([surrounding_phase_matrix1, surrounding_phase_matrix2], axis=1), axis=3)
        g0 = np.concatenate([cat_surrounding_read_matrix, cat_surrounding_base_quality_matrix, \
                             cat_surrounding_mapping_quality_matrix, cat_surrounding_mask_matrix, \
                             cat_surrounding_phase_matrix], axis=3)  # [N,depth,11,5]

        cat_read_matrix = np.expand_dims(np.concatenate([read_matrix1, read_matrix2], axis=1), axis=3)
        cat_base_quality_matrix = np.expand_dims(np.concatenate([base_quality_matrix1, base_quality_matrix2], axis=1),
                                                 axis=3)
        cat_mapping_quality_matrix = np.expand_dims(
            np.concatenate([mapping_quality_matrix1, mapping_quality_matrix2], axis=1), axis=3)
        cat_mask_matrix = np.expand_dims(np.concatenate([mask_matrix1, mask_matrix2], axis=1), axis=3)
        cat_phase_matrix = np.expand_dims(np.concatenate([phase_matrix1, phase_matrix2], axis=1), axis=3)
        g1 = np.concatenate(
            [cat_read_matrix, cat_base_quality_matrix, cat_mapping_quality_matrix, cat_mask_matrix, cat_phase_matrix],
            axis=3)  # [N,depth,adjacent*2+1,5]

        g2 = np.concatenate([np.expand_dims(edge_matrix1, axis=3), np.expand_dims(edge_matrix2, axis=3)], axis=3)
        g3 = np.concatenate([np.expand_dims(pair_route1, axis=3), np.expand_dims(pair_route2, axis=3)], axis=3)

        labels = np.array(table_file1.root.labels)[idx1]  # [N,3]
        high_condfident_label = labels[:, 0]
        gt_label = labels[:, 1]
        zy_label = labels[:, 2]

        # filter with high confident region and shuffle
        variants_idx = np.where((high_condfident_label > 0) & (zy_label >= 0) & (gt_label >= 0) & (gt_label < 10))[0]
        non_variants_idx = np.where((high_condfident_label > 0) & (zy_label == -1))[0]
        if len(variants_idx) > len(non_variants_idx):
            filtered_idx = np.concatenate((variants_idx, non_variants_idx))
        else:
            filtered_idx = np.concatenate(
                (variants_idx, np.random.choice(non_variants_idx, size=len(variants_idx), replace=False)))
        # filtered_idx = np.where((high_condfident_label > 0) & (gt_label >= 0) & (gt_label < 10))[0]
        np.random.shuffle(filtered_idx)
        filtered_idx = filtered_idx.tolist()

        # # filter with high confident region and shuffle
        # filtered_idx = np.where((high_condfident_label > 0) & (gt_label >= 0) & (gt_label < 10))[0]
        # np.random.shuffle(filtered_idx)
        # filtered_idx = filtered_idx.tolist()

        table_file1.close()
        table_file2.close()

        return position[filtered_idx].tolist(), g0[filtered_idx], g1[filtered_idx], g2[filtered_idx], g3[filtered_idx], \
               gt_label[filtered_idx]

    def __len__(self):
        return len(self.bin_paths1)


###
# 把所有的数据文件在初始化时全部读入内存
###

class EvaluateDatasetPreLoad(Dataset):
    def __init__(self, data_dir1, data_dir2, pileup_length, haplotype_length, max_depth=20, min_depth=5):
        assert pileup_length % 2 == 1
        assert haplotype_length % 2 == 1
        assert pileup_length == haplotype_length
        bin_names1 = os.listdir(data_dir1)
        bin_paths1 = []
        for name in bin_names1:
            bin_paths1.append(data_dir1 + os.sep + name)
        self.bin_paths1 = bin_paths1

        bin_names2 = os.listdir(data_dir2)
        bin_paths2 = []
        for name in bin_names2:
            bin_paths2.append(data_dir2 + os.sep + name)
        self.bin_paths2 = bin_paths2
        self.max_depth = max_depth
        self.min_depth = min_depth

        bins = []
        assert len(bin_paths1) == len(bin_paths2)
        for i in range(len(bin_paths1)):
            table_file1 = tables.open_file(self.bin_paths1[i], 'r')
            positions1 = [str(v, encoding='utf-8') for v in np.array(table_file1.root.position).squeeze(1).tolist()]
            surrounding_read_matrix1 = np.array(table_file1.root.surrounding_read_matrix)
            surrounding_mask_matrix1 = (surrounding_read_matrix1 != -2).astype(int)  # [N,depth,11]
            surrounding_depth1 = (surrounding_mask_matrix1.sum(2) > 0).sum(1)
            assert len(surrounding_depth1) == len(positions1)

            table_file2 = tables.open_file(self.bin_paths2[i], 'r')
            positions2 = [str(v, encoding='utf-8') for v in np.array(table_file2.root.position).squeeze(1).tolist()]
            surrounding_read_matrix2 = np.array(table_file2.root.surrounding_read_matrix)
            surrounding_mask_matrix2 = (surrounding_read_matrix2 != -2).astype(int)  # [N,depth,11]
            surrounding_depth2 = (surrounding_mask_matrix2.sum(2) > 0).sum(1)

            idx1, idx2 = [], []
            k, j = 0, 0
            while k < len(positions1) and j < len(positions2):
                try:
                    if positions1[k] == positions2[j]:
                        if surrounding_depth1[k] < self.min_depth or surrounding_depth2[j] < self.min_depth:
                            # reads覆盖度过低，则不作为训练数据
                            k += 1
                            j += 1
                            continue
                        idx1.append(k)
                        idx2.append(j)
                        k += 1
                        j += 1
                    elif int(positions1[k].split(':')[1]) < int(positions2[j].split(':')[1]):
                        while (int(positions1[k].split(':')[1]) < int(positions2[j].split(':')[1])):
                            k += 1

                    elif int(positions1[k].split(':')[1]) > int(positions2[j].split(':')[1]):
                        while (int(positions1[k].split(':')[1]) > int(positions2[j].split(':')[1])):
                            j += 1
                except IndexError:
                    break

            if len(idx1) == 0 or len(idx2) == 0:
                bins.append([None, None, None, None, None, None])

            position = np.array(positions1)[idx1]

            edge_matrix1 = np.array(table_file1.root.edge_matrix).transpose((0, 2, 1))[idx1]  # [N,adjacent*2,25]
            # edge_matrix1 = edge_matrix1/(np.expand_dims(edge_matrix1.sum(2) + 1e-6, axis=2))
            pair_route1 = np.array(table_file1.root.pair_route).transpose((0, 2, 1))[idx1]  # [N,adjacent*2,25]
            read_matrix1 = np.array(table_file1.root.read_matrix)[idx1, :self.max_depth,
                           :]  # [N,max_depth,adjacent*2+1]
            base_quality_matrix1 = np.array(table_file1.root.base_quality_matrix)[idx1, :self.max_depth,
                                   :]  # [N,max_depth,adjacent*2+1]
            mapping_quality_matrix1 = np.array(table_file1.root.mapping_quality_matrix)[idx1, :self.max_depth,
                                      :]  # [N,max_depth,adjacent*2+1]
            mask_matrix1 = (read_matrix1 != -2).astype(int)  # [N,depth,adjacent*2+1]
            phase_matrix1 = np.ones_like(read_matrix1)  # [N,depth,adjacent*2+1]

            surrounding_read_matrix1 = np.array(table_file1.root.surrounding_read_matrix)[idx1, :self.max_depth, :]
            surrounding_base_quality_matrix1 = np.array(table_file1.root.surrounding_base_quality_matrix)[idx1,
                                               :self.max_depth, :]
            surrounding_mapping_quality_matrix1 = np.array(table_file1.root.surrounding_mapping_quality_matrix)[idx1,
                                                  :self.max_depth, :]

            surrounding_mask_matrix1 = (surrounding_read_matrix1 != -2).astype(int)  # [N,depth,11]
            surrounding_phase_matrix1 = np.ones_like(surrounding_read_matrix1)  # [N,depth,11]

            edge_matrix2 = np.array(table_file2.root.edge_matrix).transpose((0, 2, 1))[idx2]  # [N,adjacent*2,25]
            pair_route2 = np.array(table_file2.root.pair_route).transpose((0, 2, 1))[idx2]  # [N,adjacent*2,25]
            read_matrix2 = np.array(table_file2.root.read_matrix)[idx2, :self.max_depth,
                           :]  # [N,max_depth,adjacent*2+1]
            base_quality_matrix2 = np.array(table_file2.root.base_quality_matrix)[idx2, :self.max_depth,
                                   :]  # [N,max_depth,adjacent*2+1]
            mapping_quality_matrix2 = np.array(table_file2.root.mapping_quality_matrix)[idx2, :self.max_depth,
                                      :]  # [N,depth,adjacent*2+1]
            mask_matrix2 = (read_matrix2 != -2).astype(int)  # [N,max_depth,adjacent*2+1]
            phase_matrix2 = np.ones_like(read_matrix2) + 1  # [N,max_depth,adjacent*2+1]

            surrounding_read_matrix2 = np.array(table_file2.root.surrounding_read_matrix)[idx2, :self.max_depth, :]
            surrounding_base_quality_matrix2 = np.array(table_file2.root.surrounding_base_quality_matrix)[idx2,
                                               :self.max_depth, :]
            surrounding_mapping_quality_matrix2 = np.array(table_file2.root.surrounding_mapping_quality_matrix)[idx2,
                                                  :self.max_depth, :]
            surrounding_mask_matrix2 = (surrounding_read_matrix2 != -2).astype(int)  # [N,depth,11]
            surrounding_phase_matrix2 = np.ones_like(surrounding_read_matrix2) + 1  # [N,depth,11]

            cat_surrounding_read_matrix = np.expand_dims(
                np.concatenate([surrounding_read_matrix1, surrounding_read_matrix2], axis=1), axis=3)
            cat_surrounding_base_quality_matrix = np.expand_dims(
                np.concatenate([surrounding_base_quality_matrix1, surrounding_base_quality_matrix2], axis=1), axis=3)
            cat_surrounding_mapping_quality_matrix = np.expand_dims(
                np.concatenate([surrounding_mapping_quality_matrix1, surrounding_mapping_quality_matrix2], axis=1),
                axis=3)
            cat_surrounding_mask_matrix = np.expand_dims(
                np.concatenate([surrounding_mask_matrix1, surrounding_mask_matrix2], axis=1), axis=3)
            cat_surrounding_phase_matrix = np.expand_dims(
                np.concatenate([surrounding_phase_matrix1, surrounding_phase_matrix2], axis=1), axis=3)
            g0 = np.concatenate([cat_surrounding_read_matrix, cat_surrounding_base_quality_matrix, \
                                 cat_surrounding_mapping_quality_matrix, cat_surrounding_mask_matrix, \
                                 cat_surrounding_phase_matrix], axis=3)  # [N,depth,11,5]

            cat_read_matrix = np.expand_dims(np.concatenate([read_matrix1, read_matrix2], axis=1), axis=3)
            cat_base_quality_matrix = np.expand_dims(
                np.concatenate([base_quality_matrix1, base_quality_matrix2], axis=1),
                axis=3)
            cat_mapping_quality_matrix = np.expand_dims(
                np.concatenate([mapping_quality_matrix1, mapping_quality_matrix2], axis=1), axis=3)
            cat_mask_matrix = np.expand_dims(np.concatenate([mask_matrix1, mask_matrix2], axis=1), axis=3)
            cat_phase_matrix = np.expand_dims(np.concatenate([phase_matrix1, phase_matrix2], axis=1), axis=3)
            g1 = np.concatenate(
                [cat_read_matrix, cat_base_quality_matrix, cat_mapping_quality_matrix, cat_mask_matrix,
                 cat_phase_matrix],
                axis=3)  # [N,depth,adjacent*2+1,5]

            g2 = np.concatenate([np.expand_dims(edge_matrix1, axis=3), np.expand_dims(edge_matrix2, axis=3)], axis=3)
            g3 = np.concatenate([np.expand_dims(pair_route1, axis=3), np.expand_dims(pair_route2, axis=3)], axis=3)

            labels = np.array(table_file1.root.labels)[idx1]  # [N,3]
            high_condfident_label = labels[:, 0]
            gt_label = labels[:, 1]
            zy_label = labels[:, 2]

            # filter with high confident region and shuffle
            variants_idx = np.where((high_condfident_label > 0) & (zy_label >= 0) & (gt_label >= 0) & (gt_label < 10))[
                0]
            non_variants_idx = np.where((high_condfident_label > 0) & (zy_label == -1))[0]
            if len(variants_idx) > len(non_variants_idx):
                filtered_idx = np.concatenate((variants_idx, non_variants_idx))
            else:
                filtered_idx = np.concatenate(
                    (variants_idx, np.random.choice(non_variants_idx, size=len(variants_idx), replace=False)))
            # filtered_idx = np.where((high_condfident_label > 0) & (gt_label >= 0) & (gt_label < 10))[0]
            np.random.shuffle(filtered_idx)
            filtered_idx = filtered_idx.tolist()

            # # filter with high confident region and shuffle
            # filtered_idx = np.where((high_condfident_label > 0) & (gt_label >= 0) & (gt_label < 10))[0]
            # np.random.shuffle(filtered_idx)
            # filtered_idx = filtered_idx.tolist()

            position = position[filtered_idx]
            g0 = g0[filtered_idx]
            g1 = g1[filtered_idx]  # [N,40,L,C]
            g2 = g2[filtered_idx]
            g3 = g3[filtered_idx]
            gt_label = gt_label[filtered_idx]

            # 过滤掉标签错误的位点
            g1_base = np.transpose(g1[:, :, :, 0], (2, 0, 1))
            # g1_base = g1[:, :, :, 0].permute(2, 0, 1)  # [5, N, 40]
            g1_tag1_base = g1_base[:, :, :20]
            g1_tag2_base = g1_base[:, :, 20:]

            g1_tag1_base_percentage = calculate_percentage(g1_tag1_base)  # [L,N,5]
            g1_tag2_base_percentage = calculate_percentage(g1_tag2_base)

            #TODO： 可修改haplotype特征的长度
            g1_tag1_max = np.max(g1_tag1_base_percentage[(haplotype_length - 1) // 2], axis=1)  # [N]
            g1_tag1_argmax = np.argmax(g1_tag1_base_percentage[(haplotype_length - 1) // 2], axis=1)  # [A,C,G,T,D]

            g1_tag2_max = np.max(g1_tag2_base_percentage[(haplotype_length - 1) // 2], axis=1)
            g1_tag2_argmax = np.argmax(g1_tag2_base_percentage[(haplotype_length - 1) // 2], axis=1)

            ignore_idx = []
            sel_idx = np.where((g1_tag1_max >= 0.70) & (g1_tag2_max >= 0.70))[0]
            for ti in sel_idx:
                if cal_label(g1_tag1_argmax[ti], g1_tag2_argmax[ti]) != gt_label[ti]:
                    ignore_idx.append(ti)

            mask = np.ones(len(g1_tag1_max), np.bool)
            mask[ignore_idx] = False

            position = position[mask]
            g0 = g0[mask]
            g1 = g1[mask]
            g2 = g2[mask]
            g3 = g3[mask]
            gt_label = gt_label[mask]
            print(self.bin_paths1[i], ':')
            print(len(ignore_idx), len(mask) - len(ignore_idx))

            bins.append([position, g0, g1, g2, g3, gt_label])

            table_file1.close()
            table_file2.close()

        self.bins = bins

    def __getitem__(self, i):
        position, g0, g1, g2, g3, gt_label = self.bins[i]
        idx = list(range(len(g1)))
        np.random.shuffle(idx)
        return position[idx].tolist(), g0[idx], g1[idx], g2[idx], g3[idx], gt_label[idx]

    def __len__(self):
        return len(self.bin_paths1)


class PredictDataset(Dataset):
    def __init__(self, data_dir1, data_dir2, max_depth=20, min_depth=5):
        bin_names1 = os.listdir(data_dir1)
        bin_paths1 = []
        for name in bin_names1:
            bin_paths1.append(data_dir1 + os.sep + name)
        self.bin_paths1 = bin_paths1

        bin_names2 = os.listdir(data_dir2)
        bin_paths2 = []
        for name in bin_names2:
            bin_paths2.append(data_dir2 + os.sep + name)
        self.bin_paths2 = bin_paths2
        self.max_depth = max_depth
        self.min_depth = min_depth

    def __getitem__(self, i):
        tables.set_blosc_max_threads(16)

        table_file1 = tables.open_file(self.bin_paths1[i], 'r')
        # positions1 = np.array(table_file1.root.position).tolist()
        positions1 = [str(v, encoding='utf-8') for v in np.array(table_file1.root.position).squeeze(1).tolist()]
        surrounding_read_matrix1 = np.array(table_file1.root.surrounding_read_matrix)
        surrounding_mask_matrix1 = (surrounding_read_matrix1 != -2).astype(int)  # [N,depth,11]
        surrounding_depth1 = (surrounding_mask_matrix1.sum(2) > 0).sum(1)
        assert len(surrounding_depth1) == len(positions1)

        table_file2 = tables.open_file(self.bin_paths2[i], 'r')
        # positions2 = np.array(table_file2.root.position).tolist()
        positions2 = [str(v, encoding='utf-8') for v in np.array(table_file2.root.position).squeeze(1).tolist()]
        surrounding_read_matrix2 = np.array(table_file2.root.surrounding_read_matrix)
        surrounding_mask_matrix2 = (surrounding_read_matrix2 != -2).astype(int)  # [N,depth,11]
        surrounding_depth2 = (surrounding_mask_matrix2.sum(2) > 0).sum(1)

        idx1, idx2 = [], []
        k, j = 0, 0
        while k < len(positions1) and j < len(positions2):
            try:
                if positions1[k] == positions2[j]:
                    if surrounding_depth1[k] < self.min_depth or surrounding_depth2[j] < self.min_depth:
                        # reads覆盖度过低，则不作为训练数据
                        k += 1
                        j += 1
                        continue
                    idx1.append(k)
                    idx2.append(j)
                    k += 1
                    j += 1
                elif int(positions1[k].split(':')[1]) < int(positions2[j].split(':')[1]):
                    while (int(positions1[k].split(':')[1]) < int(positions2[j].split(':')[1])):
                        k += 1

                elif int(positions1[k].split(':')[1]) > int(positions2[j].split(':')[1]):
                    while (int(positions1[k].split(':')[1]) > int(positions2[j].split(':')[1])):
                        j += 1
            except IndexError:
                break

        if len(idx1) == 0 or len(idx2) == 0:
            return None, None, None
        position = [str(v, encoding='utf-8') for v in np.array(table_file1.root.position).squeeze(1)[idx1].tolist()]

        edge_matrix1 = np.array(table_file1.root.edge_matrix).transpose((0, 2, 1))[idx1]  # [N,adjacent*2,25]
        # edge_matrix1 = edge_matrix1/(np.expand_dims(edge_matrix1.sum(2) + 1e-6, axis=2))
        pair_route1 = np.array(table_file1.root.pair_route).transpose((0, 2, 1))[idx1]  # [N,adjacent*2,25]
        read_matrix1 = np.array(table_file1.root.read_matrix)[idx1, :self.max_depth, :]  # [N,depth,adjacent*2+1]
        base_quality_matrix1 = np.array(table_file1.root.base_quality_matrix)[idx1, :self.max_depth,
                               :]  # [N,depth,adjacent*2+1]
        mapping_quality_matrix1 = np.array(table_file1.root.mapping_quality_matrix)[idx1, :self.max_depth,
                                  :]  # [N,depth,adjacent*2+1]
        mask_matrix1 = (read_matrix1 != -2).astype(int)  # [N,depth,adjacent*2+1]
        phase_matrix1 = np.ones_like(read_matrix1)  # [N,depth,adjacent*2+1]

        surrounding_read_matrix1 = np.array(table_file1.root.surrounding_read_matrix)[idx1, :self.max_depth, :]
        surrounding_base_quality_matrix1 = np.array(table_file1.root.surrounding_base_quality_matrix)[idx1,
                                           :self.max_depth, :]
        surrounding_mapping_quality_matrix1 = np.array(table_file1.root.surrounding_mapping_quality_matrix)[idx1,
                                              :self.max_depth, :]

        surrounding_mask_matrix1 = (surrounding_read_matrix1 != -2).astype(int)  # [N,depth,11]
        surrounding_phase_matrix1 = np.ones_like(surrounding_read_matrix1)  # [N,depth,11]

        edge_matrix2 = np.array(table_file2.root.edge_matrix).transpose((0, 2, 1))[idx2]  # [N,adjacent*2,25]
        pair_route2 = np.array(table_file2.root.pair_route).transpose((0, 2, 1))[idx2]  # [N,adjacent*2,25]
        read_matrix2 = np.array(table_file2.root.read_matrix)[idx2, :self.max_depth, :]  # [N,depth,adjacent*2+1]
        base_quality_matrix2 = np.array(table_file2.root.base_quality_matrix)[idx2, :self.max_depth,
                               :]  # [N,depth,adjacent*2+1]
        mapping_quality_matrix2 = np.array(table_file2.root.mapping_quality_matrix)[idx2, :self.max_depth,
                                  :]  # [N,depth,adjacent*2+1]
        mask_matrix2 = (read_matrix2 != -2).astype(int)  # [N,depth,adjacent*2+1]
        phase_matrix2 = np.ones_like(read_matrix2) + 1  # [N,depth,adjacent*2+1]

        surrounding_read_matrix2 = np.array(table_file2.root.surrounding_read_matrix)[idx2, :self.max_depth, :]
        surrounding_base_quality_matrix2 = np.array(table_file2.root.surrounding_base_quality_matrix)[idx2,
                                           :self.max_depth, :]
        surrounding_mapping_quality_matrix2 = np.array(table_file2.root.surrounding_mapping_quality_matrix)[idx2,
                                              :self.max_depth, :]
        surrounding_mask_matrix2 = (surrounding_read_matrix2 != -2).astype(int)  # [N,depth,11]
        surrounding_phase_matrix2 = np.ones_like(surrounding_read_matrix2) + 1  # [N,depth,11]

        cat_surrounding_read_matrix = np.expand_dims(
            np.concatenate([surrounding_read_matrix1, surrounding_read_matrix2], axis=1), axis=3)
        cat_surrounding_base_quality_matrix = np.expand_dims(
            np.concatenate([surrounding_base_quality_matrix1, surrounding_base_quality_matrix2], axis=1), axis=3)
        cat_surrounding_mapping_quality_matrix = np.expand_dims(
            np.concatenate([surrounding_mapping_quality_matrix1, surrounding_mapping_quality_matrix2], axis=1), axis=3)
        cat_surrounding_mask_matrix = np.expand_dims(
            np.concatenate([surrounding_mask_matrix1, surrounding_mask_matrix2], axis=1), axis=3)
        cat_surrounding_phase_matrix = np.expand_dims(
            np.concatenate([surrounding_phase_matrix1, surrounding_phase_matrix2], axis=1), axis=3)
        g0 = np.concatenate([cat_surrounding_read_matrix, cat_surrounding_base_quality_matrix, \
                             cat_surrounding_mapping_quality_matrix, cat_surrounding_mask_matrix, \
                             cat_surrounding_phase_matrix], axis=3)  # [N,depth,11,5]

        cat_read_matrix = np.expand_dims(np.concatenate([read_matrix1, read_matrix2], axis=1), axis=3)
        cat_base_quality_matrix = np.expand_dims(np.concatenate([base_quality_matrix1, base_quality_matrix2], axis=1),
                                                 axis=3)
        cat_mapping_quality_matrix = np.expand_dims(
            np.concatenate([mapping_quality_matrix1, mapping_quality_matrix2], axis=1), axis=3)
        cat_mask_matrix = np.expand_dims(np.concatenate([mask_matrix1, mask_matrix2], axis=1), axis=3)
        cat_phase_matrix = np.expand_dims(np.concatenate([phase_matrix1, phase_matrix2], axis=1), axis=3)
        g1 = np.concatenate(
            [cat_read_matrix, cat_base_quality_matrix, cat_mapping_quality_matrix, cat_mask_matrix, cat_phase_matrix],
            axis=3)  # [N,depth,adjacent*2+1,5]

        g2 = np.concatenate([np.expand_dims(edge_matrix1, axis=3), np.expand_dims(edge_matrix2, axis=3)], axis=3)
        g3 = np.concatenate([np.expand_dims(pair_route1, axis=3), np.expand_dims(pair_route2, axis=3)], axis=3)

        table_file1.close()
        table_file2.close()

        return position, g0, g1, g2, g3

    def __len__(self):
        return len(self.bin_paths1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-tag_bin1', required=True, help='directory of bin files')
    parser.add_argument('-tag_bin2', required=True, help='directory of bin files')
    opt = parser.parse_args()

    train_dataset = TrainDataset(opt.tag_bin1, opt.tag_bin2)
    train_data = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=False, num_workers=2)
    gt_cnt = {}
    for step, (g0, g1, g2, g3, gt) in enumerate(train_data):
        for i in range(21):
            if g1.shape[1] == 0:
                continue
            gt = gt.squeeze(0)
            cnt = (gt == i).int().sum().item()
            gt_cnt[GT21_LABLES[i]] = gt_cnt.get(GT21_LABLES[i], 0) + cnt
    print(gt_cnt)

    # if g1.shape[1] == 0:
    #     continue
    # print(g1.shape, g2.shape, g3.shape, gt.shape)
