from torch.utils.data import Dataset, DataLoader
import numpy as np
import argparse
import os
import tables
from options import gt_decoded_labels, zy_decoded_labels

TABLE_FILTERS = tables.Filters(complib='blosc:lz4hc', complevel=5)
shuffle_bin_size = 50000

no_flanking_bases = 16
no_of_positions = 2 * no_flanking_bases + 1
channel = ('A', 'C', 'G', 'T', 'I', 'I1', 'D', 'D1', '*',
           'a', 'c', 'g', 't', 'i', 'i1', 'd', 'd1', '#')
channel_size = len(channel)
ont_input_shape = input_shape = [no_of_positions, channel_size]
label_shape = [21, 3, no_of_positions, no_of_positions]
label_size = sum(label_shape)


def calculate_percentage(ts):
    # ts: L, N, C
    # return: L, N, 5
    ts_A = np.expand_dims(((ts == 1).sum(2) / ((ts != -2).sum(2) + 1e-9)), 2)
    ts_C = np.expand_dims(((ts == 2).sum(2) / ((ts != -2).sum(2) + 1e-9)), 2)
    ts_G = np.expand_dims(((ts == 3).sum(2) / ((ts != -2).sum(2) + 1e-9)), 2)
    ts_T = np.expand_dims(((ts == 4).sum(2) / ((ts != -2).sum(2) + 1e-9)), 2)
    ts_D = np.expand_dims(((ts == -1).sum(2) / ((ts != -2).sum(2) + 1e-9)), 2)
    return np.concatenate((ts_A, ts_C, ts_G, ts_T, ts_D), axis=2)


def balance_dataset(gt_label, zy_label):
    """
    position_matrix.shape   (495703, 33, 18)
    gt_label.shape  (495703,)
    zy_label.shape  (495703,)
    """
    num_gts = len(gt_decoded_labels)
    num_zys = len(zy_decoded_labels)
    indexes_for_gt_zy = {}
    num_of_max_categories = 0
    for i in range(num_gts):
        for j in range(num_zys):
            findex = np.where((gt_label == i) & (zy_label == j))[0]
            if len(findex) >= num_of_max_categories:
                num_of_max_categories = len(findex)
            indexes_for_gt_zy[(i, j)] = findex

    # 每个类上采样
    non_zero_categories = 0
    for k in indexes_for_gt_zy.keys():
        size_of_categories = len(indexes_for_gt_zy[k])
        if size_of_categories > 0 and size_of_categories < num_of_max_categories:
            sample_num = num_of_max_categories - size_of_categories
            sampling_index = np.random.choice(indexes_for_gt_zy[k], size=sample_num, replace=True)
            indexes_for_gt_zy[k] = indexes_for_gt_zy[k].tolist() + sampling_index.tolist()
            non_zero_categories += 1

    total_indexes = []
    for k in indexes_for_gt_zy.keys():
        total_indexes.extend(indexes_for_gt_zy[k])

    # 所有类的总和随机下采样
    np.random.shuffle(total_indexes)
    final_index = np.random.choice(total_indexes, size=int(len(total_indexes) / non_zero_categories))
    return final_index


def filter_sample(position_matrix, gt_label, zy_label):
    pass


class TrainDataset(Dataset):
    def __init__(self, datapath, use_balance=False, for_evaluate=False):
        # tables.set_blosc_max_threads(16)
        table_file = tables.open_file(datapath, 'r')
        position_matrix = np.array(table_file.root.position_matrix)  # [N,33,18]
        label = table_file.root.label  # [N,90]
        gt_label = np.array(label[:, :21].argmax(1))  # [N]
        zy_label = np.array(label[:, 21:24].argmax(1))  # [N]
        indel1_label = np.array(label[:, 24:57].argmax(1))  # [N]
        indel2_label = np.array(label[:, 57:].argmax(1))  # [N]
        table_file.close()
        if use_balance:
            train_idx = balance_dataset(gt_label=gt_label, zy_label=zy_label)
            self.position_matrix = position_matrix[train_idx]
            self.gt_label = gt_label[train_idx]
            self.zy_label = zy_label[train_idx]
            self.indel1_label = indel1_label[train_idx]
            self.indel2_label = indel2_label[train_idx]
        else:
            self.position_matrix = position_matrix
            self.gt_label = gt_label
            self.zy_label = zy_label
            self.indel1_label = indel1_label
            self.indel2_label = indel2_label
        
        if for_evaluate:
            variant_idx = np.where(self.zy_label>0)[0]  # zy_decoded_labels = ['0/0', '1/1', '0/1']
            self.position_matrix = self.position_matrix[variant_idx]
            self.gt_label = self.gt_label[variant_idx]
            self.zy_label = self.zy_label[variant_idx]
            self.indel1_label = self.indel1_label[variant_idx]
            self.indel2_label = self.indel2_label[variant_idx]

    def __getitem__(self, i):
        position_matrix = self.position_matrix[i]
        gt_label = self.gt_label[i]
        zy_label = self.zy_label[i]
        indel1_label = self.indel1_label[i]
        indel2_label = self.indel2_label[i]
        return position_matrix, gt_label, zy_label, indel1_label, indel2_label

    def __len__(self):
        return len(self.gt_label)


class PredictDataset(Dataset):
    def __init__(self, datapath):
        # tables.set_blosc_max_threads(16)
        table_file = tables.open_file(datapath, 'r')
        position_matrix = np.array(table_file.root.position_matrix)  # [N,33,18]
        position = table_file.root.position
        positions = []
        reference_bases = []
        contig_names = []
        for item in position:
            ctg_name, pos, seq = str(item[0], encoding="utf-8").strip().split(':')
            contig_names.append(ctg_name)
            pos = int(pos)
            positions.append(pos)
            reference_bases.append(ord(seq[16]))  # ASCII
        positions = np.array(positions)
        reference_bases = np.array(reference_bases)
        table_file.close()
        self.contig_names = contig_names
        self.position_matrix = position_matrix
        self.positions = positions
        self.reference_bases = reference_bases

    def __getitem__(self, i):
        contig_names = self.contig_names[i]
        positions = self.positions[i]
        reference_bases = self.reference_bases[i]
        position_matrix = self.position_matrix[i]
        return contig_names, positions, reference_bases, position_matrix

    def __len__(self):
        return len(self.position_matrix)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-data', required=True, help='directory of bin files')
    opt = parser.parse_args()

    '''
    filepaths = [opt.data + '/' + file for file in os.listdir(opt.data)]
    for file in filepaths:
        print(file)
        dataset = TrainDataset(datapath=file)
        dl = DataLoader(dataset, batch_size=2000, shuffle=True)
        for batch in dl:
            position_matrix, gt_label, zy_label, indel1_label, indel2_label = batch
            print(gt_label.shape)
    '''

    filepaths = [opt.data + '/' + file for file in os.listdir(opt.data)]
    for file in filepaths:
        print(file)
        dataset = PredictDataset(datapath=file)
        dl = DataLoader(dataset, batch_size=2000, shuffle=False)
        for batch in dl:
            contig_names, positions, reference_bases, position_matrix = batch
            print(len(positions))
