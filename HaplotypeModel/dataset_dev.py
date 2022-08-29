from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import torch
import tables
import numpy as np
import pandas as pd

class PileupFeature():
    def __init__(self, table_file, valid_idx = None):
        if valid_idx is not None:
            print(valid_idx.shape)
            self.pileup_sequences = table_file.root.pileup_sequences[valid_idx,:,:]    # [N, depth, 33]
            self.pileup_hap = table_file.root.pileup_hap[valid_idx,:,:]
            self.pileup_baseq = table_file.root.pileup_baseq[valid_idx,:,:]
            self.pileup_mapq = table_file.root.pileup_mapq[valid_idx,:,:]
            self.candidate_positions = table_file.root.candidate_positions[valid_idx,:]
        else:
            self.pileup_sequences = table_file.root.pileup_sequences    # [N, depth, 33]
            self.pileup_hap = table_file.root.pileup_hap
            self.pileup_baseq = table_file.root.pileup_baseq
            self.pileup_mapq = table_file.root.pileup_mapq
            self.candidate_positions = table_file.root.candidate_positions
    def get_all_items(self):
        return self.pileup_sequences, self.pileup_hap, self.pileup_baseq, self.pileup_mapq
    def get_feature_tensor(self):
        sequences = self.pileup_sequences
        baseq = self.pileup_baseq
        mapq = self.pileup_mapq
        hap = self.pileup_hap
        array = np.array([sequences,baseq,mapq,hap]).transpose(1,2,3,0) # [N, depth, 33, 4]
        return array

class HaplotypeFeature():
    def __init__(self, table_file, valid_idx = None):
        if valid_idx is not None:
            self.haplotype_sequences = table_file.root.haplotype_sequences[valid_idx,:,:]
            self.haplotype_hap = table_file.root.haplotype_hap[valid_idx,:,:]
            self.haplotype_baseq = table_file.root.haplotype_baseq[valid_idx,:,:]
            self.haplotype_mapq = table_file.root.haplotype_mapq[valid_idx,:,:]
            self.candidate_positions = table_file.root.candidate_positions[valid_idx,:]
            self.haplotype_positions = table_file.root.haplotype_positions[valid_idx,:]
        else:
            self.haplotype_sequences = table_file.root.haplotype_sequences
            self.haplotype_hap = table_file.root.haplotype_hap
            self.haplotype_baseq = table_file.root.haplotype_baseq
            self.haplotype_mapq = table_file.root.haplotype_mapq
            self.candidate_positions = table_file.root.candidate_positions
            self.haplotype_positions = table_file.root.haplotype_positions
    def get_all_items(self):
        return self.haplotype_sequences, self.haplotype_hap, self.haplotype_baseq, self.haplotype_mapq
    def get_feature_tensor(self):
        sequences = self.haplotype_sequences
        baseq = self.haplotype_baseq
        mapq = self.haplotype_mapq
        hap = self.haplotype_hap
        array = np.array([sequences,baseq,mapq,hap]).transpose(1,2,3,0) # [N, depth, 11, 4]
        return array

class LabelField():
    def __init__(self, table_file):
        ###
        """
        high confident site: cf == 1
        non_variant site: cf == 1 & zy == -1
        variant site: cf == 1 & zy > 0
        """
        
        cf = table_file.root.candidate_labels[:,0]
        gt = table_file.root.candidate_labels[:,1]
        zy = table_file.root.candidate_labels[:,2]
        self.valid_idx = np.where((cf == 1) & (zy >=-1) & (zy < 10))[0]
        self.cf = cf[self.valid_idx]
        self.gt = gt[self.valid_idx]
        self.zy = zy[self.valid_idx]

    def get_refcall_idx(self):
        return np.where((self.cf == 1) & (self.zy == -1) & (self.gt < 10))[0]
    def get_variant_idx(self):
        return np.where((self.cf == 1) & (self.zy > 0) & (self.gt < 10))[0]
        

class TrainingDataset(Dataset):
    def __init__(self, bin_path, pn_value=1.0):
        """
        pn_vlaue:  the number of variant sites (P) deivided by the number of refcall sites (N)
        """
        bin_file = tables.open_file(bin_path, mode='r')
        label_field = LabelField(bin_file)
        ref_calls = label_field.get_refcall_idx()
        variant_calls = label_field.get_variant_idx()
        pileup_feature = PileupFeature(bin_file, label_field.valid_idx)
        haplotype_feature = HaplotypeFeature(bin_file, label_field.valid_idx)
        ### refcall和variant按照pn_value的比例进行混合，pn_value越大，则variant越多
        ### 以refcall为遍历的基础，每次从variant中随机采样pn_value*len(refcall)个variant进行混合
        training_sample_indexes = []
        i = 0
        block_size = 1000
        while i<len(ref_calls):
            if i+1000<len(ref_calls):
                tmp_ref_calls = ref_calls[i:i+block_size]
                tmp_variant_calls = np.random.choice(variant_calls, size = int(block_size*pn_value), replace=True)
                tmp_merge_calls = np.concatenate((tmp_ref_calls, tmp_variant_calls))
                np.random.shuffle(tmp_merge_calls)
                training_sample_indexes = np.concatenate((training_sample_indexes, tmp_merge_calls))
                i += block_size
            else:
                tmp_ref_calls = ref_calls[i:]
                tmp_variant_calls = np.random.choice(variant_calls, size = int(len(tmp_ref_calls)*pn_value), replace=True)
                tmp_merge_calls = np.concatenate((tmp_ref_calls, tmp_variant_calls))
                np.random.shuffle(tmp_merge_calls)
                training_sample_indexes = np.concatenate((training_sample_indexes, tmp_merge_calls))
                i += len(tmp_ref_calls)
        self.training_sample_indexes = training_sample_indexes
        print("INFO: creating pileup feature array")
        self.pileup_feature_array = pileup_feature.get_feature_tensor()
        print("INFO: creating haplotype feature array")
        self.haplotype_feature_array = haplotype_feature.get_feature_tensor()
        self.gt = label_field.gt
        self.zy = label_field.zy
    def __len__(self):
        return len(self.training_sample_indexes)
    def __getitem__(self, idx):
        i = int(self.training_sample_indexes[idx])
        return self.pileup_feature_array[i], self.haplotype_feature_array[i], self.gt[i], self.zy[i]

class TestDataset(Dataset):
    def __init__(self, bin_path):
        """
        pn_vlaue:  the number of variant sites (P) deivided by the number of refcall sites (N)
        """
        bin_file = tables.open_file(bin_path, mode='r')
        pileup_feature = PileupFeature(bin_file)
        haplotype_feature = HaplotypeFeature(bin_file)
        self.pileup_feature_array = pileup_feature.get_feature_tensor()
        self.haplotype_feature_array = haplotype_feature.get_feature_tensor()
    def __len__(self):
        return self.pileup_feature_array.shape[0]
    def __getitem__(self, idx):
        return self.pileup_feature_array[idx], self.haplotype_feature_array[idx]
