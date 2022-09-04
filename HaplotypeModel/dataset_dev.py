from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import torch
import tables
import numpy as np
import pandas as pd
import time

BASE2INT = {'A':1, 'C':2, 'G':3, 'T':4, 'N':0}

def get_base_freq(sequence):
    A_cnt = np.sum(sequence==1,axis=0)
    C_cnt = np.sum(sequence==2,axis=0)
    G_cnt = np.sum(sequence==3,axis=0)
    T_cnt = np.sum(sequence==4,axis=0)
    D_cnt = np.sum(sequence==-1,axis=0)
    total_cnt = A_cnt + C_cnt+G_cnt+T_cnt+D_cnt+1e-6
    A_freq = A_cnt/total_cnt
    C_freq = C_cnt/total_cnt
    G_freq = G_cnt/total_cnt
    T_freq = T_cnt/total_cnt
    D_freq = D_cnt/total_cnt
    return A_freq, C_freq, G_freq, T_freq, D_freq, A_cnt, C_cnt, G_cnt, T_cnt, D_cnt

def get_base_quality(sequence,baseq):
    A_bq = np.sum(baseq*(sequence==1),axis=0)
    C_bq = np.sum(baseq*(sequence==2),axis=0)
    G_bq = np.sum(baseq*(sequence==3),axis=0)
    T_bq = np.sum(baseq*(sequence==4),axis=0)
    A_bq_mean = np.mean(baseq*(sequence==1),axis=0)
    C_bq_mean = np.mean(baseq*(sequence==1),axis=0)
    G_bq_mean = np.mean(baseq*(sequence==1),axis=0)
    T_bq_mean = np.mean(baseq*(sequence==1),axis=0)
    return A_bq, C_bq, G_bq, T_bq, A_bq_mean, C_bq_mean, G_bq_mean, T_bq_mean

def get_mapping_quality(sequence,mapq):
    A_mq = np.sum(mapq*(sequence==1),axis=0)
    C_mq = np.sum(mapq*(sequence==2),axis=0)
    G_mq = np.sum(mapq*(sequence==3),axis=0)
    T_mq = np.sum(mapq*(sequence==4),axis=0)
    A_mq_mean = np.mean(mapq*(sequence==1),axis=0)
    C_mq_mean = np.mean(mapq*(sequence==1),axis=0)
    G_mq_mean = np.mean(mapq*(sequence==1),axis=0)
    T_mq_mean = np.mean(mapq*(sequence==1),axis=0)
    return A_mq, C_mq, G_mq, T_mq, A_mq_mean, C_mq_mean, G_mq_mean, T_mq_mean

def get_seq_baseq_mapq_feat(sequence, baseq, mapq):
    A_freq, C_freq, G_freq, T_freq, D_freq, A_cnt, C_cnt, G_cnt, T_cnt, D_cnt  = get_base_freq(sequence)
    A_bq, C_bq, G_bq, T_bq, A_bq_mean, C_bq_mean, G_bq_mean, T_bq_mean = get_base_quality(sequence,baseq)
    A_mq, C_mq, G_mq, T_mq, A_mq_mean, C_mq_mean, G_mq_mean, T_mq_mean = get_mapping_quality(sequence,mapq)
    return np.array([A_freq, C_freq, G_freq, T_freq, D_freq, A_cnt, C_cnt, G_cnt, T_cnt, D_cnt, A_bq, C_bq, G_bq, T_bq, A_bq_mean, C_bq_mean, G_bq_mean, T_bq_mean, A_mq, C_mq, G_mq, T_mq, A_mq_mean, C_mq_mean, G_mq_mean, T_mq_mean])



def get_frequency_feature(sequence, baseq, mapq, hap):
    integrated_feats = get_seq_baseq_mapq_feat(sequence, baseq, mapq)
    maternal_index = np.any(hap==1,axis=1)
    paternal_index = np.any(hap==2,axis=1)
    unphased_index = np.any(hap==3,axis=1)
    maternal_depth = len(np.where(maternal_index)[0])
    paternal_depth = len(np.where(paternal_index)[0])
    unphased_depth = len(np.where(unphased_index)[0])
    if maternal_depth > 0:
        maternal_sequence = sequence[maternal_index]
        maternal_baseq = baseq[maternal_index]
        maternal_mapq = mapq[maternal_index]
        maternal_feats = get_seq_baseq_mapq_feat(maternal_sequence, maternal_baseq, maternal_mapq)
    else:
        maternal_feats = np.zeros(integrated_feats.shape)
    
    if paternal_depth > 0:
        paternal_sequence = sequence[paternal_index]
        paternal_baseq = baseq[paternal_index]
        paternal_mapq = mapq[paternal_index]
        paternal_feats = get_seq_baseq_mapq_feat(paternal_sequence, paternal_baseq, paternal_mapq)
    else:
        paternal_feats = np.zeros(integrated_feats.shape)
    
    if unphased_depth > 0:
        unphased_sequence = sequence[unphased_index]
        unphased_baseq = baseq[unphased_index]
        unphased_mapq = mapq[unphased_index]
        unphased_feats = get_seq_baseq_mapq_feat(unphased_sequence, unphased_baseq, unphased_mapq)
    else:
        unphased_feats = np.zeros(integrated_feats.shape)
    overall_feats = np.concatenate((integrated_feats, maternal_feats, paternal_feats, unphased_feats), axis=0)
    return overall_feats


    

class PileupFeature():
    def __init__(self, table_file, references, pileup_length, valid_idx = None):
        if valid_idx is not None:
            self.pileup_sequences = np.array(table_file.root.pileup_sequences)[valid_idx]    # [N, depth, 33]
            self.pileup_hap = np.array(table_file.root.pileup_hap)[valid_idx]
            self.pileup_baseq = np.array(table_file.root.pileup_baseq)[valid_idx]
            self.pileup_mapq = np.array(table_file.root.pileup_mapq)[valid_idx]
            self.candidate_positions = np.array(table_file.root.candidate_positions)[valid_idx]
        else:
            self.pileup_sequences = np.array(table_file.root.pileup_sequences)    # [N, depth, 33]
            self.pileup_hap = np.array(table_file.root.pileup_hap)
            self.pileup_baseq = np.array(table_file.root.pileup_baseq)
            self.pileup_mapq = np.array(table_file.root.pileup_mapq)
            self.candidate_positions = np.array(table_file.root.candidate_positions)
        ## get reference sequence
        candidate_reference_sequences = []
        for i in range(len(self.candidate_positions)):
            ref_seq = []
            ctg,pos = str(self.candidate_positions[i][0],encoding='utf-8').split(":")
            pos = int(pos)
            for j in range(pos-pileup_length//2, pos+pileup_length//2+1):
                ref_pos = j-1   # ctg pos is 1-based, ref_pos is 0-based
                try:
                    ref_base = references[ctg][ref_pos]
                    ref_seq.append(BASE2INT[ref_base])
                except:
                    ref_seq.append(0)   # 0 is for N
            candidate_reference_sequences.append(ref_seq)
        self.candidate_reference_sequences = np.array(candidate_reference_sequences)

    def get_all_items(self):
        return self.pileup_sequences, self.pileup_hap, self.pileup_baseq, self.pileup_mapq, self.candidate_reference_sequences
    def get_feature_tensor(self):
        sequences = self.pileup_sequences
        baseq = self.pileup_baseq
        mapq = self.pileup_mapq
        hap = self.pileup_hap
        array = np.array([sequences,baseq,mapq,hap]).transpose(1,2,3,0) # [N, depth, 33, 4]
        return array

class HaplotypeFeature():
    def __init__(self, table_file, references, haplotype_length, valid_idx = None):
        if valid_idx is not None:
            self.haplotype_sequences = np.array(table_file.root.haplotype_sequences)[valid_idx]
            self.haplotype_hap = np.array(table_file.root.haplotype_hap)[valid_idx]
            self.haplotype_baseq = np.array(table_file.root.haplotype_baseq)[valid_idx]
            self.haplotype_mapq = np.array(table_file.root.haplotype_mapq)[valid_idx]
            self.candidate_positions = np.array(table_file.root.candidate_positions)[valid_idx]
            self.haplotype_positions = np.array(table_file.root.haplotype_positions)[valid_idx]
        else:
            self.haplotype_sequences = np.array(table_file.root.haplotype_sequences)
            self.haplotype_hap = np.array(table_file.root.haplotype_hap)
            self.haplotype_baseq = np.array(table_file.root.haplotype_baseq)
            self.haplotype_mapq = np.array(table_file.root.haplotype_mapq)
            self.candidate_positions = np.array(table_file.root.candidate_positions)
            self.haplotype_positions = np.array(table_file.root.haplotype_positions)
        ## get reference sequence
        candidate_reference_sequences = []
        for i in range(len(self.haplotype_positions)):
            ref_seq = []
            for j in range(len(self.haplotype_positions[i])):
                ctg,pos = str(self.haplotype_positions[i][j],encoding='utf-8').split(":")
                pos = int(pos)
                ref_pos = pos-1
                try:
                    ref_base = references[ctg][ref_pos]
                    ref_seq.append(BASE2INT[ref_base])
                except:
                    ref_seq.append(0)   # 0 is for N
            candidate_reference_sequences.append(ref_seq)
        self.candidate_reference_sequences = np.array(candidate_reference_sequences)
        
    def get_all_items(self):
        return self.haplotype_sequences, self.haplotype_hap, self.haplotype_baseq, self.haplotype_mapq, self.candidate_reference_sequences
    def get_feature_tensor(self):
        sequences = self.haplotype_sequences
        baseq = self.haplotype_baseq
        mapq = self.haplotype_mapq
        hap = self.haplotype_hap
        array = np.array([sequences,baseq,mapq,hap]).transpose(1,2,3,0) # [N, depth, 11, 4]
        return array

class LabelField():
    def __init__(self, table_file):
        ### high confident site: cf == 1  non_variant site: cf == 1 & zy == -1   variant site: cf == 1 & zy > 0
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
    def __init__(self, bin_path, references, pileup_length=33, haplotype_length=11, pn_value=1.0):
        ### pn_vlaue:  the number of variant sites (P) deivided by the number of refcall sites (N)
        bin_file = tables.open_file(bin_path, mode='r')
        label_field = LabelField(bin_file)
        ref_calls = label_field.get_refcall_idx()
        variant_calls = label_field.get_variant_idx()
        pileup_feature = PileupFeature(bin_file, references, pileup_length, label_field.valid_idx)
        haplotype_feature = HaplotypeFeature(bin_file, references, haplotype_length, label_field.valid_idx)
        
        ### refcall和variant按照pn_value的比例进行混合，pn_value越大，则variant越多
        ### 以refcall为遍历的基础，每次从variant中随机采样pn_value*len(refcall)个variant进行混合
        training_sample_indexes = []
        i = 0
        block_size = 1000
        """
        由于refcall比variant多10倍，chr1为例，refcall=223923，variant=16463
        将variant上采样，拷贝后放入训练集
        测试效果：上采样效果更好，下采样效果不好
        """
        while i<len(ref_calls):
            if i+1000<len(ref_calls):
                tmp_ref_calls = ref_calls[i:i+block_size]
                try:
                    tmp_variant_calls = np.random.choice(variant_calls, size = int(block_size*pn_value), replace=True)
                    tmp_merge_calls = np.concatenate((tmp_ref_calls, tmp_variant_calls))
                except:
                    tmp_merge_calls = tmp_ref_calls
                np.random.shuffle(tmp_merge_calls)
                training_sample_indexes = np.concatenate((training_sample_indexes, tmp_merge_calls))
                i += block_size
            else:
                tmp_ref_calls = ref_calls[i:]
                try:
                    tmp_variant_calls = np.random.choice(variant_calls, size = int(len(tmp_ref_calls)*pn_value), replace=True)
                    tmp_merge_calls = np.concatenate((tmp_ref_calls, tmp_variant_calls))
                except:
                    tmp_merge_calls = tmp_ref_calls
                np.random.shuffle(tmp_merge_calls)
                training_sample_indexes = np.concatenate((training_sample_indexes, tmp_merge_calls))
                i += len(tmp_ref_calls)
        """
        由于refcall比variant多10倍，chr1为例，refcall=223923，variant=16463
        将refcall下采样后放入训练集
        测试效果：上采样效果更好，下采样效果不好
        """
        # while i<len(variant_calls):
        #     if i+1000<len(variant_calls):
        #         tmp_variant_calls = variant_calls[i:i+block_size]
        #         if len(ref_calls)>len(variant_calls)*pn_value:
        #             tmp_ref_calls = np.random.choice(ref_calls, size = int(block_size*pn_value), replace=False)
        #         else:
        #             tmp_ref_calls = np.random.choice(ref_calls, size = int(block_size*pn_value), replace=True)
        #         tmp_merge_calls = np.concatenate((tmp_ref_calls, tmp_variant_calls))
        #         np.random.shuffle(tmp_merge_calls)
        #         training_sample_indexes = np.concatenate((training_sample_indexes, tmp_merge_calls))
        #         i += block_size
        #     else:
        #         tmp_variant_calls = variant_calls[i:]
        #         if len(ref_calls)>len(variant_calls)*pn_value:
        #             tmp_ref_calls = np.random.choice(ref_calls, size = int(len(tmp_variant_calls)*pn_value), replace=False)
        #         else:
        #             tmp_ref_calls = np.random.choice(ref_calls, size = int(len(tmp_variant_calls)*pn_value), replace=True)
        #         tmp_merge_calls = np.concatenate((tmp_ref_calls, tmp_variant_calls))
        #         np.random.shuffle(tmp_merge_calls)
        #         training_sample_indexes = np.concatenate((training_sample_indexes, tmp_merge_calls))
        #         i += len(tmp_variant_calls)
        self.training_sample_indexes = training_sample_indexes
        print("INFO: creating pileup feature array")
        self.pileup_sequence, self.pileup_hap, self.pileup_baseq, self.pileup_mapq, self.pileup_reference_sequence = pileup_feature.get_all_items()
        # self.pileup_feature_array = pileup_feature.get_feature_tensor()
        print("INFO: creating haplotype feature array")
        # self.haplotype_feature_array = haplotype_feature.get_feature_tensor()
        self.haplotype_sequence, self.haplotype_hap, self.haplotype_baseq, self.haplotype_mapq, self.haplotype_reference_sequence = haplotype_feature.get_all_items()
        self.gt = label_field.gt
        self.zy = label_field.zy
        bin_file.close()
    def __len__(self):
        return len(self.training_sample_indexes)
    def __getitem__(self, idx):
        i = int(self.training_sample_indexes[idx])
        # return self.pileup_feature_array[i], self.haplotype_feature_array[i], self.gt[i], self.zy[i]
        # pileup_feature_array = np.array([self.pileup_sequence[i], self.pileup_baseq[i], self.pileup_mapq[i], self.pileup_hap[i]])
        # haplotype_feature_array = np.array([self.haplotype_sequence[i], self.haplotype_baseq[i], self.haplotype_mapq[i], self.haplotype_hap[i]])
        pileup_feature_array = get_frequency_feature(self.pileup_sequence[i], self.pileup_baseq[i], self.pileup_mapq[i], self.pileup_hap[i])    # [104, pileup_length]
        pileup_ref_seq = self.pileup_reference_sequence[i].reshape((1,-1))
        pileup_feature_array = np.concatenate((pileup_feature_array, pileup_ref_seq),axis=0)    #   [105, pileup_length]

        haplotype_feature_array = get_frequency_feature(self.haplotype_sequence[i], self.haplotype_baseq[i], self.haplotype_mapq[i], self.haplotype_hap[i]) # [104, haplotype_length]
        haplotype_ref_seq = self.haplotype_reference_sequence[i].reshape((1,-1))
        haplotype_feature_array = np.concatenate((haplotype_feature_array, haplotype_ref_seq),axis=0)   #   [105, haplotype_length]
        gt = self.gt[i]
        zy = self.zy[i] if self.zy[i]>= 0 else 0
        return pileup_feature_array, haplotype_feature_array, gt, zy

class EvaluateDataset(Dataset):
    def __init__(self, bin_path, references, pileup_length=33, haplotype_length=11):
        bin_file = tables.open_file(bin_path, mode='r')
        label_field = LabelField(bin_file)
        ref_calls = label_field.get_refcall_idx()
        variant_calls = label_field.get_variant_idx()
        pileup_feature = PileupFeature(bin_file, references, pileup_length, label_field.valid_idx)
        haplotype_feature = HaplotypeFeature(bin_file, references, haplotype_length, label_field.valid_idx)
        training_sample_indexes = np.concatenate((ref_calls, variant_calls),axis=0)
        np.random.shuffle(training_sample_indexes)
        self.training_sample_indexes = training_sample_indexes
        print("INFO: creating pileup feature array")
        self.pileup_sequence, self.pileup_hap, self.pileup_baseq, self.pileup_mapq, self.pileup_reference_sequence = pileup_feature.get_all_items()
        # self.pileup_feature_array = pileup_feature.get_feature_tensor()
        print("INFO: creating haplotype feature array")
        # self.haplotype_feature_array = haplotype_feature.get_feature_tensor()
        self.haplotype_sequence, self.haplotype_hap, self.haplotype_baseq, self.haplotype_mapq, self.haplotype_reference_sequence = haplotype_feature.get_all_items()
        self.gt = label_field.gt
        self.zy = label_field.zy
        bin_file.close()
    def __len__(self):
        return len(self.training_sample_indexes)
    def __getitem__(self, idx):
        i = int(self.training_sample_indexes[idx])
        # return self.pileup_feature_array[i], self.haplotype_feature_array[i], self.gt[i], self.zy[i]
        # pileup_feature_array = np.array([self.pileup_sequence[i], self.pileup_baseq[i], self.pileup_mapq[i], self.pileup_hap[i]])
        # haplotype_feature_array = np.array([self.haplotype_sequence[i], self.haplotype_baseq[i], self.haplotype_mapq[i], self.haplotype_hap[i]])
        pileup_feature_array = get_frequency_feature(self.pileup_sequence[i], self.pileup_baseq[i], self.pileup_mapq[i], self.pileup_hap[i])    # [104, pileup_length]
        pileup_ref_seq = self.pileup_reference_sequence[i].reshape((1,-1))
        pileup_feature_array = np.concatenate((pileup_feature_array, pileup_ref_seq),axis=0)    #   [105, pileup_length]

        haplotype_feature_array = get_frequency_feature(self.haplotype_sequence[i], self.haplotype_baseq[i], self.haplotype_mapq[i], self.haplotype_hap[i]) # [104, haplotype_length]
        haplotype_ref_seq = self.haplotype_reference_sequence[i].reshape((1,-1))
        haplotype_feature_array = np.concatenate((haplotype_feature_array, haplotype_ref_seq),axis=0)   #   [105, haplotype_length]
        gt = self.gt[i]
        zy = self.zy[i] if self.zy[i]>= 0 else 0
        return pileup_feature_array, haplotype_feature_array, gt, zy

class TestDataset(Dataset):
    def __init__(self, bin_path, references, pileup_length=33, haplotype_length=11):
        bin_file = tables.open_file(bin_path, mode='r')
        pileup_feature = PileupFeature(bin_file, references, pileup_length)
        haplotype_feature = HaplotypeFeature(bin_file, references, haplotype_length)
        self.pileup_sequence, self.pileup_hap, self.pileup_baseq, self.pileup_mapq, self.pileup_reference_sequence = pileup_feature.get_all_items()
        self.haplotype_sequence, self.haplotype_hap, self.haplotype_baseq, self.haplotype_mapq, self.haplotype_reference_sequence = haplotype_feature.get_all_items()
        positions = np.array(haplotype_feature.candidate_positions)
        positions = [str(v, encoding='utf-8') for v in positions.squeeze(1).tolist()]
        self.positions = positions
        # self.pileup_feature_array = pileup_feature.get_feature_tensor()
        # self.haplotype_feature_array = haplotype_feature.get_feature_tensor()
    def __len__(self):
        return self.pileup_sequence.shape[0]
    def __getitem__(self, idx):
        i = idx
        # pileup_feature_array = np.array([self.pileup_sequence[i], self.pileup_baseq[i], self.pileup_mapq[i], self.pileup_hap[i]])
        # haplotype_feature_array = np.array([self.haplotype_sequence[i], self.haplotype_baseq[i], self.haplotype_mapq[i], self.haplotype_hap[i]])
        pos = self.positions[i]
        pileup_feature_array = get_frequency_feature(self.pileup_sequence[i], self.pileup_baseq[i], self.pileup_mapq[i], self.pileup_hap[i])    # [104, pileup_length]
        pileup_ref_seq = self.pileup_reference_sequence[i].reshape((1,-1))
        pileup_feature_array = np.concatenate((pileup_feature_array, pileup_ref_seq),axis=0)    #   [105, pileup_length]

        haplotype_feature_array = get_frequency_feature(self.haplotype_sequence[i], self.haplotype_baseq[i], self.haplotype_mapq[i], self.haplotype_hap[i]) # [104, haplotype_length]
        haplotype_ref_seq = self.haplotype_reference_sequence[i].reshape((1,-1))
        haplotype_feature_array = np.concatenate((haplotype_feature_array, haplotype_ref_seq),axis=0)   #   [105, haplotype_length]
        return pos, pileup_feature_array, haplotype_feature_array