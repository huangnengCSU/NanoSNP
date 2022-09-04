import sys
import os
import shutil
import argparse
import yaml
import time
import torch
import torch.nn as nn
import numpy as np
import torch.utils.data
import math
from math import log, e
from model_dev import LSTMNetwork
from dataset_dev import TestDataset
from utils import AttrDict
from options import gt_decoded_labels, zy_decoded_labels, indel1_decoded_labels, indel2_decoded_labels, base_idx
from tqdm import tqdm
from get_truth import load_reference_file


def calculate_score(probability):
    p = probability
    tmp = max((-10 * log(e, 10)) * log(((1.0 - p) + 1e-300) / (p + 1e-300)) + 10, 0)
    return float(round(tmp, 2))


def predict(model, test_data, reference_path, batch_size, pileup_length, haplotype_length, output_file, device):
    references = load_reference_file(reference_path)
    fwriter = open(output_file, 'w')
    model.eval()
    for bin_file in os.listdir(test_data):
        predict_dataset = TestDataset(bin_path = test_data+'/'+bin_file, references=references, pileup_length=pileup_length, haplotype_length=haplotype_length)
        predict_loader = torch.utils.data.DataLoader(predict_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
        for position, pileup_feat, haplotype_feat in predict_loader:
            x_pileup = pileup_feat.type(torch.FloatTensor).to(device)
            x_haplotype = haplotype_feat.type(torch.FloatTensor).to(device)
            if x_pileup.shape[0] == 0 or x_pileup.shape[1] == 0 or x_pileup.ndim != 3:
                continue
            gt_output_,_ = model.predict(x_pileup, x_haplotype)
            gt_output_ = gt_output_.cpu().detach().numpy()
            gt_prob = np.max(gt_output_, axis=1)  # [N]
            gt_output = np.argmax(gt_output_, axis=1)  # [N]
            for j in range(len(gt_output)):
                qual = calculate_score(gt_prob[j])
                gtv = gt_decoded_labels[gt_output[j]]
                ctg, pos = position[j].split(':')
                fwriter.write(ctg + '\t' + pos + '\t' + gtv + '\t' + str(qual) + '\n')
    fwriter.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-config', type=str, required=True, help='path to config file')
    parser.add_argument('-model_path', required=True, help='path to trained model')
    parser.add_argument('-bin_paths', required=True, help='directory of bin files')
    parser.add_argument('-reference_path', required=True, help='path to reference file')
    # parser.add_argument('-contig', required=True, help='contig name of the input bin files')
    parser.add_argument('-output', required=True, help='output vcf file')
    parser.add_argument('-batch_size', type=int, default=1000, help='batch size')
    parser.add_argument('--no_cuda', action="store_true", help='If running on cpu device, set the argument.')
    opt = parser.parse_args()
    device = torch.device('cuda' if not opt.no_cuda else 'cpu')

    configfile = open(opt.config)
    config = AttrDict(yaml.load(configfile, Loader=yaml.FullLoader))
    pred_model = LSTMNetwork(config).to(device)
    pred_model.load_state_dict(torch.load(opt.model_path))
    predict(pred_model, opt.bin_paths, opt.reference_path, opt.batch_size, config.model.pileup_length, config.model.haplotype_length, opt.output, device)


if __name__ == '__main__':
    main()
