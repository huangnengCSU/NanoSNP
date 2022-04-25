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
from model import CatModel
from dataset import PredictDataset
from utils import AttrDict
from options import gt_decoded_labels, zy_decoded_labels, indel1_decoded_labels, indel2_decoded_labels, base_idx
from tqdm import tqdm


def calculate_score(probability):
    p = probability
    tmp = max((-10 * log(e, 10)) * log(((1.0 - p) + 1e-300) / (p + 1e-300)) + 10, 0)
    return float(round(tmp, 2))


def predict(model, test_data, batch_size, output_file, device):
    fwriter = open(output_file, 'w')
    model.eval()
    for positions, g0, g1, g2, g3 in tqdm(test_data, ncols=100):
        if g1.shape[1] == 0:
            continue
        # feature_tensor = feature_tensor.squeeze(0).type(torch.FloatTensor)
        g0 = g0.squeeze(0).type(torch.FloatTensor)
        g1 = g1.squeeze(0).type(torch.FloatTensor)
        g2 = g2.squeeze(0).type(torch.FloatTensor)
        g3 = g3.squeeze(0).type(torch.FloatTensor)

        batches = math.ceil(g1.shape[0] / batch_size)
        for i in range(batches):
            if i != batches - 1:
                sub_positions = positions[i * batch_size:(i + 1) * batch_size]
                sub_g0 = g0[i * batch_size:(i + 1) * batch_size].to(device)
                sub_g1 = g1[i * batch_size:(i + 1) * batch_size].to(device)
                sub_g2 = g2[i * batch_size:(i + 1) * batch_size].to(device)
                sub_g3 = g3[i * batch_size:(i + 1) * batch_size].to(device)
            else:
                sub_positions = positions[i * batch_size:]
                sub_g0 = g0[i * batch_size:].to(device)
                sub_g1 = g1[i * batch_size:].to(device)
                sub_g2 = g2[i * batch_size:].to(device)
                sub_g3 = g3[i * batch_size:].to(device)

            gt_output_ = model.predict(sub_g0, sub_g1, sub_g2, sub_g3)  # Nxnclass
            gt_output_ = gt_output_.cpu().detach().numpy()
            gt_prob = np.max(gt_output_, axis=1)  # [N]
            gt_output = np.argmax(gt_output_, axis=1)  # [N]

            for j in range(len(gt_output)):
                qual = calculate_score(gt_prob[j])
                gtv = gt_decoded_labels[gt_output[j]]
                ctg, pos = sub_positions[j][0].split(':')
                fwriter.write(ctg + '\t' + pos + '\t' + gtv + '\t' + str(qual) + '\n')
    fwriter.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-config', type=str, required=True, help='path to config file')
    parser.add_argument('-model_path', required=True, help='path to trained model')
    parser.add_argument('-data_tag1', required=True, help='directory of bin files')
    parser.add_argument('-data_tag2', required=True, help='directory of bin files')
    # parser.add_argument('-contig', required=True, help='contig name of the input bin files')
    parser.add_argument('-output', required=True, help='output vcf file')
    parser.add_argument('-batch_size', type=int, default=1000, help='batch size')
    parser.add_argument('--no_cuda', action="store_true", help='If running on cpu device, set the argument.')
    opt = parser.parse_args()
    device = torch.device('cuda' if not opt.no_cuda else 'cpu')

    configfile = open(opt.config)
    config = AttrDict(yaml.load(configfile, Loader=yaml.FullLoader))
    pred_model = CatModel(nc0=5, nc1=5, nc2=2, nclass=config.model.gt_num_class, nh=256,
                          use_g0=config.model.use_g0,
                          use_g1=config.model.use_g1,
                          use_g2=config.model.use_g2,
                          use_g3=config.model.use_g3).to(device)
    pred_model.load_state_dict(torch.load(opt.model_path))

    test_dataset = PredictDataset(data_dir1=opt.data_tag1, data_dir2=opt.data_tag2,min_depth=2)
    test_data = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)
    predict(pred_model, test_data, opt.batch_size, opt.output, device)


if __name__ == '__main__':
    main()
