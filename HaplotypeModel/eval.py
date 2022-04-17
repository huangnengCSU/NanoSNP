import sys
import os
import shutil
import argparse
import yaml
import time
import torch
import torch.nn as nn
import torch.utils.data
import math
import numpy as np
from model import CatModel
from dataset import EvaluateDatasetPreLoad
from optim import Optimizer, build_optimizer
from utils import AttrDict, init_logger, count_parameters, save_model
from tensorboardX import SummaryWriter
import torchnet.meter as meter
from datetime import datetime
from options import gt_decoded_labels, zy_decoded_labels, indel1_decoded_labels, indel2_decoded_labels
from math import log, e


def calculate_score(probability):
    p = probability
    tmp = max((-10 * log(e, 10)) * log(((1.0 - p) + 1e-300) / (p + 1e-300)) + 10, 0)
    return float(round(tmp, 2))


def eval(model, test_data, batch_size, output_file, device):
    fwriter = open(output_file, 'w')
    fwriter.write(
        "# Contig" + '\t' + "Pos" + '\t' + "Truth" + '\t' + "Pred" + '\t' + "Qual" + '\t' + "True/False" + '\n')
    model.eval()
    total_acc = 0
    total_cnt = 0
    for step, (positions, g0, g1, g2, g3, gt_label) in enumerate(test_data):
        if g1.shape[1] == 0:
            continue
        g0 = g0.squeeze(0).type(torch.FloatTensor)
        g1 = g1.squeeze(0).type(torch.FloatTensor)
        g2 = g2.squeeze(0).type(torch.FloatTensor)
        g3 = g3.squeeze(0).type(torch.FloatTensor)
        gt_label = gt_label.squeeze(0).type(torch.LongTensor)  # [batch,]

        batches = math.ceil(g1.shape[0] / batch_size)
        for i in range(batches):
            if i != batches - 1:
                sub_positions = positions[i * batch_size:(i + 1) * batch_size]
                sub_g0 = g0[i * batch_size:(i + 1) * batch_size].to(device)
                sub_g1 = g1[i * batch_size:(i + 1) * batch_size].to(device)
                sub_g2 = g2[i * batch_size:(i + 1) * batch_size].to(device)
                sub_g3 = g3[i * batch_size:(i + 1) * batch_size].to(device)
                sub_gt_label = gt_label[i * batch_size:(i + 1) * batch_size].to(device)
            else:
                sub_positions = positions[i * batch_size:]
                sub_g0 = g0[i * batch_size:].to(device)
                sub_g1 = g1[i * batch_size:].to(device)
                sub_g2 = g2[i * batch_size:].to(device)
                sub_g3 = g3[i * batch_size:].to(device)
                sub_gt_label = gt_label[i * batch_size:].to(device)

            loss, gt_out = model(sub_g0, sub_g1, sub_g2, sub_g3, sub_gt_label)

            gt_logits = torch.softmax(gt_out, 1).detach().cpu().numpy()
            gt_prob = np.max(gt_logits, axis=1)  # [N]
            gt_output = np.argmax(gt_logits, axis=1)
            # print('pred:',gt_output[:50].tolist())
            # print('labl:',sub_gt_label.detach().cpu().numpy()[:50].tolist())
            total_acc += sum((gt_output == sub_gt_label.detach().cpu().numpy()).astype(int)) / len(gt_output)
            total_cnt += 1

            sub_gt_label = sub_gt_label.detach().cpu().numpy()
            for j in range(len(gt_output)):
                qual = calculate_score(gt_prob[j])
                gtv = gt_decoded_labels[gt_output[j]]
                lbl = gt_decoded_labels[sub_gt_label[j]]
                ctg, pos = sub_positions[j][0].split(':')
                if gtv == lbl:
                    fwriter.write(ctg + '\t' + pos + '\t' + lbl + '\t' + gtv + '\t' + str(qual) + '\t' + "-" + '\n')
                else:
                    fwriter.write(ctg + '\t' + pos + '\t' + lbl + '\t' + gtv + '\t' + str(qual) + '\t' + "False" + '\n')
    print("Accuracy:", total_acc / total_cnt)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-config', type=str, default='config/pore-c.yaml', help='path to config file')
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
    pred_model = CatModel(nc0=5, nc1=5, nc2=2, nclass=10, nh=256,
                          use_g0=config.model.use_g0,
                          use_g1=config.model.use_g1,
                          use_g2=config.model.use_g2,
                          use_g3=config.model.use_g3).to(device)
    pred_model.load_state_dict(torch.load(opt.model_path))

    test_dataset = EvaluateDatasetPreLoad(data_dir1=opt.data_tag1, data_dir2=opt.data_tag2)
    test_data = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4)
    eval(pred_model, test_data, opt.batch_size, opt.output, device)


if __name__ == '__main__':
    main()
