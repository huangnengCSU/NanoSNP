
import os
import argparse
import yaml
import torch
import torch.nn as nn
import numpy as np
import torch.utils.data
from model_dev import LSTMNetwork
from dataset_dev import EvaluateDataset
from utils import AttrDict
import torchnet.meter as meter
from get_truth import load_reference_file


def eval(config, model, validating_data, references, batch_size):
    model.eval()
    total_loss = 0
    total_images = 0
    gt_confusion_matrix = meter.ConfusionMeter(config.model.gt_num_class)
    batch_steps = len(validating_data)
    total_acc = 0
    total_cnt = 0
    for bin_file in os.listdir(validating_data):
        validate_dataset = EvaluateDataset(bin_path = validating_data+'/'+bin_file, references=references, pileup_length=config.model.pileup_length, haplotype_length=config.model.haplotype_length)
        validate_loader = torch.utils.data.DataLoader(validate_dataset, batch_size=batch_size, shuffle=False, num_workers=config.training.num_gpu * 3)
        for step, (pileup_feat, haplotype_feat, gt, zy) in enumerate(validate_loader):
            """
            pileup_feat: [N, 104, 33]
            haplotype_feat: [N, 104, 11]
            gt: [N,]
            zy: [N,]
            """
            x_pileup = pileup_feat.type(torch.FloatTensor)
            x_haplotype = haplotype_feat.type(torch.FloatTensor)
            gt_label = gt.type(torch.LongTensor)
            zy_label = zy.type(torch.LongTensor)
            if x_pileup.shape[0] == 0 or x_pileup.shape[1] == 0 or x_pileup.ndim != 3:
                continue
            if config.training.num_gpu > 0:
                x_pileup = x_pileup.cuda()
                x_haplotype = x_haplotype.cuda()
                gt_label = gt_label.cuda()
                zy_label = zy_label.cuda()
                loss, gt_out,zy_out = model(x_pileup, x_haplotype, gt_label, zy_label)
                gt_logits = torch.softmax(gt_out, 1).detach().cpu().numpy()
                gt_output = np.argmax(gt_logits, axis=1)
                # print('pred:',gt_output[:50].tolist())
                # print('labl:',sub_gt_label.detach().cpu().numpy()[:50].tolist())
                total_acc += sum((gt_output == gt_label.detach().cpu().numpy()).astype(int)) / len(gt_output)
                total_cnt += 1
                gt_confusion_matrix.add(gt_out.data.contiguous().view(-1, config.model.gt_num_class), gt_label.data.contiguous().view(-1))

                total_images += x_pileup.shape[0]

                gt_cm_value = gt_confusion_matrix.value()

                gt_denom = gt_cm_value.sum() if gt_cm_value.sum() > 0 else 1.0

                gt_total_accurate = 0
                for j in range(0, config.model.gt_num_class):
                    gt_total_accurate = gt_total_accurate + gt_cm_value[j][j]
                gt_accuracy = (100.0 * gt_total_accurate) / gt_denom

                total_loss += loss.item()
                avg_loss = total_loss / (step + 1)
    gtcm = gt_confusion_matrix.value()
    gt_tp, gt_fp, gt_fn = np.zeros(config.model.gt_num_class), np.zeros(config.model.gt_num_class), np.zeros(
        config.model.gt_num_class)
    for x in range(len(gtcm)):
        gt_tp[x] += gtcm[x][x]
        gt_fp[x] += sum(gtcm[:, x]) - gtcm[x][x]
        gt_fn[x] += sum(gtcm[x, :]) - gtcm[x][x]
    gt_recall = gt_tp / (gt_tp + gt_fn + 1e-6)
    gt_precision = gt_tp / (gt_tp + gt_fp + 1e-6)
    gt_f1 = 2 * (gt_precision * gt_recall) / (gt_precision + gt_recall + 1e-6)
    gt_recall = gt_recall.mean()
    gt_precision = gt_precision.mean()
    gt_f1 = gt_f1.mean()

    print("AverageLoss: %.5f, GT:|Recall: %.4f, Precision: %.4f, F1: %.4f |, ACC: %.4f" % (avg_loss, gt_recall, gt_precision, gt_f1, total_acc / total_cnt))

    return {'loss': avg_loss,
            'gt_accuracy': gt_accuracy,
            'gt_confusion_matrix': str(gt_confusion_matrix.conf.tolist()),
            'gt_f1_score': gt_f1}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-config', type=str, required=True, help='path to config file')
    parser.add_argument('-model_path', required=True, help='path to trained model')
    parser.add_argument('-bin_paths', required=True, help='directory of bin files')
    parser.add_argument('-reference_path', type=str, required=True, help='path to reference file')
    opt = parser.parse_args()

    configfile = open(opt.config)
    config = AttrDict(yaml.load(configfile, Loader=yaml.FullLoader))
    pred_model = LSTMNetwork(config).cuda()
    pred_model.load_state_dict(torch.load(opt.model_path))

    references = load_reference_file(opt.reference_path)
    eval(config, pred_model, opt.bin_paths, references, config.training.batch_size)


if __name__ == '__main__':
    main()
