import argparse
import os
import yaml
import torch
import numpy as np
from torch.utils.data import DataLoader
from math import log, e
from model import LSTMNetwork
from dataset import PredictDataset
from utils import AttrDict
from options import gt_decoded_labels, zy_decoded_labels, base_idx


def calculate_score(probability):
    p = probability
    tmp = max((-10 * log(e, 10)) * log(((1.0 - p) + 1e-300) / (p + 1e-300)) + 10, 0)
    return float(round(tmp, 2))


def predict(model, testing_paths, batch_size, output_file, device):
    fwriter = open(output_file, 'w')
    # TODO: header需要根据不同的参考序列修改，读取.fai文件可以得到每个contig的长度
    fwriter.write("""\
##fileformat=VCFv4.3
##FILTER=<ID=PASS,Description="All filters passed">
##FILTER=<ID=LowQual,Description="Low quality variant">
##FILTER=<ID=RefCall,Description="Reference call">
##contig=<ID=chr1,length=248956422>
##contig=<ID=chr2,length=242193529>
##contig=<ID=chr3,length=198295559>
##contig=<ID=chr4,length=190214555>
##contig=<ID=chr5,length=181538259>
##contig=<ID=chr6,length=170805979>
##contig=<ID=chr7,length=159345973>
##contig=<ID=chr8,length=145138636>
##contig=<ID=chr9,length=138394717>
##contig=<ID=chr10,length=133797422>
##contig=<ID=chr11,length=135086622>
##contig=<ID=chr12,length=133275309>
##contig=<ID=chr13,length=114364328>
##contig=<ID=chr14,length=107043718>
##contig=<ID=chr15,length=101991189>
##contig=<ID=chr16,length=90338345>
##contig=<ID=chr17,length=83257441>
##contig=<ID=chr18,length=80373285>
##contig=<ID=chr19,length=58617616>
##contig=<ID=chr20,length=64444167>
##contig=<ID=chr21,length=46709983>
##contig=<ID=chr22,length=50818468>
##contig=<ID=chrX,length=156040895>
##contig=<ID=chrY,length=57227415>
##contig=<ID=chrM,length=16569>
##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">
##FORMAT=<ID=GQ,Number=1,Type=Integer,Description="Genotype Quality">
##FORMAT=<ID=DP,Number=1,Type=Integer,Description="Read Depth">
##FORMAT=<ID=AF,Number=A,Type=Float,Description="Allele Frequency">
#CHROM	POS	ID	REF	ALT	QUAL	FILTER	INFO	FORMAT	HG002
""")
    model.eval()
    for testing_file in testing_paths:
        dataset = PredictDataset(datapath=testing_file)
        dl = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)
        for batch in dl:
            ctg_names, positions, reference_bases, position_matrix = batch
            ctg_names = np.array(ctg_names)
            positions = positions.type(torch.LongTensor)
            reference_bases = reference_bases.type(torch.LongTensor)
            feature_tensor = position_matrix.type(torch.FloatTensor).to(device)

            gt_output_, zy_output_ = model.predict(feature_tensor)
            gt_output_ = gt_output_.detach().cpu().numpy()
            zy_output_ = zy_output_.detach().cpu().numpy()
            gt_prob = np.max(gt_output_, axis=1)  # [N]
            zy_prob = np.max(zy_output_, axis=1)  # [N]
            gt_output = np.argmax(gt_output_, axis=1)  # [N]
            zy_output = np.argmax(zy_output_, axis=1)  # [N]
            gt_prob = gt_prob[zy_output >= 0]
            zy_prob = zy_prob[zy_output >= 0]
            ctg_names = ctg_names[zy_output >= 0]
            positions = positions[zy_output >= 0].detach().numpy()
            reference_bases = reference_bases[zy_output >= 0].detach().numpy()
            cov_feature = feature_tensor[zy_output >= 0][:, 16, [0, 1, 2, 3, 9, 10, 11, 12]].cpu().numpy()  # [8]
            gt_output = gt_output[zy_output >= 0]
            zy_output = zy_output[zy_output >= 0]
            for j in range(zy_output.shape[0]):
                try:
                    if gt_output[j] >= 10:
                        continue
                    contig_name = ctg_names[j]
                    spos = positions[j]
                    sref = chr(int(reference_bases[j]))  # 'A', 'C', 'G', 'T'
                    alt = gt_decoded_labels[gt_output[j]]  # AA, AT, ...
                    zy = zy_decoded_labels[zy_output[j]]  # 0/0, 1/1, 0/1
                    cov = cov_feature[j]
                    depth = -1 * cov[np.where(cov < 0)].sum()
                    support_count = 0
                    for base in alt.replace(sref, ''):
                        bidx = base_idx[base]
                        support_count += cov[bidx]
                        support_count += cov[bidx + 4]
                    af = support_count / depth
                    if af > 1.0:
                        af = 1.0

                    gt_qual = calculate_score(gt_prob[j])
                    zy_qual = calculate_score(zy_prob[j])
                    qual = min(gt_qual, zy_qual)

                    alt = alt.replace(sref, '')
                    if len(alt) == 0:
                        zy = '0/0'
                        fwriter.write(
                            "{0}\t{1}\t.\t{2}\t{3}\t{4}\t{5}\t{6}\t{7}\t{8}\n".format(contig_name, spos, sref, sref,
                                                                                      str(qual), \
                                                                                      'RefCall', '.', 'GT:GQ:DP:AF',
                                                                                      zy + ":%s:%d:%f" % (
                                                                                          str(int(qual)), depth, af)))
                        continue
                    elif len(alt) == 1:
                        alt = alt
                    else:
                        if alt[0] == alt[1]:  # AA -> A
                            alt = alt[0]
                        alt = ','.join(list(alt))

                    if len(alt) >= 3 and zy_output[j] != 2:
                        zy = '1/2'

                    """
                    if alt == sref and zy_output[j] != 0:
                        # 预测结果是纯合参考，丢弃
                        zy = '0/0'
                        continue
                    if alt != sref and zy_output[j] == 0:
                        # 不确定是纯合突变还是杂合突变，丢弃
                        continue
                    """

                    if alt != sref and zy_output[j] == 0:
                        zy = '0/1'

                    fwriter.write(
                        "{0}\t{1}\t.\t{2}\t{3}\t{4}\t{5}\t{6}\t{7}\t{8}\n".format(contig_name, spos, sref, alt,
                                                                                  str(qual), \
                                                                                  'PASS', '.', 'GT:GQ:DP:AF',
                                                                                  zy + ":%s:%d:%f" % (
                                                                                      str(int(qual)), depth, af)))
                except:
                    continue
    fwriter.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-config', type=str, required=True, help='path to config file')
    parser.add_argument('-model_path', required=True, help='path to trained model')
    parser.add_argument('-data', required=True, help='directory of bin files')
    # parser.add_argument('-contig', required=True, help='contig name of the input bin files')
    parser.add_argument('-output', required=True, help='output vcf file')
    parser.add_argument('-batch_size', type=int, default=1000, help='batch size')
    parser.add_argument('--no_cuda', action="store_true", help='If running on cpu device, set the argument.')
    opt = parser.parse_args()
    device = torch.device('cuda' if not opt.no_cuda else 'cpu')
    configfile = open(opt.config)
    config = AttrDict(yaml.load(configfile, Loader=yaml.FullLoader))
    pred_model = LSTMNetwork(config.model).to(device)
    checkpoint = torch.load(opt.model_path, map_location=device)
    pred_model.encoder.load_state_dict(checkpoint['encoder'])
    pred_model.forward_layer.load_state_dict(checkpoint['forward_layer'])
    testing_paths = [opt.data + '/' + fname for fname in os.listdir(opt.data)]
    predict(pred_model, testing_paths, opt.batch_size, opt.output, device)


if __name__ == '__main__':
    main()
