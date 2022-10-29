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

def write_head(reference_index_file, fwriter):
    fwriter.write("##fileformat=VCFv4.3"+'\n')
    fwriter.write("##FILTER=<ID=PASS,Description=\"All filters passed\">"+'\n')
    fwriter.write("##FILTER=<ID=RefCall,Description=\"Reference call\">"+'\n')
    with open(reference_index_file) as f:
        for line in f.readline():
            contig_name = line.strip().split()[0]
            contig_length = line.strip().split()[0]
            fwriter.write('##contig=<ID={},length={}>'.format(contig_name, contig_length) + '\n')
    
    fwriter.write("##FORMAT=<ID=GT,Number=1,Type=String,Description=\"Genotype\">"+'\n')
    fwriter.write("##FORMAT=<ID=GQ,Number=1,Type=Integer,Description=\"Genotype Quality\">"+'\n')
    fwriter.write("##FORMAT=<ID=DP,Number=1,Type=Integer,Description=\"Read Depth\">"+'\n')
    fwriter.write("##FORMAT=<ID=AF,Number=A,Type=Float,Description=\"Allele Frequency\">"+'\n')
    fwriter.write("#CHROM	POS	ID	REF	ALT	QUAL	FILTER	INFO	FORMAT	Sample"+'\n')



def calculate_score(probability):
    p = probability
    tmp = max((-10 * log(e, 10)) * log(((1.0 - p) + 1e-300) / (p + 1e-300)) + 10, 0)
    return float(round(tmp, 2))


def predict(model, testing_paths, reference_index_file, batch_size, output_file, device):
    fwriter = open(output_file, 'w')
    write_head(reference_index_file, fwriter)
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
                    
                    ## 为了保证PileupModel尽可能少漏掉SNP位点，如果gt和zy中有一个表明是变异，则以变异的结果为准输出到vcf

                    alt = alt.replace(sref, '')
                    if len(alt) == 0:
                        if zy == '0/0':
                            fwriter.write(
                            "{0}\t{1}\t.\t{2}\t{3}\t{4}\t{5}\t{6}\t{7}\t{8}\n".format(contig_name, spos, sref, sref,
                                                                                      str(qual), \
                                                                                      'RefCall', '.', 'GT:GQ:DP:AF',
                                                                                      zy + ":%s:%d:%f" % (
                                                                                          str(int(qual)), depth, af)))
                        elif zy=='1/1':
                            max_ti, max_v = -1,-1
                            for ti in [0,4,7,9]:
                                if gt_decoded_labels[ti][0] == sref:    ## AA == A
                                    continue
                                if gt_output[ti]>max_v:
                                    max_v = gt_output[ti]
                                    max_ti = ti
                            new_alt = gt_decoded_labels[max_ti][0]
                            fwriter.write(
                            "{0}\t{1}\t.\t{2}\t{3}\t{4}\t{5}\t{6}\t{7}\t{8}\n".format(contig_name, spos, sref, new_alt,
                                                                                      str(zy_qual), \
                                                                                      'PASS', '.', 'GT:GQ:DP:AF',
                                                                                      zy + ":%s:%d:%f" % (
                                                                                          str(int(zy_qual)), depth, af)))
                        elif zy=='0/1':
                            max_ti, max_v = -1,-1
                            for ti in [1,2,3,5,6,8]:
                                if gt_output[ti]>max_v:
                                    max_v = gt_output[ti]
                                    max_ti = ti
                            if gt_decoded_labels[max_ti][0]==sref:
                                new_alt = gt_decoded_labels[max_ti][1]
                            else:
                                new_alt = gt_decoded_labels[max_ti][0]
                            fwriter.write(
                            "{0}\t{1}\t.\t{2}\t{3}\t{4}\t{5}\t{6}\t{7}\t{8}\n".format(contig_name, spos, sref, new_alt,
                                                                                      str(zy_qual), \
                                                                                      'PASS', '.', 'GT:GQ:DP:AF',
                                                                                      zy + ":%s:%d:%f" % (
                                                                                          str(int(zy_qual)), depth, af)))
                        continue
                    elif len(alt) == 1:
                        alt = alt
                    else:
                        if alt[0] == alt[1]:  # AA -> A
                            alt = alt[0]
                        alt = ','.join(list(alt))

                    if len(alt) >= 3 and zy_output[j] != 2:
                        zy = '1/2'

                    if alt == sref and zy_output[j] != 0:
                        # 预测结果是纯合参考，丢弃
                        if zy=='1/1':
                            max_ti, max_v = -1,-1
                            for ti in [0,4,7,9]:
                                if gt_decoded_labels[ti][0] == sref:    ## AA == A
                                    continue
                                if gt_output[ti]>max_v:
                                    max_v = gt_output[ti]
                                    max_ti = ti
                            new_alt = gt_decoded_labels[max_ti][0]
                            fwriter.write(
                            "{0}\t{1}\t.\t{2}\t{3}\t{4}\t{5}\t{6}\t{7}\t{8}\n".format(contig_name, spos, sref, new_alt,
                                                                                      str(zy_qual), \
                                                                                      'PASS', '.', 'GT:GQ:DP:AF',
                                                                                      zy + ":%s:%d:%f" % (
                                                                                          str(int(zy_qual)), depth, af)))
                        elif zy=='0/1':
                            max_ti, max_v = -1,-1
                            for ti in [1,2,3,5,6,8]:
                                if gt_output[ti]>max_v:
                                    max_v = gt_output[ti]
                                    max_ti = ti
                            if gt_decoded_labels[max_ti][0]==sref:
                                new_alt = gt_decoded_labels[max_ti][1]
                            else:
                                new_alt = gt_decoded_labels[max_ti][0]
                            fwriter.write(
                            "{0}\t{1}\t.\t{2}\t{3}\t{4}\t{5}\t{6}\t{7}\t{8}\n".format(contig_name, spos, sref, new_alt,
                                                                                      str(zy_qual), \
                                                                                      'PASS', '.', 'GT:GQ:DP:AF',
                                                                                      zy + ":%s:%d:%f" % (
                                                                                          str(int(zy_qual)), depth, af))) 
                        continue
                    if alt != sref and zy_output[j] == 0:
                        # 不确定是纯合突变还是杂合突变，则按纯和突变为结果，不影响后面分型
                        fwriter.write(
                            "{0}\t{1}\t.\t{2}\t{3}\t{4}\t{5}\t{6}\t{7}\t{8}\n".format(contig_name, spos, sref, alt,
                                                                                      str(gt_qual), \
                                                                                      'PASS', '.', 'GT:GQ:DP:AF',
                                                                                      zy + ":%s:%d:%f" % (
                                                                                          str(int(gt_qual)), depth, af))) 
                        continue

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
    parser.add_argument('-reference', required=True, help='path to reference file')
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
    testing_paths = [opt.data + '/' + fname for fname in os.listdir(opt.data) if fname.endswith('.bin')]

    assert os.path.exists(opt.reference+'.fai'), 'reference index file does not exist.'
    predict(pred_model, testing_paths, opt.reference+'.fai', opt.batch_size, opt.output, device)


if __name__ == '__main__':
    main()
