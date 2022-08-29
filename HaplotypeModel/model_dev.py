import torch.nn.functional as F
import torch.nn as nn
import torch
import sys
from focal_loss import FocalLoss
from optim import LabelSmoothingLoss

from crnn import CRNN, BidirectionalLSTM, ResCRNN
from resnet import ResNet


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def calculate_percentage(ts):
    # ts: L, N, C
    # return: L, N, 5
    ts_A = ((ts == 1).sum(2) / ((ts != -2).sum(2) + 1e-9)).unsqueeze(2)
    ts_C = ((ts == 2).sum(2) / ((ts != -2).sum(2) + 1e-9)).unsqueeze(2)
    ts_G = ((ts == 3).sum(2) / ((ts != -2).sum(2) + 1e-9)).unsqueeze(2)
    ts_T = ((ts == 4).sum(2) / ((ts != -2).sum(2) + 1e-9)).unsqueeze(2)
    ts_D = ((ts == -1).sum(2) / ((ts != -2).sum(2) + 1e-9)).unsqueeze(2)
    return torch.cat((ts_A, ts_C, ts_G, ts_T, ts_D), dim=2)


class CatModel(nn.Module):
    def __init__(self, pileup_dim, haplotype_dim, hidden_size, nclass):
        super(CatModel, self).__init__()
        self.nclass = nclass
        self.haplotype_net = ResNet(input_channels=haplotype_dim, num_classes=hidden_size)
        self.pileup_net = ResNet(input_channels=pileup_dim, num_classes=hidden_size)
        self.out_layer = nn.Linear(hidden_size*2, nclass) 

        self.gt_crit = LabelSmoothingLoss(nclass, 0.1)

    def forward(self, x_pileup, x_haplotype, gt_target):
        # x_pileup: [N, Depth1, Length1, 4]
        # x_haplotype: [N, Depth2, Length2, 4]

        x_pileup = x_pileup.permute(0, 3, 1, 2) # [N, 4, Depth1, Length1]
        x_haplotype = x_haplotype.permute(0, 3, 1, 2) # [N, 4, Depth2, Length2]
        y_pileup = self.pileup_net(x_pileup)
        y_haplotype = self.haplotype_net(x_haplotype)
        out = self.out_layer(torch.cat((y_pileup, y_haplotype), 1))
        gt_loss = self.gt_crit(out.contiguous().view(-1, self.nclass), gt_target.contiguous().view(-1))
        return gt_loss, out

    def predict(self, x_pileup, x_haplotype):
        x_pileup = x_pileup.permute(0, 3, 1, 2) # [N, 4, Depth1, Length1]
        x_haplotype = x_haplotype.permute(0, 3, 1, 2) # [N, 4, Depth2, Length2]
        y_pileup = self.pileup_net(x_pileup)
        y_haplotype = self.haplotype_net(x_haplotype)
        out = self.out_layer(torch.cat((y_pileup, y_haplotype), 1))
        out = torch.softmax(out, 1)
        return out