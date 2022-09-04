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
        # x_pileup: [N, 4, Depth1, Length1]
        # x_haplotype: [N, 4, Depth2, Length2]
        y_pileup = self.pileup_net(x_pileup)
        y_haplotype = self.haplotype_net(x_haplotype)
        out = self.out_layer(torch.cat((y_pileup, y_haplotype), 1))
        gt_loss = self.gt_crit(out.contiguous().view(-1, self.nclass), gt_target.contiguous().view(-1))
        return gt_loss, out

    def predict(self, x_pileup, x_haplotype):
        y_pileup = self.pileup_net(x_pileup)
        y_haplotype = self.haplotype_net(x_haplotype)
        out = self.out_layer(torch.cat((y_pileup, y_haplotype), 1))
        out = torch.softmax(out, 1)
        return out


class BaseEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, n_layers, dropout=0.2, bidirectional=True):
        super(BaseEncoder, self).__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=bidirectional
        )

        self.output_proj = nn.Linear(2 * hidden_size if bidirectional else hidden_size,
                                     output_size,
                                     bias=True)

    def forward(self, inputs):
        assert inputs.dim() == 3

        self.lstm.flatten_parameters()
        outputs, hidden = self.lstm(inputs)

        logits = self.output_proj(outputs)  # N, L, output_size

        return logits, hidden

class ForwardLayer(nn.Module):
    def __init__(self, input_size, inner_size, gt_class, zy_class, pileup_length, haplotype_length):
        super(ForwardLayer, self).__init__()
        self.pileup_length = pileup_length
        self.haplotype_length = haplotype_length
        self.dense = nn.Linear(input_size, inner_size, bias=True)
        self.tanh = nn.Tanh()
        self.genotype_layer = nn.Linear(inner_size, gt_class, bias=True)
        self.zygosity_layer = nn.Linear(inner_size, zy_class, bias=True)

    def forward(self, pileup_inputs, haplotype_inputs):
        ## pileup_inputs: 
        ## haplotype_inputs:
        pileup_inputs = pileup_inputs[:,self.pileup_length//2, :]   # [N, 128]
        haplotype_inputs = haplotype_inputs[:,self.haplotype_length//2, :]  # [N, 128]
        inputs = torch.cat((pileup_inputs, haplotype_inputs), axis=1)  # [N, 256]
        out = self.tanh(self.dense(inputs))  # [batch, inner_size]
        gt_outputs = self.genotype_layer(out)
        zy_outputs = self.zygosity_layer(out)
        return gt_outputs, zy_outputs


class LSTMNetwork(nn.Module):
    def __init__(self, config):
        super(LSTMNetwork, self).__init__()
        # define encoder
        self.gt_class = config.model.gt_num_class
        self.zy_class = config.model.zy_num_class
        self.pileup_encoder = BaseEncoder(input_size=config.model.pileup_dim,hidden_size=config.model.hidden_size,output_size=config.model.hidden_size,n_layers=config.model.lstm_layers,dropout=config.model.dropout,bidirectional=True)  # N, 33, 128
        self.haplotype_encoder = BaseEncoder(input_size=config.model.haplotype_dim,hidden_size=config.model.hidden_size,output_size=config.model.hidden_size,n_layers=config.model.lstm_layers,dropout=config.model.dropout,bidirectional=True)   # N, 11, 128
        self.forward_layer = ForwardLayer(input_size=config.model.hidden_size*2,inner_size=config.model.hidden_size,gt_class=config.model.gt_num_class,zy_class=config.model.zy_num_class,pileup_length=config.model.pileup_length,haplotype_length=config.model.haplotype_length)
        self.gt_crit = LabelSmoothingLoss(config.model.gt_num_class, 0.1)
        self.zy_crit = LabelSmoothingLoss(config.model.zy_num_class, 0.1)

    def forward(self, pileup_x, haplotype_x, gt_target, zy_target):
        # pileup_x: N, 52, 33
        # haplotype_x: N, 52, 11
        pileup_x = pileup_x.permute(0, 2, 1)
        haplotype_x = haplotype_x.permute(0, 2, 1)
        enc_state1, _ = self.pileup_encoder(pileup_x)  # [N, L, o]
        enc_state2, _ = self.haplotype_encoder(haplotype_x)  # [N, L, o]
        gt_logits, zy_logits = self.forward_layer(enc_state1, enc_state2)
        gt_loss = self.gt_crit(gt_logits.contiguous().view(-1, self.gt_class), gt_target.contiguous().view(-1))
        zy_loss = self.zy_crit(zy_logits.contiguous().view(-1, self.zy_class), zy_target.contiguous().view(-1))
        loss = gt_loss + zy_loss
        return loss, gt_logits, zy_logits

    def predict(self, pileup_x, haplotype_x):
        # pileup_x: N, 52, 33
        # haplotype_x: N, 52, 11
        pileup_x = pileup_x.permute(0, 2, 1)
        haplotype_x = haplotype_x.permute(0, 2, 1)
        enc_state1, _ = self.pileup_encoder(pileup_x)  # [N, L, o]
        enc_state2, _ = self.haplotype_encoder(haplotype_x)  # [N, L, o]
        gt_logits, zy_logits = self.forward_layer(enc_state1, enc_state2)
        gt_logits = torch.softmax(gt_logits, 1)
        zy_logits = torch.softmax(zy_logits, 1)
        return gt_logits, zy_logits


# if __name__=="__main__":
#     net = LSTMNetwork(10, 3)
#     print(net)
#     pileup_x = torch.randn(2, 52, 33)
#     haplotype_x = torch.randn(2, 52, 11)
#     a,b = net.predict(pileup_x, haplotype_x)
#     print(a.shape, b.shape)