import torch.nn.functional as F
import torch.nn as nn
import torch
import sys
from focal_loss import FocalLoss
from optim import LabelSmoothingLoss

from crnn import CRNN, BidirectionalLSTM, ResCRNN


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


def build_encoder(config):
    if config.enc.type == 'lstm':
        return BaseEncoder(
            input_size=config.feature_dim,
            hidden_size=config.enc.hidden_size,
            output_size=config.enc.output_size,
            n_layers=config.enc.n_layers,
            dropout=config.dropout,
            bidirectional=config.enc.bidirectional
        )
    else:
        raise NotImplementedError


class ForwardLayer(nn.Module):
    def __init__(self, input_size, inner_size, zy_class, win_length):
        super(ForwardLayer, self).__init__()
        self.win_length = win_length
        self.dense = nn.Linear(input_size, inner_size, bias=True)
        self.tanh = nn.Tanh()
        self.zygosity_layer = nn.Linear(inner_size, zy_class, bias=True)

    def forward(self, inputs):
        out = self.tanh(self.dense(inputs))  # [batch, length, hidden*2]
        out = out[:, self.win_length // 2, :]  # [batch, hidden*2]
        zy_outputs = self.zygosity_layer(out)
        return zy_outputs


def build_forward(config):
    return ForwardLayer(config.enc.output_size,
                        config.joint.inner_size,
                        config.zy_num_class,
                        win_length=11)


class LSTMNetwork(nn.Module):
    def __init__(self, config):
        super(LSTMNetwork, self).__init__()
        # define encoder
        self.config = config
        self.encoder = build_encoder(config)
        self.forward_layer = build_forward(config)
        # self.gt_crit = LabelSmoothingLoss(config.gt_num_class, 0.1)
        self.zy_crit = LabelSmoothingLoss(config.zy_num_class, 0.1)

    def forward(self, inputs, zy_target):
        # inputs: N x L x c
        enc_state, _ = self.encoder(inputs)  # [N, L, o]
        zy_logits = self.forward_layer(enc_state)
        zy_loss = self.zy_crit(zy_logits.contiguous().view(-1, self.config.zy_num_class),
                               zy_target.contiguous().view(-1))
        loss = zy_loss

        return loss, zy_logits

    def predict(self, inputs):
        enc_state, _ = self.encoder(inputs)  # [N, L, o]
        zy_logits = self.forward_layer(enc_state)
        return zy_logits


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


# class RNN(nn.Module):
#     def __init__(self, nIn, nh, nOut):
#         super(RNN, self).__init__()
#         self.rnn = nn.Sequential(
#             BidirectionalLSTM(nIn, nh, nh),
#             BidirectionalLSTM(nh, 2*nh, nOut))

#     def forward(self, input):
#         # input: L,N,C
#         output = self.rnn(input)    # # [L, N, nclass]
#         return output


class RNN(nn.Module):
    def __init__(self, nIn, nh, nOut):
        super(RNN, self).__init__()
        self.rnn = nn.LSTM(input_size=nIn, hidden_size=nh, num_layers=3, bidirectional=True, dropout=0.5)
        self.out_layer = nn.Linear(nh * 2, nOut)

    def forward(self, input):
        # input: L,N,C
        output, _ = self.rnn(input)  # # [L, N, nh*2]
        output = self.out_layer(output)  # [L,N,nOut]
        return output


class CNN(nn.Module):

    def __init__(self, nc, nOut, leakyRelu=False):
        super(CNN, self).__init__()

        cnn = nn.Sequential()

        def convRelu2(i, nIn, nOut, ks, ss, ps, batchNormalization=False):
            cnn.add_module('conv{0}'.format(i), nn.Conv2d(nIn, nOut, ks, ss, ps))
            if batchNormalization:
                cnn.add_module('batchnorm{0}'.format(i), nn.BatchNorm2d(nOut))
            if leakyRelu:
                cnn.add_module('relu{0}'.format(i), nn.LeakyReLU(0.2, inplace=True))
            else:
                cnn.add_module('relu{0}'.format(i), nn.ReLU(True))

        convRelu2(0, nIn=nc, nOut=32, ks=3, ss=1, ps=1, batchNormalization=False)  # 32x25x4
        cnn.add_module('pooling{0}'.format(0),
                       nn.MaxPool2d(kernel_size=(2, 3), stride=(2, 1), padding=(0, 1)))  # 32x12x4
        convRelu2(1, nIn=32, nOut=64, ks=3, ss=1, ps=1, batchNormalization=False)  # 64x12x4
        cnn.add_module('pooling{0}'.format(1),
                       nn.MaxPool2d(kernel_size=(2, 3), stride=(2, 1), padding=(0, 1)))  # 64x6x4
        convRelu2(2, nIn=64, nOut=128, ks=3, ss=1, ps=1, batchNormalization=True)  # 128x6x4
        convRelu2(3, nIn=128, nOut=128, ks=3, ss=1, ps=1, batchNormalization=False)  # 128x6x4
        cnn.add_module('pooling{0}'.format(2),
                       nn.MaxPool2d(kernel_size=(3, 3), stride=(3, 1), padding=(0, 1)))  # 128x3x4
        convRelu2(4, nIn=128, nOut=256, ks=3, ss=1, ps=1, batchNormalization=True)  # 256x3x4
        convRelu2(5, nIn=256, nOut=256, ks=3, ss=1, ps=1, batchNormalization=False)  # 256x3x4
        cnn.add_module('pooling{0}'.format(3),
                       nn.MaxPool2d(kernel_size=(2, 3), stride=(2, 1), padding=(0, 1)))  # 256x1x4
        self.cnn = cnn
        self.dnn = nn.Sequential(
            nn.Linear(1024, 512),
            nn.Dropout(p=0.5),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.Dropout(p=0.5),
            nn.ReLU(),
            nn.Linear(512, nOut))

    def forward(self, input):
        # conv features
        conv = self.cnn(input)
        b, c, h, w = conv.size()
        assert h == 1, "the height of conv must be 1"
        conv = conv.squeeze(2)
        conv = conv.reshape(b, -1)  # [b, 1024]

        # dnn features
        output = self.dnn(conv)

        return output


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
    def __init__(self, nclass, pileup_length=11, haplotype_length=11, use_g0=True, use_g1=True):
        super(CatModel, self).__init__()
        self.nclass = nclass
        self.use_pileup = use_g0
        self.use_haplotype = use_g1
        assert pileup_length % 2 == 1
        assert haplotype_length % 2 == 1
        assert pileup_length == haplotype_length
        self.pileup_length = pileup_length
        self.haplotype_length = haplotype_length
        # self.surrounding_base = RNN(nIn=5, nh=256, nOut=512)
        # self.haplotype_base = RNN(nIn=40, nh=256, nOut=512)
        if use_g0 and not use_g1:
            self.haplotype_base = ResCRNN(nc=5, nclass=256, nh=256)
            self.haplotype_percentage = RNN(nIn=10, nh=256, nOut=256)
        elif not use_g0 and use_g1:
            self.haplotype_base = ResCRNN(nc=5, nclass=256, nh=256)
            self.haplotype_percentage = RNN(nIn=10, nh=256, nOut=256)
        elif use_g0 and use_g1:
            self.haplotype_base = ResCRNN(nc=10, nclass=256, nh=256)
            self.haplotype_percentage = RNN(nIn=20, nh=256, nOut=256)

        # if use_g0 and use_g1:
        #     self.out_layer = nn.Linear(1024, nclass)
        # elif use_g0 and not use_g1:
        #     self.out_layer = nn.Linear(512, nclass)
        # elif not use_g0 and use_g1:
        #     self.out_layer = nn.Linear(512, nclass)
        # else:
        #     print("create model unsuccessfully.")
        #     sys.exit(1)

        self.out_layer = nn.Linear(512, nclass)

        self.gt_crit = LabelSmoothingLoss(nclass, 0.1)

    def forward(self, g0, g1, g2, g3, gt_target):
        # g0: [N, 40, pileup_length, 5]
        # g1: [N, 40, haplotype_length, 5]
        # g2: [N, 4, 25, 2]
        # g3: [N, 4, 25, 2]

        if self.use_pileup:
            g0_p = g0.permute(2, 0, 1, 3)  # [pl,N,40,5]
            g0_s = g0.permute(0, 3, 1, 2)  # [N, 5, 40, pl]
            g0_tag1 = g0_p[:, :, :20, 0]  # [pl,N,20]
            g0_tag2 = g0_p[:, :, 20:, 0]  # [pl,N,20]
            g0_tag1 = calculate_percentage(g0_tag1)  # [pl,N,5]
            g0_tag2 = calculate_percentage(g0_tag2)  # [pl,N,5]

        if self.use_haplotype:
            g1_p = g1.permute(2, 0, 1, 3)  # [hl,N,40,5]
            g1_s = g1.permute(0, 3, 1, 2)  # [N, 5, 40, hl]
            g1_tag1 = g1_p[:, :, :20, 0]  # [hl,N,20]
            g1_tag2 = g1_p[:, :, 20:, 0]  # [hl,N,20]
            g1_tag1 = calculate_percentage(g1_tag1)  # [hl,N,5]
            g1_tag2 = calculate_percentage(g1_tag2)  # [hl,N,5]

        if self.use_pileup and self.use_haplotype:
            g0_g1_tag1_tag2_cat = torch.cat((g0_tag1, g0_tag2, g1_tag1, g1_tag2), 2)
            g0_g1_p_out = self.haplotype_percentage(g0_g1_tag1_tag2_cat)[(self.haplotype_length - 1) // 2, :, :]
            g0_g1_s_out = self.haplotype_base(torch.cat((g0_s, g1_s), 1))[(self.haplotype_length - 1) // 2]
            out = self.out_layer(torch.cat((g0_g1_p_out, g0_g1_s_out), 1))
        elif self.use_pileup and not self.use_haplotype:
            g0_tag1_tag2_cat = torch.cat((g0_tag1, g0_tag2), 2)  # [pl, N, 10]
            g0_p_out = self.haplotype_percentage(g0_tag1_tag2_cat)[(self.pileup_length - 1) // 2, :, :]  # [N, 256]
            g0_s_out = self.haplotype_base(g0_s)[(self.pileup_length - 1) // 2]  # [N, 256]
            out = self.out_layer(torch.cat((g0_p_out, g0_s_out), 1))
        elif not self.use_pileup and self.use_haplotype:
            g1_tag1_tag2_cat = torch.cat((g1_tag1, g1_tag2), 2)  # [hl, N, 10]
            g1_p_out = self.haplotype_percentage(g1_tag1_tag2_cat)[(self.haplotype_length - 1) // 2, :, :]  # [N, 256]
            g1_s_out = self.haplotype_base(g1_s)[(self.haplotype_length - 1) // 2]  # [N, 256]
            out = self.out_layer(torch.cat((g1_p_out, g1_s_out), 1))
        else:
            print("model forward unsuccessfully.")
            sys.exit(1)

        gt_loss = self.gt_crit(out.contiguous().view(-1, self.nclass), gt_target.contiguous().view(-1))

        """
        # g0_p shape: torch.Size([23, 27, 40, 5])
        # g1_p shape: torch.Size([11, 27, 40, 5])
        # g0_s shape: torch.Size([27, 5, 40, 23])
        # g1_s shape: torch.Size([27, 5, 40, 11])
        # g0_tag1 shape: torch.Size([23, 27, 5])
        # g1_tag1 shape: torch.Size([11, 27, 5])
        # g0_tag1_tag2_cat shape: torch.Size([23, 27, 10])
        # g1_tag1_tag2_cat shape: torch.Size([11, 27, 10])
        # g0_p_out shape: torch.Size([27, 256])
        # g1_p_out shape: torch.Size([27, 256])
        # g0_s_out shape: torch.Size([27, 256])
        # g1_s_out shape: torch.Size([27, 256])
        # out shape: torch.Size([27, 10])

        print('g0_p shape:', g0_p.shape)
        print('g1_p shape:', g1_p.shape)
        print('g0_s shape:', g0_s.shape)
        print('g1_s shape:', g1_s.shape)
        print('g0_tag1 shape:', g0_tag1.shape)
        print('g1_tag1 shape:', g1_tag1.shape)
        print('g0_tag1_tag2_cat shape:', g0_tag1_tag2_cat.shape)
        print('g1_tag1_tag2_cat shape:', g1_tag1_tag2_cat.shape)
        print('g0_p_out shape:', g0_p_out.shape)
        print('g1_p_out shape:', g1_p_out.shape)
        print('g0_s_out shape:', g0_s_out.shape)
        print('g1_s_out shape:', g1_s_out.shape)
        print('out shape:', out.shape)
        """
        return gt_loss, out

    def predict(self, g0, g1, g2, g3):
        if self.use_pileup:
            g0_p = g0.permute(2, 0, 1, 3)  # [pl,N,40,5]
            g0_s = g0.permute(0, 3, 1, 2)  # [N, 5, 40, pl]
            g0_tag1 = g0_p[:, :, :20, 0]  # [pl,N,20]
            g0_tag2 = g0_p[:, :, 20:, 0]  # [pl,N,20]
            g0_tag1 = calculate_percentage(g0_tag1)  # [pl,N,5]
            g0_tag2 = calculate_percentage(g0_tag2)  # [pl,N,5]
            # g0_tag1_tag2_cat = torch.cat((g0_tag1, g0_tag2), 2)  # [pl, N, 10]
            # g0_p_out = self.haplotype_percentage_g0(g0_tag1_tag2_cat)[(self.pileup_length - 1) // 2, :, :]  # [N, 256]
            # g0_s_out = self.haplotype_base_g0(g0_s)[(self.pileup_length - 1) // 2]  # [N, 256]

        if self.use_haplotype:
            g1_p = g1.permute(2, 0, 1, 3)  # [hl,N,40,5]
            g1_s = g1.permute(0, 3, 1, 2)  # [N, 5, 40, hl]
            g1_tag1 = g1_p[:, :, :20, 0]  # [hl,N,20]
            g1_tag2 = g1_p[:, :, 20:, 0]  # [hl,N,20]
            g1_tag1 = calculate_percentage(g1_tag1)  # [hl,N,5]
            g1_tag2 = calculate_percentage(g1_tag2)  # [hl,N,5]
            # g1_tag2_tag2_cat = torch.cat((g1_tag1, g1_tag2), 2)  # [hl, N, 10]
            # g1_p_out = self.haplotype_percentage_g1(g1_tag2_tag2_cat)[(self.haplotype_length - 1) // 2, :,
            #            :]  # [N, 256]
            # g1_s_out = self.haplotype_base_g1(g1_s)[(self.haplotype_length - 1) // 2]  # [N, 256]

        if self.use_pileup and self.use_haplotype:
            g0_g1_tag1_tag2_cat = torch.cat((g0_tag1, g0_tag2, g1_tag1, g1_tag2), 2)
            g0_g1_p_out = self.haplotype_percentage(g0_g1_tag1_tag2_cat)[(self.haplotype_length - 1) // 2, :, :]
            g0_g1_s_out = self.haplotype_base(torch.cat((g0_s, g1_s), 1))[(self.haplotype_length - 1) // 2]
            out = self.out_layer(torch.cat((g0_g1_p_out, g0_g1_s_out), 1))
        elif self.use_pileup and not self.use_haplotype:
            g0_tag1_tag2_cat = torch.cat((g0_tag1, g0_tag2), 2)  # [pl, N, 10]
            g0_p_out = self.haplotype_percentage(g0_tag1_tag2_cat)[(self.pileup_length - 1) // 2, :, :]  # [N, 256]
            g0_s_out = self.haplotype_base(g0_s)[(self.pileup_length - 1) // 2]  # [N, 256]
            out = self.out_layer(torch.cat((g0_p_out, g0_s_out), 1))
        elif not self.use_pileup and self.use_haplotype:
            g1_tag1_tag2_cat = torch.cat((g1_tag1, g1_tag2), 2)  # [hl, N, 10]
            g1_p_out = self.haplotype_percentage(g1_tag1_tag2_cat)[(self.haplotype_length - 1) // 2, :, :]  # [N, 256]
            g1_s_out = self.haplotype_base(g1_s)[(self.haplotype_length - 1) // 2]  # [N, 256]
            out = self.out_layer(torch.cat((g1_p_out, g1_s_out), 1))
        else:
            print("model predict unsuccessfully.")
            sys.exit(1)

        out = torch.softmax(out, 1)
        return out
