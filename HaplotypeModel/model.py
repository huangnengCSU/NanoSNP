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
    def __init__(self, nc0, nc1, nc2, nclass, nh, use_g0=True, use_g1=True, use_g2=True, use_g3=True):
        super(CatModel, self).__init__()
        self.nclass = nclass
        # self.surrounding_base = RNN(nIn=5, nh=256, nOut=512)
        # self.haplotype_base = RNN(nIn=40, nh=256, nOut=512)
        self.haplotype_base = ResCRNN(nc=10, nclass=256, nh=256)
        self.haplotype_percentage = RNN(nIn=20, nh=256, nOut=256)
        # self.cnn_edge = CNN(nc2, nOut=256)
        # self.cnn_pair = CNN(nc2, nOut=256)

        # num_of_out = int(use_g0)+int(use_g1)+int(use_g2)+int(use_g3)
        # self.out_layer = nn.Linear(nclass * num_of_out, nclass)
        # self.num_of_out = num_of_out
        # self.use_g0 = use_g0
        # self.use_g1 = use_g1
        # self.use_g2 = use_g2
        # self.use_g3 = use_g3

        # if self.use_g0 and self.use_g1 and self.use_g2 and self.use_g3:
        #     self.out_layer = nn.Linear(2560, nclass)
        # elif self.use_g0 and self.use_g1 and self.use_g2 and not self.use_g3:
        #     self.out_layer = nn.Linear(2304, nclass)
        # elif self.use_g0 and self.use_g1 and not self.use_g2 and not self.use_g3:
        #     self.out_layer = nn.Linear(2048, nclass)
        # elif self.use_g0 and not self.use_g1 and not self.use_g2 and not self.use_g3:
        #     self.out_layer = nn.Linear(1024, nclass)
        # elif not self.use_g0 and self.use_g1 and not self.use_g2 and not self.use_g3:
        #     self.out_layer = nn.Linear(1024, nclass)
        # else:
        #     print("create model unsuccessfully.")
        #     sys.exit(1)

        self.out_layer = nn.Linear(512, nclass)

        self.gt_crit = LabelSmoothingLoss(nclass, 0.1)

    def forward(self, g0, g1, g2, g3, gt_target):
        # g0: [N, 40, 11, 5]
        # g1: [N, 40, 5, 5]
        # g2: [N, 4, 25, 2]
        # g3: [N, 4, 25, 2]

        g0_p = g0.permute(2, 0, 1, 3)   # [11,N,40,5]
        g0_s = g0.permute(0, 3, 1, 2) # [N, 5, 40, 11]

        g1_p = g1.permute(2, 0, 1, 3)   # [11,N,40,5]
        g1_s = g1.permute(0, 3, 1, 2) # [N, 5, 40, 11]
        

        # g0 = g0.permute(2, 0, 1, 3)  # [11, N, 40, 5]
        # g1 = g1.permute(2, 0, 1, 3)  # [5, N, 40, 5]
        # g2 = g2.permute(0, 3, 2, 1)  # [N, 2, 25, 4]
        # g3 = g3.permute(0, 3, 2, 1)  # [N, 2, 25, 4]

        # g0_tag1 = g0[:, :, :20, 0]  # [11,N,20]
        # g0_tag2 = g0[:, :, 20:, 0]  # [11,N,20]
        g0_tag1 = g0_p[:, :, :20, 0]  # [5,N,20]
        g0_tag2 = g0_p[:, :, 20:, 0]  # [5,N,20]

        g0_tag1 = calculate_percentage(g0_tag1) #[11,N,5]
        g0_tag2 = calculate_percentage(g0_tag2) #[11,N,5]


        g1_tag1 = g1_p[:, :, :20, 0]  # [5,N,20]
        g1_tag2 = g1_p[:, :, 20:, 0]  # [5,N,20]

        g1_tag1 = calculate_percentage(g1_tag1) #[11,N,5]
        g1_tag2 = calculate_percentage(g1_tag2) #[11,N,5]

        g0_g1_tag1_tag2_cat = torch.cat((g0_tag1,g0_tag2,g1_tag1,g1_tag2),2)   # [L, N, 20]


        
        g0_g1_p_out = self.haplotype_percentage(g0_g1_tag1_tag2_cat)[5, :, :]  # [N, 256]



        # g0_tag1 = calculate_percentage(g0[:, :, :20, 0])  # [11,N,5]
        # g0_tag2 = calculate_percentage(g0[:, :, 20:, 0])  # [11,N,5]
        # g1_tag1 = calculate_percentage(g1[:, :, :20, 0])  # [5,N,5]
        # g1_tag2 = calculate_percentage(g1[:, :, 20:, 0])  # [5,N,5]


        # g0_tag1_out = self.surrounding_base(g0_tag1)  # [11, N, nOut]
        # g0_tag1_out = g0_tag1_out[5, :, :]
        # g0_tag2_out = self.surrounding_base(g0_tag2)  # [11, N, nOut]
        # g0_tag2_out = g0_tag2_out[5, :, :]

        # g1_tag1_tag2_cat = torch.cat((g1_tag1, g1_tag2), 2)   # [L, N, C]
        # g1_tag1_tag2_cat = g1[:, :, :, 0]  # [L, N, C]
        # out = self.haplotype_base(g1_tag1_tag2_cat)[2, :, :]

        # g1_tag1_out = self.haplotype_base(g1_tag1)  # [5, N, nOut]
        # g1_tag1_out = g1_tag1_out[2, :, :]
        # g1_tag2_out = self.haplotype_base(g1_tag2)  # [5, N, nOut]
        # g1_tag2_out = g1_tag2_out[2, :, :]

        g0_g1_s_out = self.haplotype_base(torch.cat((g0_s, g1_s),1))[5]   # [N, 256]

        out = self.out_layer(torch.cat((g0_g1_p_out, g0_g1_s_out), 1))





        # g0_o = g0_o[5,:,:]
        # g1_o = self.crnn_base(g1)   # [5, N, nclass]
        # g1_o = g1_o[2,:,:]

        # g2_o = self.cnn_edge(g2)
        # g3_o = self.cnn_pair(g3)

        # if self.use_g0 and self.use_g1 and self.use_g2 and self.use_g3:
        #     out = torch.cat((g0_tag1_out, g0_tag2_out, g1_tag1_out, g1_tag2_out, g2_o, g3_o), dim=1)
        # elif self.use_g0 and self.use_g1 and self.use_g2 and not self.use_g3:
        #     out = torch.cat((g0_tag1_out, g0_tag2_out, g1_tag1_out, g1_tag2_out, g2_o), dim=1)
        # elif self.use_g0 and self.use_g1 and not self.use_g2 and not self.use_g3:
        #     out = torch.cat((g0_tag1_out, g0_tag2_out, g1_tag1_out, g1_tag2_out), dim=1)
        # elif self.use_g0 and not self.use_g1 and not self.use_g2 and not self.use_g3:
        #     out = torch.cat((g0_tag1_out, g0_tag2_out), dim=1)
        # elif not self.use_g0 and self.use_g1 and not self.use_g2 and not self.use_g3:
        #     out = torch.cat((g1_tag1_out, g1_tag2_out), dim=1)
        # else:
        #     print("create model unsuccessfully.")
        #     sys.exit(1)
        # out = self.out_layer(out)

        gt_loss = self.gt_crit(out.contiguous().view(-1, self.nclass), gt_target.contiguous().view(-1))
        return gt_loss, out

    def predict(self, g0, g1, g2, g3):
        g0_p = g0.permute(2, 0, 1, 3)   # [11,N,40,5]
        g0_s = g0.permute(0, 3, 1, 2) # [N, 5, 40, 11]
        g1_p = g1.permute(2, 0, 1, 3)   # [11,N,40,5]
        g1_s = g1.permute(0, 3, 1, 2) # [N, 5, 40, 11]

        g0_tag1 = g0_p[:, :, :20, 0]  # [5,N,20]
        g0_tag2 = g0_p[:, :, 20:, 0]  # [5,N,20]

        g0_tag1 = calculate_percentage(g0_tag1) #[11,N,5]
        g0_tag2 = calculate_percentage(g0_tag2) #[11,N,5]


        g1_tag1 = g1_p[:, :, :20, 0]  # [5,N,20]
        g1_tag2 = g1_p[:, :, 20:, 0]  # [5,N,20]

        g1_tag1 = calculate_percentage(g1_tag1) #[11,N,5]
        g1_tag2 = calculate_percentage(g1_tag2) #[11,N,5]

        g0_g1_tag1_tag2_cat = torch.cat((g0_tag1,g0_tag2,g1_tag1,g1_tag2),2)   # [L, N, 20]

        g0_g1_p_out = self.haplotype_percentage(g0_g1_tag1_tag2_cat)[5, :, :]  # [N, 256]

        g0_g1_s_out = self.haplotype_base(torch.cat((g0_s, g1_s),1))[5]   # [N, 256]

        out = self.out_layer(torch.cat((g0_g1_p_out, g0_g1_s_out), 1))

        out = torch.softmax(out, 1)
        return out
