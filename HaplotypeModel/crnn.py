import torch.nn as nn


class BidirectionalLSTM(nn.Module):

    def __init__(self, nIn, nHidden, nOut):
        super(BidirectionalLSTM, self).__init__()

        self.rnn = nn.LSTM(nIn, nHidden, bidirectional=True)
        self.embedding = nn.Linear(nHidden * 2, nOut)

    def forward(self, input):
        recurrent, _ = self.rnn(input)
        T, b, h = recurrent.size()
        t_rec = recurrent.view(T * b, h)

        output = self.embedding(t_rec)  # [T * b, nOut]
        output = output.view(T, b, -1)

        return output


class CRNN(nn.Module):

    def __init__(self, nc, nclass, nh, leakyRelu=False):
        super(CRNN, self).__init__()
        # assert imgH % 16 == 0, 'imgH has to be a multiple of 16'
        #
        # ks = [3, 3, 3, 3, 3, 3, 2]
        # ps = [1, 1, 1, 1, 1, 1, 0]
        # ss = [1, 1, 1, 1, 1, 1, 1]
        # nm = [64, 128, 256, 256, 512, 512, 512]

        cnn = nn.Sequential()

        # def convRelu(i, batchNormalization=False):
        #     nIn = nc if i == 0 else nm[i - 1]
        #     nOut = nm[i]
        #     cnn.add_module('conv{0}'.format(i),
        #                    nn.Conv2d(nIn, nOut, ks[i], ss[i], ps[i]))
        #     if batchNormalization:
        #         cnn.add_module('batchnorm{0}'.format(i), nn.BatchNorm2d(nOut))
        #     if leakyRelu:
        #         cnn.add_module('relu{0}'.format(i),
        #                        nn.LeakyReLU(0.2, inplace=True))
        #     else:
        #         cnn.add_module('relu{0}'.format(i), nn.ReLU(True))

        def convRelu2(i, nIn, nOut, ks, ss, ps, batchNormalization=False):
            cnn.add_module('conv{0}'.format(i), nn.Conv2d(nIn, nOut, ks, ss, ps))
            if batchNormalization:
                cnn.add_module('batchnorm{0}'.format(i), nn.BatchNorm2d(nOut))
            if leakyRelu:
                cnn.add_module('relu{0}'.format(i), nn.LeakyReLU(0.2, inplace=True))
            else:
                cnn.add_module('relu{0}'.format(i), nn.ReLU(True))

        convRelu2(0, nIn=nc, nOut=32, ks=3, ss=1, ps=1, batchNormalization=False)  # 32x40x5
        cnn.add_module('pooling{0}'.format(0),
                       nn.MaxPool2d(kernel_size=(2, 3), stride=(2, 1), padding=(0, 1)))  # 32x20x5
        convRelu2(1, nIn=32, nOut=64, ks=3, ss=1, ps=1, batchNormalization=False)  # 64x20x5
        cnn.add_module('pooling{0}'.format(1),
                       nn.MaxPool2d(kernel_size=(2, 3), stride=(2, 1), padding=(0, 1)))  # 64x10x5
        convRelu2(2, nIn=64, nOut=128, ks=3, ss=1, ps=1, batchNormalization=True)  # 128x10x5
        convRelu2(3, nIn=128, nOut=128, ks=3, ss=1, ps=1, batchNormalization=False)  # 128x10x5
        cnn.add_module('pooling{0}'.format(2),
                       nn.MaxPool2d(kernel_size=(3, 3), stride=(3, 1), padding=(0, 1)))  # 128x3x5
        convRelu2(4, nIn=128, nOut=256, ks=3, ss=1, ps=1, batchNormalization=True)  # 256x3x5
        convRelu2(5, nIn=256, nOut=256, ks=3, ss=1, ps=1, batchNormalization=False)  # 256x3x5
        cnn.add_module('pooling{0}'.format(3),
                       nn.MaxPool2d(kernel_size=(2, 3), stride=(2, 1), padding=(0, 1)))  # 256x1x5

        self.cnn = cnn
        self.rnn = nn.Sequential(
            BidirectionalLSTM(256, nh, nh),
            BidirectionalLSTM(nh, nh, nclass))

    def forward(self, input):
        # conv features
        conv = self.cnn(input)
        b, c, h, w = conv.size()
        assert h == 1, "the height of conv must be 1"
        conv = conv.squeeze(2)
        conv = conv.permute(2, 0, 1)  # [w, b, c]

        # rnn features
        output = self.rnn(conv)

        return output


class ResBlock(nn.Module):
    def __init__(self, i, nIn, nOut, ks, ss, ps):
        super(ResBlock, self).__init__()
        cnn = nn.Sequential()
        cnn.add_module('conv{0}_base_conv1'.format(i),
                       nn.Conv2d(in_channels=nIn, out_channels=nOut, kernel_size=ks, stride=ss, padding=ps))
        cnn.add_module('conv{0}_base_bn1'.format(i), nn.BatchNorm2d(nOut))
        cnn.add_module('conv{0}_base_relu1'.format(i), nn.ReLU(True))
        cnn.add_module('conv{0}_base_conv2'.format(i),
                       nn.Conv2d(in_channels=nOut, out_channels=nOut, kernel_size=ks, stride=ss, padding=ps))
        cnn.add_module('conv{0}_base_bn2'.format(i), nn.BatchNorm2d(nOut))
        self.base = cnn

        self.shortcut = nn.Sequential()
        self.shortcut.add_module('conv{0}_shortcut_conv1'.format(i),
                                 nn.Conv2d(in_channels=nIn, out_channels=nOut, kernel_size=1, stride=1, padding=0))
        self.act = nn.ReLU()

    def forward(self, x):
        res = x
        x = self.base(x)
        x = x + self.shortcut(res)
        x = self.act(x)
        return x


class ResCRNN(nn.Module):

    def __init__(self, nc, nclass, nh, leakyRelu=False):
        super(ResCRNN, self).__init__()
        # assert imgH % 16 == 0, 'imgH has to be a multiple of 16'
        #
        # ks = [3, 3, 3, 3, 3, 3, 2]
        # ps = [1, 1, 1, 1, 1, 1, 0]
        # ss = [1, 1, 1, 1, 1, 1, 1]
        # nm = [64, 128, 256, 256, 512, 512, 512]

        cnn = nn.Sequential()

        # def convRelu(i, batchNormalization=False):
        #     nIn = nc if i == 0 else nm[i - 1]
        #     nOut = nm[i]
        #     cnn.add_module('conv{0}'.format(i),
        #                    nn.Conv2d(nIn, nOut, ks[i], ss[i], ps[i]))
        #     if batchNormalization:
        #         cnn.add_module('batchnorm{0}'.format(i), nn.BatchNorm2d(nOut))
        #     if leakyRelu:
        #         cnn.add_module('relu{0}'.format(i),
        #                        nn.LeakyReLU(0.2, inplace=True))
        #     else:
        #         cnn.add_module('relu{0}'.format(i), nn.ReLU(True))

        # def convRelu2(i, nIn, nOut, ks, ss, ps, batchNormalization=False):
        #     cnn.add_module('conv{0}'.format(i), nn.Conv2d(nIn, nOut, ks, ss, ps))
        #     if batchNormalization:
        #         cnn.add_module('batchnorm{0}'.format(i), nn.BatchNorm2d(nOut))
        #     if leakyRelu:
        #         cnn.add_module('relu{0}'.format(i), nn.LeakyReLU(0.2, inplace=True))
        #     else:
        #         cnn.add_module('relu{0}'.format(i), nn.ReLU(True))

        # convRelu2(0, nIn=nc, nOut=32, ks=3, ss=1, ps=1, batchNormalization=False)  # 32x40x5
        cnn.add_module('conv{0}'.format(0), ResBlock(0, nIn=nc, nOut=32, ks=3, ss=1, ps=1))
        cnn.add_module('pooling{0}'.format(0),
                       nn.MaxPool2d(kernel_size=(2, 3), stride=(2, 1), padding=(0, 1)))  # 32x20x5
        cnn.add_module('conv{0}'.format(1), ResBlock(1, nIn=32, nOut=64, ks=3, ss=1, ps=1))
        # convRelu2(1, nIn=32, nOut=64, ks=3, ss=1, ps=1, batchNormalization=False)  # 64x20x5
        cnn.add_module('pooling{0}'.format(1),
                       nn.MaxPool2d(kernel_size=(2, 3), stride=(2, 1), padding=(0, 1)))  # 64x10x5
        cnn.add_module('conv{0}'.format(2), ResBlock(2, nIn=64, nOut=128, ks=3, ss=1, ps=1))
        cnn.add_module('conv{0}'.format(3), ResBlock(3, nIn=128, nOut=128, ks=3, ss=1, ps=1))
        # convRelu2(2, nIn=64, nOut=128, ks=3, ss=1, ps=1, batchNormalization=True)  # 128x10x5
        # convRelu2(3, nIn=128, nOut=128, ks=3, ss=1, ps=1, batchNormalization=False)  # 128x10x5
        cnn.add_module('pooling{0}'.format(2),
                       nn.MaxPool2d(kernel_size=(3, 3), stride=(3, 1), padding=(0, 1)))  # 128x3x5
        cnn.add_module('conv{0}'.format(4), ResBlock(4, nIn=128, nOut=256, ks=3, ss=1, ps=1))
        cnn.add_module('conv{0}'.format(5), ResBlock(5, nIn=256, nOut=256, ks=3, ss=1, ps=1))
        # convRelu2(4, nIn=128, nOut=256, ks=3, ss=1, ps=1, batchNormalization=True)  # 256x3x5
        # convRelu2(5, nIn=256, nOut=256, ks=3, ss=1, ps=1, batchNormalization=False)  # 256x3x5
        cnn.add_module('pooling{0}'.format(3),
                       nn.MaxPool2d(kernel_size=(2, 3), stride=(2, 1), padding=(0, 1)))  # 256x1x5

        self.cnn = cnn
        self.rnn = nn.Sequential(
            BidirectionalLSTM(256, nh, nh),
            BidirectionalLSTM(nh, nh, nclass))

    def forward(self, input):
        # conv features
        conv = self.cnn(input)
        b, c, h, w = conv.size()
        assert h == 1, "the height of conv must be 1"
        conv = conv.squeeze(2)
        conv = conv.permute(2, 0, 1)  # [w, b, c]

        # rnn features
        output = self.rnn(conv)

        return output
