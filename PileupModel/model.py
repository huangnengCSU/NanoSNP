import torch.nn as nn
import torch
from optim import LabelSmoothingLoss

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


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
    def __init__(self, input_size, inner_size, gt_class, zy_class, id1_class, id2_class):
        super(ForwardLayer, self).__init__()
        self.dense = nn.Linear(input_size, inner_size, bias=True)
        self.tanh = nn.Tanh()
        self.genotype_layer = nn.Linear(inner_size, gt_class, bias=True)
        self.zygosity_layer = nn.Linear(inner_size, zy_class, bias=True)
        self.indel1_layer = nn.Linear(inner_size, id1_class, bias=True)
        self.indel2_layer = nn.Linear(inner_size, id2_class, bias=True)

    def forward(self, inputs):
        out = self.tanh(self.dense(inputs))  # [batch, length, hidden*2]
        out = out[:, 16, :]  # [batch, hidden*2]
        gt_outputs = self.genotype_layer(out)
        zy_outputs = self.zygosity_layer(out)
        id1_outputs = self.indel1_layer(out)
        id2_outputs = self.indel2_layer(out)
        return gt_outputs, zy_outputs, id1_outputs, id2_outputs


def build_forward(config):
    return ForwardLayer(config.enc.output_size,
                        config.joint.inner_size,
                        config.gt_num_class,
                        config.zy_num_class,
                        config.indel1_num_class,
                        config.indel2_num_class)


class LSTMNetwork(nn.Module):
    def __init__(self, config):
        super(LSTMNetwork, self).__init__()
        # define encoder
        self.config = config
        self.encoder = build_encoder(config)
        self.forward_layer = build_forward(config)
        self.gt_crit = LabelSmoothingLoss(config.gt_num_class, 0.1)
        self.zy_crit = LabelSmoothingLoss(config.zy_num_class, 0.1)
        self.indel1_crit = LabelSmoothingLoss(config.indel1_num_class, 0.1)
        self.indel2_crit = LabelSmoothingLoss(config.indel2_num_class, 0.1)

    def forward(self, inputs, gt_target, zy_target, id1_target, id2_target):
        # inputs: N x L x c
        enc_state, _ = self.encoder(inputs)  # [N, L, o]
        gt_logits, zy_logits, id1_logits, id2_logits = self.forward_layer(enc_state)
        gt_loss = self.gt_crit(gt_logits.contiguous().view(-1, self.config.gt_num_class),
                               gt_target.contiguous().view(-1))
        zy_loss = self.zy_crit(zy_logits.contiguous().view(-1, self.config.zy_num_class),
                               zy_target.contiguous().view(-1))
        id1_loss = self.indel1_crit(id1_logits.contiguous().view(-1, self.config.indel1_num_class),
                                    id1_target.contiguous().view(-1))
        id2_loss = self.indel2_crit(id2_logits.contiguous().view(-1, self.config.indel2_num_class),
                                    id2_target.contiguous().view(-1))
        # loss = gt_loss + zy_loss + id1_loss + id2_loss
        loss = gt_loss + zy_loss

        return loss, gt_logits, zy_logits, id1_logits, id2_logits

    def predict(self, inputs):
        enc_state, _ = self.encoder(inputs)  # [N, L, o]
        gt_logits, zy_logits, id1_logits, id2_logits = self.forward_layer(enc_state)
        gt_logits = torch.softmax(gt_logits, 1)
        zy_logits = torch.softmax(zy_logits, 1)
        return gt_logits, zy_logits
