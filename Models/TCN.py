import torch
import torch.nn as nn
from torch.nn.utils import weight_norm

# 这个函数是用来修剪卷积之后的数据的尺寸，让其与输入数据尺寸相同。
class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()

class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2, is_last=False):
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)
        self.is_last = is_last
        if not self.is_last:
            self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                     self.conv2, self.chomp2, self.relu2, self.dropout2)
        else:
            self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                     self.conv2, self.chomp2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        if not self.is_last:
            self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        if not self.is_last:
            output = self.relu(out + res)
        else:
            output = out + res
        return output

class Model(nn.Module):
    def __init__(self, enc_in, dec_in, c_out, out_len,
                 d_model=512, n_heads=8, e_layers=3, d_layers=2, d_ff=512,
                 dropout=0.1, embed='fixed', freq='h', activation='gelu',
                 output_attention=False, distil=True, mix=True, args=None):
        super(Model, self).__init__()
        self.pred_len = out_len
        self.output_attention = output_attention
        num_inputs = args.seq_len
        num_channels = [2,3,3,2]
        kernel_size = 2
        dropout = 0.2
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels,
                                     out_channels,
                                     kernel_size,
                                     stride=1,
                                     dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size,
                                     dropout=dropout,
                                     is_last=True if i == num_levels-1 else False)]
        self.network = nn.Sequential(*layers)
        # self.fc = nn.Sequential(
        #     nn.Linear(3*20, 19),
        # )


    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
        attns = [None]
        res = self.network(x_enc)

        if self.output_attention:
            return res[:, -self.pred_len:, :], attns
        else:
            return res[:, -self.pred_len:, :]