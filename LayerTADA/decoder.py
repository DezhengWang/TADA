import torch
import torch.nn as nn
import torch.nn.functional as F

class DecoderLayer(nn.Module):
    def __init__(self, self_attention, cross_attention, d_model, d_ff=None,
                 dropout=0.1, activation="relu"):
        super(DecoderLayer, self).__init__()
        d_ff = d_ff or 4*d_model
        self.self_attention = self_attention
        self.cross_attention = cross_attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, data, cross_data, x_mask=None, cross_mask=None, ii=0):
        x = data[0]
        cross = cross_data[0]
        if ii == 0:
            x = x + self.dropout(self.cross_attention(
                data[1], cross_data[1], cross,
                attn_mask=cross_mask
            )[0])
            x = self.norm1(x)
            x = x + self.dropout(self.self_attention(
                data[1], data[1], x,
                attn_mask=x_mask
            )[0])
        else:
            x = x + self.dropout(self.self_attention(
                x, x, x,
                attn_mask=x_mask
            )[0])
            x = self.norm1(x)

            x = x + self.dropout(self.cross_attention(
                x, cross, cross,
                attn_mask=cross_mask
            )[0])

        y = x = self.norm2(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1,1))))
        y = self.dropout(self.conv2(y).transpose(-1,1))
        x = self.norm3(x + y)
        return torch.stack((x, data[1]))

class Decoder(nn.Module):
    def __init__(self, layers, norm_layer=None):
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList(layers)
        self.norm = norm_layer

    def forward(self, x, cross, x_mask=None, cross_mask=None):
        for ii, layer in enumerate(self.layers):
            x = layer(x, cross, x_mask=x_mask, cross_mask=cross_mask, ii=ii)

        x_info = x[0]
        if self.norm is not None:
            x_info = self.norm(x_info)

        return x_info