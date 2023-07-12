import torch
import torch.nn as nn
from LayersTransformer.encoder import Encoder, EncoderLayer, ConvLayer
from LayersTransformer.decoder import Decoder, DecoderLayer
from LayersTransformer.attention import TempAttention, AttentionLayer
from utils.embed import DataEmbeddingStand


class Model(nn.Module):
    def __init__(self, enc_in, dec_in, c_out, out_len,
                 d_model=512, n_heads=8, e_layers=3, d_layers=2, d_ff=512,
                 dropout=0.0, embed='fixed', freq='h', activation='gelu',
                 output_attention=False, mix=True, args=None):
        super(Model, self).__init__()
        self.args = args
        self.pred_len = out_len
        self.output_attention = output_attention

        # Encoding
        self.enc_embedding = DataEmbeddingStand(enc_in, d_model, embed, freq, dropout, args)
        self.dec_embedding = DataEmbeddingStand(dec_in, d_model, embed, freq, dropout, args)
        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(TempAttention(False, dropout=dropout, atts=output_attention),
                                   d_model, n_heads, mix=False),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation
                ) for l in range(e_layers)
            ],
            None,
            norm_layer=torch.nn.LayerNorm(d_model)
        )
        # Decoder
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AttentionLayer(TempAttention(True, dropout=dropout, atts=output_attention),
                                   d_model, n_heads, mix=mix),
                    AttentionLayer(TempAttention(False, dropout=dropout, atts=output_attention),
                                   d_model, n_heads, mix=False),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation,
                )
                for l in range(d_layers)
            ],
            norm_layer=torch.nn.LayerNorm(d_model)
        )
        self.projection = nn.Linear(d_model, c_out, bias=True)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):

        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=enc_self_mask)

        dec_out = self.dec_embedding(x_dec, x_mark_dec)
        dec_out = self.decoder(dec_out, enc_out, x_mask=dec_self_mask, cross_mask=dec_enc_mask)
        dec_out = self.projection(dec_out)

        if self.output_attention:
            return dec_out[:, -self.pred_len:, :], attns
        else:
            return dec_out[:, -self.pred_len:, :]  # [B, L, D]