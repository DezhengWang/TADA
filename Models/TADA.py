import torch
import torch.nn as nn
from LayerTADA.encoder import Encoder, EncoderLayer
from LayerTADA.decoder import Decoder, DecoderLayer
from LayerTADA.attention import TempAttention, AttentionLayer
from utils.embed import DataEmbedding


class Model(nn.Module):
    def __init__(self, enc_in, dec_in, c_out, out_len,
                 d_model=512, n_heads=8, e_layers=3, d_layers=2, d_ff=512,
                 dropout=0.1, embed='fixed', freq='h', activation='gelu',
                 output_attention=False, distil=True, mix=True, args=None):
        super(Model, self).__init__()
        self.pred_len = out_len
        self.output_attention = output_attention

        self.args = args

        # Encoding
        self.enc_embedding = DataEmbedding(enc_in, d_model, embed, freq, dropout)
        self.dec_embedding = DataEmbedding(dec_in, d_model, embed, freq, dropout)
        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(TempAttention(False, attention_dropout=dropout, output_attention=output_attention),
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
                    AttentionLayer(TempAttention(True, attention_dropout=dropout, output_attention=output_attention),
                                   d_model, n_heads, mix=mix),
                    AttentionLayer(TempAttention(False, attention_dropout=dropout, output_attention=output_attention),
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