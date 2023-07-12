import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self, enc_in, dec_in, c_out, out_len,
                 d_model=512, n_heads=8, e_layers=3, d_layers=2, d_ff=512,
                 dropout=0.1, embed='fixed', freq='h', activation='gelu',
                 output_attention=False, distil=True, mix=True, args=None):
        super(Model, self).__init__()
        self.P = args.seq_len
        self.pred_len = out_len
        self.m = args.feature
        self.output_attention = output_attention
        self.hidR = 256
        self.hidC = 256
        self.hidS = 256
        self.Ck = 7
        self.skip = 10
        self.pt = int((self.P - self.Ck)/self.skip)
        self.hw = 20
        self.conv1 = nn.Conv2d(1, self.hidC, kernel_size = (self.Ck, self.m))
        self.GRU1 = nn.GRU(self.hidC, self.hidR)
        self.dropout = nn.Dropout(p=dropout)
        if (self.skip > 0):
            self.GRUskip = nn.GRU(self.hidC, self.hidS)
            self.linear1 = nn.Linear(self.hidR + self.skip * self.hidS, self.m)
        else:
            self.linear1 = nn.Linear(self.hidR, self.m)
        if (self.hw > 0):
            self.highway = nn.Linear(self.hw, 1)
        self.output = F.relu
 
    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
        attns = [None]
        batch_size = x_enc.size(0)
        
        #CNN
        c = x_enc.view(-1, 1, self.P, self.m)
        c = F.relu(self.conv1(c))
        c = self.dropout(c)
        c = torch.squeeze(c, 3)
        
        # RNN 
        r = c.permute(2, 0, 1).contiguous()
        _, r = self.GRU1(r)
        r = self.dropout(torch.squeeze(r,0))

        
        #skip-rnn
        
        if (self.skip > 0):
            s = c[:,:, int(-self.pt * self.skip):].contiguous()
            s = s.view(batch_size, self.hidC, self.pt, self.skip)
            s = s.permute(2,0,3,1).contiguous()
            s = s.view(self.pt, batch_size * self.skip, self.hidC)
            _, s = self.GRUskip(s)
            s = s.view(batch_size, self.skip * self.hidS)
            s = self.dropout(s)
            r = torch.cat((r,s),1)
        
        res = self.linear1(r)
        
        #highway
        if (self.hw > 0):
            z = x_enc[:, -self.hw:, :]
            z = z.permute(0,2,1).contiguous().view(-1, self.hw)
            z = self.highway(z)
            z = z.view(-1,self.m)
            res = res + z
        res = res.unsqueeze(1)
        if (self.output):
            res = self.output(res)

        if self.output_attention:
            return res[:, -self.pred_len:, :], attns
        else:
            return res[:, -self.pred_len:, :]  # [B, L, D]
    
        
        
        
