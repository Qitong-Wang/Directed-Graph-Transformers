import torch
from torch import nn
from typing import Tuple
import einops


class Combine_Attention_Layer(nn.Module):
    def __init__(self,
                 node_width= 48,
                 num_heads = 8,
                 dot_dim = 6
                 ):
        super().__init__()
        self.node_width = node_width
        self.num_heads = num_heads
        self.dot_dim = dot_dim
        self.lin_QKVs = nn.Linear(self.node_width, self.node_width * 3)
        self.lin_QKVt = nn.Linear(self.node_width, self.node_width * 3)


    def forward(self, training, Dst, s, t,E,G,mask_s):
        # S:
        E_T = einops.rearrange(E, 'b l m h -> b m l h')
        G_T = einops.rearrange(G, 'b l m h -> b m l h')

        if training:
            mask_s = mask_s + torch.empty_like(E).bernoulli_(0.1) * -1e9

        QKVs =  self.lin_QKVs(s)
        QKVt = self.lin_QKVt(t)
        shp = QKVs.shape
        Qs, Ks, Vs = QKVs.view(shp[0], shp[1], -1, self.num_heads).split(self.dot_dim, dim=2)
        Qt, Kt, Vt = QKVt.view(shp[0], shp[1], -1, self.num_heads).split(self.dot_dim, dim=2)
        As_hat = torch.einsum('bldh,bmdh->blmh', Qs, Kt) * (self.dot_dim ** -0.5)
        At_hat = torch.einsum('bldh,bmdh->blmh', Qt, Ks) * (self.dot_dim ** -0.5)


        gates_s = torch.sigmoid(G+mask_s )
        gates_t = torch.sigmoid(G_T+mask_s)
        As_hat = As_hat+E
        At_hat = At_hat+E_T

        As_tilde = torch.einsum('blmh,blm->blmh', As_hat, Dst)
        Dts = einops.rearrange(Dst, 'b l m-> b m l')
        At_tilde = torch.einsum('blmh,blm->blmh', At_hat, Dts)
        sum_att = torch.stack((As_tilde, At_tilde), dim=4)

        soft_att = torch.softmax(sum_att, dim=4)
        Y =  torch.einsum('blmh,bmkh->blkh', soft_att[:,:,:,:,0]*gates_s,Vt ) + torch.einsum('blmh,bmkh->blkh', soft_att[:,:,:,:,1]*gates_t,Vs )
        Y = Y.reshape(shp[0], shp[1], self.num_heads * self.dot_dim)

        return Y, As_hat


