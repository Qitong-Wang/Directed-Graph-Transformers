import torch
from torch import nn
import torch.nn.functional as F
import einops
from .mutual_att import Combine_Attention_Layer


class Graph(dict):
    def __dir__(self):
        return super().__dir__() + list(self.keys())

    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError:
            raise AttributeError('No such attribute: ' + key)

    def __setattr__(self, key, value):
        self[key] = value

    def copy(self):
        return self.__class__(self)




class DIGT_Layer(nn.Module):
    def __init__(self,
                 node_width,
                 edge_width,
                 num_heads,
                 node_mha_dropout=0,
                 edge_mha_dropout=0,
                 node_ffn_dropout=0,
                 edge_ffn_dropout=0,
                 attn_dropout=0,
                 attn_maskout=0,
                 activation='elu',
                 clip_logits_value=[-5, 5],
                 node_ffn_multiplier=2.,
                 edge_ffn_multiplier=2.,
                 scale_degree=False,
                 node_update=True,
                 edge_update=True,
                 batch_norm=True,
                 layer_norm=False,
                 ):
        super().__init__()
        self.node_width = node_width
        self.edge_width = edge_width
        self.num_heads = num_heads
        self.node_mha_dropout = node_mha_dropout
        self.edge_mha_dropout = edge_mha_dropout
        self.node_ffn_dropout = node_ffn_dropout
        self.edge_ffn_dropout = edge_ffn_dropout
        self.attn_dropout = attn_dropout
        self.attn_maskout = attn_maskout
        self.activation = activation
        self.clip_logits_value = clip_logits_value
        self.node_ffn_multiplier = node_ffn_multiplier
        self.edge_ffn_multiplier = edge_ffn_multiplier
        self.scale_degree = scale_degree
        self.node_update = node_update
        self.edge_update = edge_update
        self.batch_norm = batch_norm
        self.layer_norm = layer_norm

        assert not (self.node_width % self.num_heads)
        self.dot_dim = self.node_width // self.num_heads


        self.lin_E = nn.Linear(self.edge_width, self.num_heads)
        self.lin_G = nn.Linear(self.edge_width, self.num_heads)

        self.ffn_fn = getattr(F, self.activation)

        self.lin_O_s = nn.Linear(self.node_width, self.node_width)
        self.lin_O_t = nn.Linear(self.node_width, self.node_width)

        node_inner_dim = round(self.node_width * self.node_ffn_multiplier)

        self.lin_W_s_1 = nn.Linear(self.node_width, node_inner_dim)
        self.lin_W_t_1 = nn.Linear(self.node_width, node_inner_dim)
        self.lin_W_s_2 = nn.Linear(node_inner_dim, self.node_width)
        self.lin_W_t_2 = nn.Linear(node_inner_dim, self.node_width)

        self.lin_O_e = nn.Linear(self.num_heads, self.edge_width)

        edge_inner_dim = round(self.edge_width * self.edge_ffn_multiplier)

        self.lin_W_e_1 = nn.Linear(self.edge_width, edge_inner_dim)
        self.lin_W_e_2 = nn.Linear(edge_inner_dim, self.edge_width)

        self.digt_layer = Combine_Attention_Layer(self.node_width, self.num_heads, self.dot_dim)

        if batch_norm:
            self.ffn_ln_s = nn.BatchNorm1d(self.node_width)
            self.ffn_ln_t = nn.BatchNorm1d(self.node_width)
            self.mha_ln_V = nn.BatchNorm1d(self.node_width)
            self.mha_ln_H = nn.BatchNorm1d(self.edge_width)
            self.ffn_ln_e = nn.BatchNorm1d(self.edge_width)
        elif layer_norm:
            self.ffn_ln_s = nn.LayerNorm(self.node_width)
            self.ffn_ln_t = nn.LayerNorm(self.node_width)
            self.mha_ln_V = nn.LayerNorm(self.node_width)
            self.mha_ln_H = nn.LayerNorm(self.edge_width)
            self.ffn_ln_e = nn.LayerNorm(self.edge_width)
        else:
            self.ffn_ln_s = None
            self.ffn_ln_t = None
            self.mha_ln_V = None
            self.mha_ln_H = None
            self.ffn_ln_e = None

    def normalization_layer(self, norm_layer, input_feature):
        if norm_layer is None:
            return input_feature
        if  self.batch_norm:
            if len(input_feature.shape) == 3:
                b, c, h = input_feature.shape
                input_feature = einops.rearrange(input_feature, 'b c h -> (b c) h')
                input_feature = norm_layer(input_feature)
                input_feature = einops.rearrange(input_feature, '(b c) h -> b c h', b=b, c=c, h=h)
                return input_feature
            elif len(input_feature.shape) == 4:
                b, c, d, h = input_feature.shape
                input_feature = einops.rearrange(input_feature, 'b c d h -> (b c d ) h')
                input_feature = norm_layer(input_feature)
                input_feature = einops.rearrange(input_feature, ' (b c d ) h -> b c d  h ', b=b, c=c, d=d, h=h)
                return input_feature
        elif self.layer_norm:
            input_feature = norm_layer(input_feature)
            return input_feature


    def forward(self, g):
        s, t, e = g.s, g.t, g.e
        mask_s = g.mask_s

        s_r1 = s
        t_r1 = t
        e_r1 = e


        E = self.lin_E(e)
        G = self.lin_G(e)

        Dst = g.filter_matrix

        # Attention
        V_att, H_hat = self.digt_layer(self.training, Dst, s, t, E, G, mask_s)
        V_att = self.normalization_layer(self.mha_ln_V, V_att)
        s = self.lin_O_s(V_att)
        t = self.lin_O_t(V_att)

        s = F.elu(s)
        t = F.elu(t)

        s.add_(s_r1)
        t.add_(t_r1)

        s_r2 = s
        t_r2 = t
        s = self.normalization_layer(self.ffn_ln_s, s)
        t = self.normalization_layer(self.ffn_ln_t, t)

        s = self.lin_W_s_2(self.ffn_fn(self.lin_W_s_1(s)))
        t = self.lin_W_t_2(self.ffn_fn(self.lin_W_t_1(t)))
        s = F.elu(s)
        t = F.elu(t)

        s.add_(s_r2)
        t.add_(t_r2)

        e = self.lin_O_e(H_hat)
        e = self.normalization_layer(self.mha_ln_H, e)
        e.add_(e_r1)

        e_r2 = e
        e = self.normalization_layer(self.ffn_ln_e, e)
        e = self.lin_W_e_2(self.ffn_fn(self.lin_W_e_1(e)))
        e.add_(e_r2)

        g = g.copy()
        g.s, g.t, g.e = s, t, e
        return g

    def __repr__(self):
        rep = super().__repr__()
        rep = (rep + ' ('
               + f'num_heads: {self.num_heads},'
               + f'activation: {self.activation},'
               + f'attn_maskout: {self.attn_maskout},'
               + f'attn_dropout: {self.attn_dropout}'
               + ')')
        return rep


class VirtualNodes(nn.Module):
    def __init__(self, node_width, edge_width, num_virtual_nodes=1):
        super().__init__()
        self.node_width = node_width
        self.edge_width = edge_width
        self.num_virtual_nodes = num_virtual_nodes

        self.vn_node_embeddings = nn.Parameter(torch.empty(num_virtual_nodes,
                                                           self.node_width))
        self.vn_edge_embeddings = nn.Parameter(torch.empty(num_virtual_nodes,
                                                           self.edge_width))
        nn.init.normal_(self.vn_node_embeddings)
        nn.init.normal_(self.vn_edge_embeddings)

    def forward(self, g):
        s, t, e = g.s, g.t, g.e
        mask_s = g.mask_s

        node_emb_s = self.vn_node_embeddings.unsqueeze(0).expand(s.shape[0], -1, -1)
        node_emb_t = self.vn_node_embeddings.unsqueeze(0).expand(s.shape[0], -1, -1)
        s = torch.cat([node_emb_s, s], dim=1)
        t = torch.cat([node_emb_t, t], dim=1)

        e_shape = e.shape
        edge_emb_row = self.vn_edge_embeddings.unsqueeze(1)
        edge_emb_col = self.vn_edge_embeddings.unsqueeze(0)
        edge_emb_box = 0.5 * (edge_emb_row + edge_emb_col)

        edge_emb_row = edge_emb_row.unsqueeze(0).expand(e_shape[0], -1, e_shape[2], -1)
        edge_emb_col = edge_emb_col.unsqueeze(0).expand(e_shape[0], e_shape[1], -1, -1)
        edge_emb_box = edge_emb_box.unsqueeze(0).expand(e_shape[0], -1, -1, -1)

        e = torch.cat([edge_emb_row, e], dim=1)
        e_col_box = torch.cat([edge_emb_box, edge_emb_col], dim=1)
        e = torch.cat([e_col_box, e], dim=2)

        dist_col = g.node_mask.unsqueeze(2).repeat(1, 1, self.num_virtual_nodes)
        dist_row = einops.rearrange(dist_col, ' b r n -> b n r ')

        filter_matrix = g.filter_matrix
        dist_box = torch.ones(filter_matrix.shape[0], self.num_virtual_nodes, self.num_virtual_nodes).to(
            filter_matrix.device)
        filter_matrix = torch.cat([dist_row, filter_matrix], dim=1)
        filter_col_box = torch.cat([dist_box, dist_col], dim=1)
        filter_matrix = torch.cat([filter_col_box, filter_matrix], dim=2)

        g = g.copy()
        g.s, g.t, g.e = s, t, e
        g.filter_matrix = filter_matrix

        g.num_vns = self.num_virtual_nodes

        if mask_s is not None:
            g.mask_s = F.pad(mask_s, (0, 0, self.num_virtual_nodes, 0, self.num_virtual_nodes, 0),
                             mode='constant', value=0)
        return g