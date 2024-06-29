import torch
from torch import nn
import torch.nn.functional as F
from .digt import DIGT
from .mutual_att import Combine_Attention_Layer
from .digt_layers import VirtualNodes

class DIGT_MOL(DIGT):
    def __init__(self,
                 upto_hop          = 16,
                 mlp_ratios        = [0.5, 0.25],
                 num_virtual_nodes = 0,
                 pe_encodings     = 0,
                 output_dim        = 1,
                 input_node_dim    = 1,
                 input_edge_dim    = 1,
                 **kwargs):
        super().__init__(node_ended=True, **kwargs)
        self.experiment_id = kwargs['experiment_id']
        self.task = kwargs['task']
        
        self.upto_hop          = upto_hop
        self.mlp_ratios        = mlp_ratios
        self.num_virtual_nodes = num_virtual_nodes
        self.pe_encodings     = pe_encodings
        self.output_dim        = output_dim
        self. input_edge_dim = input_edge_dim


        if self.pe_encodings:
            self.s_pe_embed = nn.Linear(self.pe_encodings, self.node_width)
            self.t_pe_embed = nn.Linear(self.pe_encodings, self.node_width)
            self.s_pe_embed2 = nn.Linear(self.pe_encodings, self.node_width)
            self.t_pe_embed2 = nn.Linear(self.pe_encodings, self.node_width)

        if input_node_dim > 0:
            self.node_linear =  nn.Linear(input_node_dim,self.node_width)
        if input_edge_dim > 0:
            self.edge_linear = nn.Linear(input_edge_dim, self.edge_width)

        self.dist_embed = nn.Embedding(self.upto_hop+2, self.edge_width)
        if self.num_virtual_nodes > 0:
            self.vn_layer = VirtualNodes(self.node_width, self.edge_width,
                                         self.num_virtual_nodes)

        self.final_ln_s = nn.LayerNorm(self.node_width)
        self.final_ln_t = nn.LayerNorm(self.node_width)

        mlp_dims = [self.node_width * max(self.num_virtual_nodes, 1)*2]\
                    +[round(self.node_width*r) for r in self.mlp_ratios]\
                        +[self.output_dim]
        self.mlp_layers = nn.ModuleList([nn.Linear(mlp_dims[i],mlp_dims[i+1])
                                         for i in range(len(mlp_dims)-1)])
        self.mlp_fn = getattr(F, self.activation)

    
    def input_block(self, inputs):
        g = super().input_block(inputs)
        s = self.s_pe_embed(g.u_pe_encodings) + self.s_pe_embed2(g.v_pe_encodings)
        t = self.t_pe_embed(g.u_pe_encodings)  + self.t_pe_embed2(g.v_pe_encodings)
        if 'node_features' in g.keys():
            node_features = g.node_features
            h = self.node_linear(node_features)
            s = s + h
            t = t + h

        e = self.dist_embed(g.edge_encoding)
        if self.input_edge_dim > 0:
            feature_matrix = g.feature_matrix
            e = e + self.edge_linear(feature_matrix)
        if not 'node_mask' in g.keys():
            print("no node mask")
        nodem = g.node_mask.float()
        g.mask_s = (nodem[:,:,None,None] * nodem[:,None,:,None] - 1)*1e9
        g.s,g.t, g.e = s,t, e
        if self.num_virtual_nodes > 0:
            g = self.vn_layer(g)
        return g

    def final_embedding(self, g):

        if self.task == "graph_classification":
            s,t = g.s,g.t

            s = self.final_ln_s(s)
            t = self.final_ln_t(t)

            if self.num_virtual_nodes > 0:
                s = s[:, :self.num_virtual_nodes].reshape(s.shape[0], -1)
                t = t[:, :self.num_virtual_nodes].reshape(t.shape[0], -1)
            else:
                nodem = g.node_mask.float().unsqueeze(dim=-1)
                s = (s * nodem).sum(dim=1) / (nodem.sum(dim=1) + 1e-9)
                t = (t * nodem).sum(dim=1) / (nodem.sum(dim=1) + 1e-9)


            g.s = s
            g.t = t


            return g
        elif self.task == "node_classification":
            s,t = g.s,g.t
            s = self.final_ln_s(s)
            t = self.final_ln_t(t)
          
            g.s = s
            g.t = t
            return g
        else:
            raise ValueError('Unsupported task.')

    def output_block(self, g):
        s,t = g.s,g.t
        if self.task == "graph_classification":
            s = torch.cat((s,t),dim=1)
        elif self.task == "node_classification":
            s = torch.cat((s,t),dim=2)
  
        s = self.mlp_layers[0](s)
        for layer in self.mlp_layers[1:]:
            s = layer(self.mlp_fn(s))

        return s


