from ..digt_molgraph import DIGT_MOL

class DIGT_Flow2(DIGT_MOL):
    def __init__(self, **kwargs):
        super().__init__(output_dim=2, input_node_dim = 0, input_edge_dim  = 0,**kwargs)

