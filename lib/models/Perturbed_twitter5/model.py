from ..digt_molgraph import DIGT_MOL

class DIGT_Perturbed_twitter5(DIGT_MOL):
    def __init__(self, **kwargs):
        super().__init__(output_dim=5, input_node_dim = 0, input_edge_dim  = 0,**kwargs)

