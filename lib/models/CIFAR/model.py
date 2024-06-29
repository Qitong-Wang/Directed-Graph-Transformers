from ..digt_molgraph import DIGT_MOL

class DIGT_CIFAR(DIGT_MOL):
    def __init__(self, **kwargs):
        super().__init__(output_dim=10, input_node_dim = 5, input_edge_dim  = 1,**kwargs)

