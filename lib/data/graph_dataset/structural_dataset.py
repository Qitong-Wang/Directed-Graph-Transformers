import numpy as np
import numba as nb
import pickle
from .graph_dataset import GraphDataset



@nb.njit
def floyd_warshall(A,num_of_hops):
    n = A.shape[0]
    D = np.zeros((n, n), dtype=np.int16)
    filter = np.zeros((n, n), dtype=np.int16)
    for i in range(n):
        for j in range(n):
            if i == j:
                pass
            elif A[i, j] == 0:
                D[i, j] = 510
            else:
                D[i, j] = 1


    for k in range(n):
        for i in range(n):
            for j in range(n):
                old_dist = D[i, j]
                new_dist = D[i, k] + D[k, j]
                if new_dist < old_dist:
                    D[i, j] = new_dist
    D = np.clip(D, a_min=0, a_max=num_of_hops+1)
    filter = np.where(D==num_of_hops+1,0,D)
    filter = np.clip(filter,a_min=0,a_max=1)


    return D, filter


@nb.njit
def preprocess_data_superpixel(num_nodes, edges, edge_feats,num_of_hops):
    A = np.zeros((num_nodes, num_nodes), dtype=np.int16)
    E = np.zeros((num_nodes, num_nodes, edge_feats.shape[-1]), dtype=np.float32)
    for k in range(edges.shape[0]):
        i, j = edges[k, 0], edges[k, 1]
        A[i, j] = 1
        E[i, j] = edge_feats[k]

    D, filter = floyd_warshall(A,num_of_hops)
    return  D, filter, E, A


def preprocess_data_others( A,num_of_hops):
    D, filter= floyd_warshall(A,num_of_hops)
    return  D, filter

class StructuralDataset(GraphDataset):
    def __init__(self,
                 distance_matrix_key='distance_matrix',
                 feature_matrix_key='feature_matrix',
                 **kwargs):
        super().__init__(**kwargs)
        self.distance_matrix_key = distance_matrix_key
        self.feature_matrix_key = feature_matrix_key

    def __getitem__(self, index):
        upto_hop = self.upto_hop
        if "MALNET" in self.dataset_name:
            if self.split == "training":
                file = open("./temp_malnet/train_{}.pkl".format(str(index)), 'rb')
            elif self.split == "validation":
                file = open("./temp_malnet/val_{}.pkl".format(str(index)), 'rb')
            else:
                file = open("./temp_malnet/test_{}.pkl".format(str(index)), 'rb')
            item = pickle.load(file)
            file.close()
            num_nodes = int(item[self.num_nodes_key])
            dist_mat = item[self.distance_matrix_key]
            item.pop(self.edges_key)
            D = np.clip(dist_mat, a_min=0, a_max=upto_hop + 1)
            filter = np.where(D == upto_hop + 1, 0, D)
            filter = np.clip(filter, a_min=0, a_max=1)
            item["edge_encoding"] = D.astype(np.long)
            item["filter_matrix"] = filter
            item["node_mask"] = np.ones((num_nodes,), dtype=np.uint8)
            return item

        item = super().__getitem__(index)
        edges = item.pop(self.edges_key)
        num_nodes = int(item[self.num_nodes_key])
       
        if 'Flow' in self.dataset_name or "twitter" in self.dataset_name:
            A = item.pop('A')
            item['adj_matrix'] = A.toarray()
            dist_mat,filter = preprocess_data_others( item['adj_matrix'],upto_hop)
            item[self.distance_matrix_key] = dist_mat


        elif self.dataset_name=="MNIST" or self.dataset_name=="CIFAR":

            node_feats = item.pop(self.node_features_key)
            edge_feats = item.pop(self.edge_features_key)
            dist_mat,filter, edge_feats_mat, adj_mat = preprocess_data_superpixel(num_nodes, edges, edge_feats,upto_hop)
            item[self.node_features_key] = node_feats
            item[self.distance_matrix_key] = dist_mat
            item[self.feature_matrix_key] = edge_feats_mat
            item['adj_matrix'] = adj_mat


        else:
            raise ValueError('Please implement customize method to get feature matrix.')

        # concat edge encoding
        item["edge_encoding"] = dist_mat.astype(np.long)
        item["filter_matrix"] = filter
        return item