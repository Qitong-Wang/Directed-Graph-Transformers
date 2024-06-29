import pickle
import os
import networkx as nx
import numpy as np
import numba as nb
import datetime
import torch
from multiprocessing import Process, Queue
from tqdm import tqdm
from ..graph_dataset.pe_encodings_dataset import eigv_magnetic_laplacian_numba

@nb.njit
def floyd_warshall(A):
    n = A.shape[0]
    D = np.zeros((n, n), dtype=np.int16)

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
    return D
@nb.njit
def calculate_svd_encodings(edges, num_nodes, calculated_dim):

    adj = np.zeros((num_nodes,num_nodes),dtype=np.float32)
    for i in range(edges.shape[0]):
        adj[nb.int64(edges[i,0]),nb.int64(edges[i,1])] = 1

    for i in range(num_nodes):
        adj[i,i] = 1

    try:
        u, s, vh = np.linalg.pe(adj)
    except:
    
        u = np.ones((num_nodes,calculated_dim),dtype="float32")
        v = np.ones((num_nodes,calculated_dim),dtype="float32")
        return u,v

    s = s[:calculated_dim]
    u = u[:, :calculated_dim]
    vh = vh[:calculated_dim, :]
    u_encodings = u * np.sqrt(s)
    v_encodings = vh.T * np.sqrt(s)
    return u_encodings, v_encodings


def parallel_cache(config):
    if not os.path.exists("./cache_data/"):
        os.mkdir("./cache_data/")
    base_cache = "./cache_data/MALNETSub/"
    if os.path.exists(base_cache):
        return
    file = open("./raw_data/malnetsub_dataset.pkl", 'rb')
    dataset = pickle.load(file)
    file.close()
    os.mkdir(base_cache)
    if not os.path.exists("./temp_malnet/"):
        os.mkdir("./temp_malnet/")

    for type in ["train","val","test"]:
        

        n_process = 4
        n_samples = len(dataset[type])
        n_sublist = (n_samples // n_process) + 1
        samples = range(n_samples)
        # https://stackoverflow.com/questions/2231663/slicing-a-list-into-a-list-of-sub-lists
        indices = [samples[i:i + n_sublist] for i in range(0, n_samples, n_sublist)]

        processes = []
        for local_rank in range(n_process):  # + 1 for test process
            p = Process(target=start_cache, args=(config,dataset,type, indices[local_rank], local_rank))
            p.start()
            processes.append(p)
        for p in processes:
            p.join()

    for type in ["train", "val", "test"]:
        print("collecting ",type)

        type_path = base_cache
        if type == "train":
            type_path += "training/"
        elif type =="val":
            type_path += "validation/"
        elif type == "test":
            type_path += "test/"
        if not os.path.exists(type_path):
            os.mkdir(type_path)
        cache_temp_dict = dict()
        for i in range(len(dataset[type])):
            cache_temp_dict[i] = dict()
            cache_temp_dict[i]["path"] = "./temp_malnet/{}_{}.pkl".format(type,str(i))
            cache_temp_dict[i]["num_nodes"] = dataset[type][i]['G'].number_of_nodes()


        torch.save(cache_temp_dict,type_path+"records.pt")
        torch.save(cache_temp_dict, type_path + "pe_encodings.pt")
        max_node_dict = dict()
        max_node_dict['max_nodes_index'] = 0
        max_node_dict['max_nodes'] = 1
        torch.save(max_node_dict, type_path + "max_nodes_data.pt")

@nb.njit
def calculate_magnet(edges,num_nodes):
    
        

    senders = edges[:, 0].astype(np.int64)
    receivers = edges[:, 1].astype(np.int64)
    n_node = np.array([num_nodes, 0], dtype=np.int64)
    q = 0.25
    eigenvalues, eigenvectors, laplacian = eigv_magnetic_laplacian_numba(
        senders=senders, receivers=receivers, n_node=n_node,
        padded_nodes_size=num_nodes, k=25, k_excl=0, q=q, q_absolute=False,
        norm_comps_sep=False, l2_norm=True, sign_rotate=True,
        use_symmetric_norm=True)

    if q == 0:
        eigenvectors = eigenvectors.real
        eigenvectors = eigenvectors.astype(np.float32)

        return eigenvectors, eigenvectors
    else:
        eigenvec_real = eigenvectors.real
        eigenvec_imag = eigenvectors.imag
        eigenvec_real = eigenvec_real.astype(np.float32)
        eigenvec_imag = eigenvec_imag.astype(np.float32)
        return eigenvec_real, eigenvec_imag


def start_cache(config,dataset,type, indices, local_rank):
    print("cache " + type)
    type_dataset = dataset[type]

    record_dict = dict()
    u_dict = dict()
    v_dict = dict()

    for i in tqdm(indices):
        data_graph = type_dataset[i]
        graph = dict()
        #print(local_rank,i)

        G = data_graph['G']
        A = nx.adjacency_matrix(G, weight="None").todense()
        D = floyd_warshall(A)
        graph['distance_matrix'] = D
        num_nodes = G.number_of_nodes()
        graph['num_nodes'] = num_nodes
        graph['edges'] = np.array(nx.to_edgelist(G)._viewer)
        graph['target'] = np.array(data_graph["label"]).astype(np.int64)


        u,v = calculate_magnet(graph['edges'], num_nodes)
    
        if u.shape[1] < 25:
            padding_needed = 25 - u.shape[1]


            u = np.pad(u, ((0, 0), (0, padding_needed)), 'constant', constant_values=1)
            v = np.pad(v, ((0, 0), (0, padding_needed)), 'constant', constant_values=1)

        graph['u_pe_encodings'] = u
        graph['v_pe_encodings'] = v
        with open('./temp_malnet/{}_{}.pkl'.format(type,str(i)), 'wb') as file:
            pickle.dump(graph, file)

