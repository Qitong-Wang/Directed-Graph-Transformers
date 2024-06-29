import numpy as np
import torch
from tqdm import tqdm
import pickle
from ..dataset_base import DatasetBase
from ..graph_dataset import GraphDataset
from ..graph_dataset import PEEncodingsGraphDataset
from ..graph_dataset import StructuralDataset
import networkx as nx


class Perturbed_twitter3Dataset(DatasetBase):
    def __init__(self, 
                 dataset_path         ,
                 upto_hop,
                 dataset_name = 'Perturbed_twitter3',
                 **kwargs
                 ):
        super().__init__(dataset_name = dataset_name,
                         **kwargs)
        self.dataset_path = dataset_path
        self.upto_hop = upto_hop

    def __getitem__(self, index):
        '''
               token  = self.record_tokens[index]
               try:
                   return self._records[token]
               except AttributeError:
                   record = self.read_record(token)
                   self._records = {token:record}
                   return record
               except KeyError:
                   record = self.read_record(token)
                   self._records[token] = record
                   return record
               '''

        return self._records[index]

    @property
    def record_tokens(self):
        try:
            return self._record_tokens
        except AttributeError:
            self._record_tokens = np.arange(len(self._records ))
            return self._record_tokens

    @property
    def dataset(self):
        try:
            return self._dataset
        except AttributeError:
            path = "./raw_data/perturbed_twitter_3_dataset.pkl"
            if self.split =="training":
                with open(path, 'rb') as file:
                    self._dataset = pickle.load(file)["train"]
            elif self.split == "validation":
                with open(path, 'rb') as file:
                    self._dataset = pickle.load(file)["valid"]
            elif self.split == "test":
                with open(path, 'rb') as file:
                    self._dataset = pickle.load(file)["test"]
            return self._dataset

    def cache_load_and_save(self, base_path, op, verbose):
        #tokens_path = base_path / 'tokens.pt'
        records_path = base_path / 'records.pt'

        if op == 'load':
            self._records = torch.load(str(records_path))
        elif op == 'save':
            if  records_path.exists()  and hasattr(self, '_records'):
                return
            self.read_all_records(verbose=verbose)
            torch.save(self._records, str(records_path))
            self._records = torch.load(str(records_path))
            del self._dataset
        else:
            raise ValueError(f'Unknown operation: {op}')

    def read_all_records(self, verbose=1):
        self._records = {}
        if verbose:
            print(f'Reading all {self.split} records...', flush=True)
            for token in tqdm(np.arange(len(self.dataset))):
                self._records[token] = self.read_record(token)
        else:
            for token in self.record_tokens:
                self._records[token] = self.read_record(token)


    def read_record(self, token):
        data_graph = self.dataset[token]
        graph =  dict()

        graph['A'] = data_graph["adjacency_matrix"]
        num_nodes = data_graph["num_nodes"]
        graph['num_nodes'] =num_nodes
        G = nx.from_scipy_sparse_matrix(data_graph["adjacency_matrix"], create_using=nx.DiGraph)
        graph['edges'] =  np.array(nx.to_edgelist(G)._viewer)

        if data_graph["label"] == 0:
            label = 0
        elif data_graph["label"] == 50:
            label = 1
        elif data_graph["label"] == 100:
            label = 2
        else:
            raise ValueError('Wrong label value')
        graph['target'] = np.array(label).astype(np.int64)


        return graph


class Perturbed_twitter3GraphDataset(GraphDataset,Perturbed_twitter3Dataset):
    pass

class Perturbed_twitter3PEGraphDataset(PEEncodingsGraphDataset,Perturbed_twitter3Dataset):
    pass

class Perturbed_twitter3StructuralGraphDataset(StructuralDataset,Perturbed_twitter3GraphDataset):
    pass

class Perturbed_twitter3StructuralPEGraphDataset(StructuralDataset,Perturbed_twitter3PEGraphDataset):
    pass

