import numpy as np
import torch
import h5py
from tqdm import tqdm
import networkx as nx

from ..dataset_base import DatasetBase
from ..graph_dataset import GraphDataset
from ..graph_dataset import PEEncodingsGraphDataset
from ..graph_dataset import StructuralDataset


class MNISTDataset(DatasetBase):
    def __init__(self, 
                 dataset_path         ,
                 upto_hop,
                 dataset_name = 'MNIST',
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
            f = h5py.File('./raw_data/MNIST.h5', 'r')
            self._dataset = dict(f['MNIST'][self.split])
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
            for token in tqdm(np.arange(len(self.dataset.keys()))):
                self._records[token] = self.read_record(token)
        else:
            for token in self.record_tokens:
                self._records[token] = self.read_record(token)


    def read_record(self, token):
        token = f'{token:0>10d}'
        data_graph = self.dataset[token]
        data = data_graph['data']
        graph = dict()
        num_nodes = data.attrs['num_nodes']
        graph['num_nodes'] = np.array(num_nodes).astype(np.int16)
        graph['edges'] = np.array(data['edges']).astype(np.int16)
        graph['edge_features'] = np.array(data['features/edges/feat']).astype(np.float32)
        graph['node_features'] = np.array(data['features/nodes/feat']).astype(np.float32)
        graph['target'] = np.array(data_graph['targets/label'], np.int64)
        return graph



class MNISTGraphDataset(GraphDataset,MNISTDataset):
    pass

class MNISTPEGraphDataset(PEEncodingsGraphDataset,MNISTDataset):
    pass

class MNISTStructuralGraphDataset(StructuralDataset,MNISTGraphDataset):
    pass

class MNISTStructuralPEGraphDataset(StructuralDataset,MNISTPEGraphDataset):
    pass

'''
f = h5py.File('./MNIST.h5','r')
data_train = dict(f['MNIST']['training'])
print("here")
'''