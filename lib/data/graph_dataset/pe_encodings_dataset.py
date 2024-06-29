
import numpy as np
import torch
from tqdm import trange
import numba
import numba as nb
from typing import Tuple

from .graph_dataset import GraphDataset

class PEEncodingsDatasetBase:
    def __init__(self,
                 u_pe_encodings_key = 'u_pe_encodings',
                 v_pe_encodings_key='v_pe_encodings',
                 calculated_dim    = 8,
                 output_dim        = 8,
                 random_neg_splits = ['training'],
                 **kwargs):
        if output_dim > calculated_dim:
            raise ValueError('PE: output_dim > calculated_dim')
        super().__init__(**kwargs)   
        self.u_pe_encodings_key = u_pe_encodings_key
        self.v_pe_encodings_key = v_pe_encodings_key
        self.calculated_dim    = calculated_dim    
        self.output_dim        = output_dim        
        self.random_neg_splits = random_neg_splits
    
    def calculate_encodings(self, item):
        raise NotImplementedError('PEEncodingsDatasetBase.calculate_encodings()')
    
    def __getitem__(self, index):
        item = super().__getitem__(index)
        token  = self.record_tokens[index]
        
        try:
            u_encodings = self._u_pe_encodings[token]
            v_encodings = self._v_pe_encodings[token]
        except AttributeError:
            u_encodings, v_encodings= self.calculate_encodings(item)
            self._u_pe_encodings = {token:u_encodings}
            self._v_pe_encodings = {token:v_encodings}
        except KeyError:
            u_encodings, v_encodings= self.calculate_encodings(item)
            self._u_pe_encodings = u_encodings
            self._v_pe_encodings = v_encodings
        
        if self.output_dim < self.calculated_dim:
            u_encodings = u_encodings[:,:self.output_dim,:]
            v_encodings = v_encodings[:, :self.output_dim, :]
        
        if self.split in self.random_neg_splits:
            rn_factors = np.random.randint(0, high=2, size=u_encodings.shape[1])*2-1 #size=(encodings.shape[0],1,1)
            u_encodings = u_encodings * rn_factors.astype(u_encodings.dtype)
            v_encodings = v_encodings * rn_factors.astype(v_encodings.dtype)
        
        item[self.u_pe_encodings_key] = u_encodings.reshape(u_encodings.shape[0],-1)
        item[self.v_pe_encodings_key] = v_encodings.reshape(v_encodings.shape[0], -1)
        return item
    
    def calculate_all_pe_encodings(self,verbose=1):
        self._u_pe_encodings = {}
        self._v_pe_encodings = {}
        if verbose:
            print(f'Calculating all {self.split} PE encodings...', flush=True)
            for index in trange(super().__len__()):
                item = super().__getitem__(index)
                token  = self.record_tokens[index]
                self._u_pe_encodings[token], self._v_pe_encodings[token] = self.calculate_encodings(item)
        else:
            for index in range(super().__len__()):
                item = super().__getitem__(index)
                token = self.record_tokens[index]
                self._u_pe_encodings[token], self._v_pe_encodings[token] = self.calculate_encodings(item)
    
    def cache_load_and_save(self, base_path, op, verbose):
        super().cache_load_and_save(base_path, op, verbose)
        pe_encodings_path = base_path/'pe_encodings.pt'
        
        if op == 'load':
            pe_encodings = torch.load(str(pe_encodings_path))
            self._u_pe_encodings = pe_encodings[0]
            self._v_pe_encodings = pe_encodings[1]
        elif op == 'save':
            if verbose: print(f'{self.split} PE encodings cache does not exist! Cacheing...', flush=True)
            self.calculate_all_pe_encodings(verbose=verbose)
            torch.save((self._u_pe_encodings,self._v_pe_encodings), str(pe_encodings_path))
            if verbose: print(f'Saved {self.split} PE encodings cache to disk.', flush=True)
        else:
            raise ValueError(f'Unknown operation: {op}')


@nb.njit
def calculate_svd_encodings(edges, num_nodes, calculated_dim):

    adj = np.zeros((num_nodes,num_nodes),dtype=np.float32)
    for i in range(edges.shape[0]):
        adj[nb.int64(edges[i,0]),nb.int64(edges[i,1])] = 1

    for i in range(num_nodes):
        adj[i,i] = 1

    try:
        u, s, vh = np.linalg.svd(adj)
    except:

        u = np.ones((num_nodes,calculated_dim),dtype=np.float32)
        v = np.ones((num_nodes,calculated_dim),dtype=np.float32)
        return u,v


    s = s[:calculated_dim]
    u = u[:, :calculated_dim]
    vh = vh[:calculated_dim, :]
    u_encodings = u * np.sqrt(s)
    v_encodings = vh.T * np.sqrt(s)
    return u_encodings, v_encodings

# Modify from https://github.com/google-deepmind/digraph_transformer/blob/main/utils.py
# Constant required for numerical reasons
EPS = 1e-8

# Necessary to work around numbas limitations with specifying axis in norm and
# braodcasting in parallel loops.
@numba.njit('float64[:, :](float64[:, :])', parallel=False)
def _norm_2d_along_first_dim_and_broadcast(array):
  """Equivalent to `linalg.norm(array, axis=0)[None, :] * ones_like(array)`."""
  output = np.zeros(array.shape, dtype=array.dtype)
  for i in numba.prange(array.shape[-1]):
    output[:, i] = np.linalg.norm(array[:, i])
  return output


# Necessary to work around numbas limitations with specifying axis in norm and
# braodcasting in parallel loops.
@numba.njit('float64[:, :](float64[:, :])', parallel=False)
def _max_2d_along_first_dim_and_broadcast(array):
  """Equivalent to `array.max(0)[None, :] * ones_like(array)`."""
  output = np.zeros(array.shape, dtype=array.dtype)
  for i in numba.prange(array.shape[-1]):
    output[:, i] = array[:, i].max()
  return output

@numba.njit([
    'Tuple((float64[::1], complex128[:, :], complex128[:, ::1]))(int64[:], ' +
    'int64[:], int64[:], int64, int64, int64, float64, b1, b1, b1, b1, b1)'
])
def eigv_magnetic_laplacian_numba(
    senders: np.ndarray, receivers: np.ndarray, n_node: np.ndarray,
    padded_nodes_size: int, k: int, k_excl: int, q: float, q_absolute: bool,
    norm_comps_sep: bool, l2_norm: bool, sign_rotate: bool,
    use_symmetric_norm: bool) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
  """k *complex* eigenvectors of the smallest k eigenvectors of the magnetic laplacian.

  Args:
    senders: Origin of the edges of shape [m].
    receivers: Target of the edges of shape [m].
    n_node: array shape [2]
    padded_nodes_size: int the number of nodes including padding.
    k: Returns top k eigenvectors.
    k_excl: The top (trivial) eigenvalues / -vectors to exclude.
    q: Factor in magnetic laplacian. Default 0.25.
    q_absolute: If true `q` will be used, otherwise `q / m_imag / 2`.
    norm_comps_sep: If true first imaginary part is separately normalized.
    l2_norm: If true we use l2 normalization and otherwise the abs max value.
    sign_rotate: If true we decide on the sign based on max real values and
      rotate the imaginary part.
    use_symmetric_norm: symmetric (True) or row normalization (False).

  Returns:
    array of shape [<= k] containing the k eigenvalues.
    array of shape [n, <= k] containing the k eigenvectors.
    array of shape [n, n] the laplacian.
  """
  # Handle -1 padding
  edges_padding_mask = senders >= 0

  adj = np.zeros(int(padded_nodes_size * padded_nodes_size), dtype=np.float64)
  linear_index = receivers + (senders * padded_nodes_size).astype(senders.dtype)
  adj[linear_index] = edges_padding_mask.astype(adj.dtype)
  adj = adj.reshape(padded_nodes_size, padded_nodes_size)
  # TODO(simongeisler): maybe also allow weighted matrices etc.
  adj = np.where(adj > 1, 1, adj)

  symmetric_adj = adj + adj.T
  symmetric_adj = np.where((adj != 0) & (adj.T != 0), symmetric_adj / 2,
                           symmetric_adj)

  symmetric_deg = symmetric_adj.sum(-2)

  if not q_absolute:
    m_imag = (adj != adj.T).sum() / 2
    m_imag = min(m_imag, n_node[0])
    q = q / (m_imag if m_imag > 0 else 1)

  theta = 1j * 2 * np.pi * q * (adj - adj.T)

  if use_symmetric_norm:
    inv_deg = np.zeros((padded_nodes_size, padded_nodes_size), dtype=np.float64)
    np.fill_diagonal(
        inv_deg, 1. / np.sqrt(np.where(symmetric_deg < 1, 1, symmetric_deg)))
    eye = np.eye(padded_nodes_size)
    inv_deg = inv_deg.astype(adj.dtype)
    deg = inv_deg @ symmetric_adj.astype(adj.dtype) @ inv_deg
    laplacian = eye - deg * np.exp(theta)

    mask = np.arange(padded_nodes_size) < n_node[:1]
    mask = np.expand_dims(mask, -1) & np.expand_dims(mask, 0)
    laplacian = mask.astype(adj.dtype) * laplacian
  else:
    deg = np.zeros((padded_nodes_size, padded_nodes_size), dtype=np.float64)
    np.fill_diagonal(deg, symmetric_deg)
    laplacian = deg - symmetric_adj * np.exp(theta)

  if q == 0:
    laplacian_r = np.real(laplacian)
    assert (laplacian_r == laplacian_r.T).all()
    # Avoid rounding errors of any sort
    eigenvalues, eigenvectors = np.linalg.eigh(laplacian_r)
    eigenvalues = eigenvalues[..., k_excl:k_excl + k]
    eigenvectors = eigenvectors[..., :, k_excl:k_excl + k]
    return eigenvalues.real, eigenvectors.astype(np.complex128), laplacian

  eigenvalues, eigenvectors = np.linalg.eigh(laplacian)

  eigenvalues = eigenvalues[..., k_excl:k_excl + k]
  eigenvectors = eigenvectors[..., k_excl:k_excl + k]

  if sign_rotate:
    sign = np.zeros((eigenvectors.shape[1],), dtype=eigenvectors.dtype)
    for i in range(eigenvectors.shape[1]):
      argmax_i = np.abs(eigenvectors[:, i].real).argmax()
      sign[i] = np.sign(eigenvectors[argmax_i, i].real)
    eigenvectors = np.expand_dims(sign, 0) * eigenvectors

    argmax_imag_0 = eigenvectors[:, 0].imag.argmax()
    rotation = np.angle(eigenvectors[argmax_imag_0:argmax_imag_0 + 1])
    eigenvectors = eigenvectors * np.exp(-1j * rotation)

  if norm_comps_sep:
    # Only scale eigenvectors that seems to be more than numerical errors
    eps = EPS / np.sqrt(eigenvectors.shape[0])
    if l2_norm:
      scale_real = _norm_2d_along_first_dim_and_broadcast(np.real(eigenvectors))
      real = np.real(eigenvectors) / scale_real
    else:
      scale_real = _max_2d_along_first_dim_and_broadcast(
          np.abs(np.real(eigenvectors)))
      real = np.real(eigenvectors) / scale_real
    scale_mask = np.abs(
        np.real(eigenvectors)).sum(0) / eigenvectors.shape[0] > eps
    eigenvectors[:, scale_mask] = (
        real[:, scale_mask] + 1j * np.imag(eigenvectors)[:, scale_mask])

    if l2_norm:
      scale_imag = _norm_2d_along_first_dim_and_broadcast(np.imag(eigenvectors))
      imag = np.imag(eigenvectors) / scale_imag
    else:
      scale_imag = _max_2d_along_first_dim_and_broadcast(
          np.abs(np.imag(eigenvectors)))
      imag = np.imag(eigenvectors) / scale_imag
    scale_mask = np.abs(
        np.imag(eigenvectors)).sum(0) / eigenvectors.shape[0] > eps
    eigenvectors[:, scale_mask] = (
        np.real(eigenvectors)[:, scale_mask] + 1j * imag[:, scale_mask])
  elif not l2_norm:
    scale = _max_2d_along_first_dim_and_broadcast(np.absolute(eigenvectors))
    eigenvectors = eigenvectors / scale

  return eigenvalues.real, eigenvectors, laplacian




class PEEncodingsGraphDataset(PEEncodingsDatasetBase, GraphDataset):
    def calculate_encodings(self, item):
        num_nodes = int(item[self.num_nodes_key])
        edges = item[self.edges_key]
        if self.pe_type == 'svd':
            u_encodings, v_encodings = calculate_svd_encodings(edges, num_nodes, self.calculated_dim)
            return u_encodings, v_encodings
        else:
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

