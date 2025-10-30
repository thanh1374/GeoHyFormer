import os
import os.path as osp
import pickle
import numpy as np
import scipy.sparse as sp
import torch
from torch_geometric.data import Data, InMemoryDataset

# Utilities
def safe_load_npz(path, default_shape=None):
    if path is None:
        return None
    if osp.exists(path):
        try:
            return sp.load_npz(path).todense()
        except Exception:
            # maybe it's a dense .npz saved as np.load
            try:
                return np.load(path)
            except Exception:
                return None
    else:
        return None

def load_npy_safe(path):
    if path is None or not osp.exists(path):
        return None
    return np.load(path)

def zscore_normalize(mat):
    """
    mat: np.ndarray (N, D)
    returns normalized mat (N, D) as float32
    """
    if mat is None:
        return None
    mat = np.asarray(mat, dtype=np.float32)
    mean = mat.mean(axis=0, keepdims=True)
    std = mat.std(axis=0, keepdims=True)
    std[std < 1e-6] = 1.0
    return (mat - mean) / std

# Load features & edges
def load_feature_matrix(folder):
   
    f_bert = safe_load_npz(osp.join(folder, 'new_bert_feature.npz'))
    f_content = safe_load_npz(osp.join(folder, 'new_content_feature.npz'))
    f_profile = safe_load_npz(osp.join(folder, 'new_profile_feature.npz'))

    # convert None -> zeros with inferred sizes if possible
    mats = []
    N = None
    for f in (f_bert, f_content, f_profile):
        if f is not None:
            arr = np.array(f)
            mats.append(arr)
            N = arr.shape[0] if N is None else N
        else:
            mats.append(None)

    # if no inputs exist, raise
    if N is None:
        raise FileNotFoundError("No feature files found in folder: %s" % folder)

    for i, m in enumerate(mats):
        if m is None:
            # fallback zero columns (choose width 16 as default)
            mats[i] = np.zeros((N, 16), dtype=np.float32)

    # concatenate and normalize
    x = np.concatenate(mats, axis=1)
    x = zscore_normalize(x)
    return torch.tensor(x, dtype=torch.float32)

def load_edge_index(folder):
    path = osp.join(folder, 'A.txt')
    if not osp.exists(path):
        # try alternative
        raise FileNotFoundError(f"Edge file not found: {path}")
    arr = np.loadtxt(path, delimiter=',', dtype=np.int64)
    if arr.ndim == 1 and arr.size == 0:
        return torch.empty((2, 0), dtype=torch.long)
    return torch.from_numpy(arr).t().contiguous()

def load_extra_info(folder, name):
    # try several naming conventions
    time_candidates = [f'{name}_id_time_mapping.pkl', 'id_time_mapping.pkl', 'nodeid_time_mapping.pkl']
    user_candidates = [f'{name}_id_twitter_mapping.pkl', 'id_twitter_mapping.pkl', 'nodeid_twitter_mapping.pkl']
    id_time = {}
    id_user = {}
    for p in time_candidates:
        full = osp.join(folder, p)
        if osp.exists(full):
            try:
                with open(full, 'rb') as f:
                    id_time = pickle.load(f)
                break
            except Exception:
                continue
    for p in user_candidates:
        full = osp.join(folder, p)
        if osp.exists(full):
            try:
                with open(full, 'rb') as f:
                    id_user = pickle.load(f)
                break
            except Exception:
                continue
    # ensure keys are ints
    id_time = {int(k): v for k, v in id_time.items()} if id_time else {}
    id_user = {int(k): v for k, v in id_user.items()} if id_user else {}
    return id_time, id_user

# Build per-graph Data objects (vectorized)
def build_graphs(folder, name, min_nodes=1, min_edges=0, verbose=False):
    """
    Build list of torch_geometric.data.Data objects for each graph in dataset.
    - min_nodes, min_edges: filters to skip tiny graphs
    """
    raw_dir = osp.join(folder)
    x_all = load_feature_matrix(raw_dir)                     # (N_total, D)
    edge_index_all = load_edge_index(raw_dir)                # (2, E_total)
    labels = load_npy_safe(osp.join(raw_dir, 'graph_labels.npy'))
    if labels is None:
        raise FileNotFoundError("graph_labels.npy not found in %s" % raw_dir)
    labels = torch.from_numpy(labels).long()
    # handle labels mapping (unique)
    _, labels = labels.unique(sorted=True, return_inverse=True)

    node_graph_id_arr = load_npy_safe(osp.join(raw_dir, 'node_graph_id.npy'))
    if node_graph_id_arr is None:
        raise FileNotFoundError("node_graph_id.npy not found in %s" % raw_dir)
    node_graph_id = torch.from_numpy(node_graph_id_arr).long()

    id_time, id_user = load_extra_info(raw_dir, name)

    num_graphs = int(labels.size(0))
    data_list = []

    # convert edge arrays to numpy for vector ops
    if edge_index_all.numel() == 0:
        srcs = np.array([], dtype=np.int64)
        dsts = np.array([], dtype=np.int64)
    else:
        srcs = edge_index_all[0].numpy().astype(np.int64)
        dsts = edge_index_all[1].numpy().astype(np.int64)
    N_total = x_all.size(0)

    # pre-allocate a mapping array for speed per graph
    for g in range(num_graphs):
        mask = (node_graph_id == g)
        node_idx = torch.nonzero(mask, as_tuple=False).view(-1)
        if node_idx.numel() == 0:
            if verbose:
                print(f"graph {g}: empty, skip")
            continue
        node_idx_np = node_idx.numpy().astype(np.int64)
        n_nodes = node_idx_np.size

        # vectorized mapping: arr of size N_total with -1 default
        mapping = -np.ones((N_total,), dtype=np.int64)
        mapping[node_idx_np] = np.arange(n_nodes, dtype=np.int64)

        # select edges where both endpoints in mapping (mapped != -1)
        if srcs.size == 0:
            sel_mask = np.zeros((0,), dtype=bool)
        else:
            sel_mask = (mapping[srcs] != -1) & (mapping[dsts] != -1)

        sel_count = int(sel_mask.sum())
        if sel_count == 0:
            edge_index = torch.empty((2, 0), dtype=torch.long)
            edge_type = torch.empty((0,), dtype=torch.long)
            edge_time = torch.empty((0,), dtype=torch.float)
        else:
            sel_src_global = srcs[sel_mask]
            sel_dst_global = dsts[sel_mask]
            src_local = mapping[sel_src_global]
            dst_local = mapping[sel_dst_global]
            edge_index = torch.from_numpy(np.vstack([src_local, dst_local]).astype(np.int64)).long().contiguous()

            # edge attributes vectorized
            # type heuristic: 0 if source is root (local idx 0), 1 if same user, 2 otherwise
            etype = np.ones((sel_count,), dtype=np.int64)  # default 1
            etype[src_local == 0] = 0
            if id_user:
                # same user -> type 2
                same_user = np.array([1 if id_user.get(int(sg), None) is not None and id_user.get(int(sg)) == id_user.get(int(dg), None) else 0
                                       for sg, dg in zip(sel_src_global, sel_dst_global)], dtype=np.int64)
                etype[same_user == 1] = 2

            # edge times: dt = max(0, t_d - t_s) if both exist, else 0
            etime = np.zeros((sel_count,), dtype=np.float32)
            if id_time:
                for i, (sg, dg) in enumerate(zip(sel_src_global, sel_dst_global)):
                    t_s = id_time.get(int(sg), None)
                    t_d = id_time.get(int(dg), None)
                    if t_s is not None and t_d is not None:
                        try:
                            dt = float(t_d) - float(t_s)
                        except Exception:
                            dt = 0.0
                        etime[i] = max(0.0, dt)
                    else:
                        etime[i] = 0.0

            edge_type = torch.from_numpy(etype).long()
            edge_time = torch.from_numpy(etime).float()

        # node features subset
        x = x_all[node_idx]

        # graph-level label
        y = labels[g].view(1)

        # build Data object
        data = Data(x=x, edge_index=edge_index, edge_type=edge_type, edge_time=edge_time, y=y)
        data.num_nodes = x.size(0)
        data.graph_id = torch.tensor([g], dtype=torch.long)

        # add simple structural node features: in-degree/out-degree within subgraph
        if edge_index.numel() > 0:
            src_local = edge_index[0].numpy().astype(np.int64)
            dst_local = edge_index[1].numpy().astype(np.int64)
            indeg = np.zeros((data.num_nodes,), dtype=np.int64)
            outdeg = np.zeros((data.num_nodes,), dtype=np.int64)
            for s_local, d_local in zip(src_local, dst_local):
                outdeg[s_local] += 1
                indeg[d_local] += 1
            # attach as node-level features (optional)
            data.indegree = torch.tensor(indeg, dtype=torch.float).unsqueeze(-1)
            data.outdegree = torch.tensor(outdeg, dtype=torch.float).unsqueeze(-1)
        else:
            data.indegree = torch.zeros((data.num_nodes, 1), dtype=torch.float)
            data.outdegree = torch.zeros((data.num_nodes, 1), dtype=torch.float)

        # optional filtering tiny graphs
        if data.num_nodes < min_nodes or (edge_index.size(1) < min_edges):
            if verbose:
                print(f"graph {g}: filtered by size nodes={data.num_nodes} edges={edge_index.size(1)}")
            continue

        data_list.append(data)

    return data_list

# InMemoryDataset implementation
class GHPDataset(InMemoryDataset):
    def __init__(self, root, name, transform=None, pre_transform=None, pre_filter=None, rebuild=False, verbose=False):
        self.name = name
        self.root = root
        self.verbose = verbose
        super(GHPDataset, self).__init__(root, transform, pre_transform, pre_filter)
        if rebuild or not osp.exists(self.processed_paths[0]):
            if self.verbose:
                print("Processing dataset from raw files...")
            self.process()
        loaded = torch.load(self.processed_paths[0], weights_only=False)
        # expect (data, slices, [train_idx, val_idx, test_idx])
        if len(loaded) == 5:
            self.data, self.slices, self.train_idx, self.val_idx, self.test_idx = loaded
        else:
            self.data, self.slices = loaded
            self.train_idx = self.val_idx = self.test_idx = None

    @property
    def raw_dir(self):
        return osp.join(self.root, self.name, 'raw')

    @property
    def processed_dir(self):
        return osp.join(self.root, self.name, 'processed')

    @property
    def raw_file_names(self):
        return [
            'new_bert_feature.npz',
            'new_content_feature.npz',
            'new_profile_feature.npz',
            'A.txt',
            'node_graph_id.npy',
            'graph_labels.npy',
        ]

    @property
    def processed_file_names(self):
        return f'{self.name[:3]}_data_GHP.pt'

    def _download(self):
        return

    def process(self):
        data_list = build_graphs(self.raw_dir, self.name, verbose=self.verbose)

        if self.pre_filter is not None:
            data_list = [d for d in data_list if self.pre_filter(d)]
        if self.pre_transform is not None:
            data_list = [self.pre_transform(d) for d in data_list]

        data, slices = self.collate(data_list)

        # try to save split idxs if available
        train_idx = load_npy_safe(osp.join(self.raw_dir, 'train_idx.npy'))
        val_idx = load_npy_safe(osp.join(self.raw_dir, 'val_idx.npy'))
        test_idx = load_npy_safe(osp.join(self.raw_dir, 'test_idx.npy'))

        if train_idx is not None and val_idx is not None and test_idx is not None:
            train_idx = torch.from_numpy(train_idx).long()
            val_idx = torch.from_numpy(val_idx).long()
            test_idx = torch.from_numpy(test_idx).long()
            torch.save((data, slices, train_idx, val_idx, test_idx), self.processed_paths[0])
            if self.verbose:
                print("Saved processed with splits.")
        else:
            torch.save((data, slices), self.processed_paths[0])
            if self.verbose:
                print("Saved processed without splits.")

    def __repr__(self):
        return f'{self.name}({len(self)})'
