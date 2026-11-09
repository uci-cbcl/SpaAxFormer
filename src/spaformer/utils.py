import argparse
import random
import yaml
import logging
from functools import partial
import numpy as np
import anndata as ad
import scanpy as sc
import dgl
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch import optim as optim
from tensorboardX import SummaryWriter
import scipy.sparse as sparse
from scipy.sparse import csc_matrix
import pickle
import os
from tqdm import tqdm
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import NearestNeighbors
from typing import Optional
import pandas as pd

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
from scipy.spatial.distance import cdist

logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO)

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def set_imitate_seed(seed):
    import random
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

def get_current_lr(optimizer):
    return optimizer.state_dict()["param_groups"][0]["lr"]

def objective_metadata(args):
    components = ["gep"]

    if args.gepc:
        components.append("gepc")
    if args.contrast:
        components.append("pca")

    return {
        "objective": "_".join(components),
        "use_cls": bool(args.cls),
        "use_gepc": bool(args.gepc),
        "use_contrastive": bool(args.contrast),
    }

def build_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
        
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--data_path", "--data-root", type=str, required=True)
        
    parser.add_argument("--standardscale", type=int, default=250)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--max_epoch", "--epochs", type=int, required=True,
                        help="number of training epochs")
    parser.add_argument("--warmup_steps", type=int, default=-1)
    
    parser.add_argument("--num_bins", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=15)
    parser.add_argument("--mode", type=str, default="neighbors", choices=['simple', 'neighbors'])
    parser.add_argument("--mask_ratio", type=float, default=30)
    parser.add_argument("--num_heads", type=int, default=1,
                        help="number of hidden attention heads")
    parser.add_argument("--num_out_heads", type=int, default=1,
                        help="number of output attention heads")
    parser.add_argument("--num_layers", type=int, default=4,
                        help="number of hidden layers")
    parser.add_argument("--num_hidden", type=int, default=64,
                        help="number of hidden units")
    parser.add_argument("--dropout", type=float, default=.2,
                        help="network dropout")
    parser.add_argument("--attn_drop", type=float, default=.1,
                        help="attention dropout")
    parser.add_argument("--norm", type=str, default='layernorm', choices=[None, 'layernorm', 'batchnorm', 'groupnorm'])
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="learning rate")
    parser.add_argument("--weight_decay", type=float, default=5e-4,
                        help="weight decay")
    parser.add_argument("--negative_slope", type=float, default=0.2,
                        help="the negative slope of leaky relu for GAT")
    parser.add_argument("--activation", type=str, default="gelu")
    parser.add_argument("--mask_node_rate", type=float, default=0.5)
    parser.add_argument("--mask_feature_rate", type=float, default=0.5)
    parser.add_argument("--drop_node_rate", type=float, default=0.3)

    parser.add_argument("--encoder", type=str, default="spaaxformer", choices=["spaaxformer"])
    parser.add_argument("--optimizer", type=str, default="adamw")
        
    parser.add_argument("--no_cls", dest="cls", action="store_false")
    parser.set_defaults(cls=True)
    
    parser.add_argument("--no_gepc", dest="gepc", action="store_false")
    parser.set_defaults(gepc=True)
    
    parser.add_argument("--contrast", action="store_true", default=False)
    parser.add_argument("--replicate_exp", action="store_true", default=False)
    parser.add_argument("--first_time", action="store_true", default=False)
    
    parser.add_argument("--load_model", action="store_true")
    parser.add_argument("--scheduler", action="store_true", default=False)
    parser.add_argument("--concat_hidden", action="store_true", default=False)
    parser.add_argument("--cat_pe", action="store_true", default=False)
    parser.add_argument("--latent_dim", type=int, default=20)
    parser.add_argument("--log_path", type=str, default="results/run/logs")
    parser.add_argument("--cache_path", type=str, default='./preprocessed')
    parser.add_argument("--pe", type=str, default=None, choices=['lap', 'signnet', 'bin', 'mlp', 'sinu', 'None', None])
    parser.add_argument("--test_pe", action="store_true")
    parser.add_argument("--pe_aug", action="store_true")
    parser.add_argument("--objective", type=str, default='mask', choices=['mask', 'vae', 'maskvae', 'ae'])
    parser.add_argument("--noise_pc", type=int, default=5)
    parser.add_argument("--task", required=True, choices=["imputation", "cluster"])
    parser.add_argument("--sample", required=True)
    parser.add_argument("--spatial-radius", type=float, required=True)
    parser.add_argument("--contrastive-weight", type=float, required=True)
    parser.add_argument("--repetitions", type=int, required=True)
    parser.add_argument("--expression-pca-dim", type=int, default=100)
    parser.add_argument("--output-dir", default="results/run")
    parser.add_argument("--checkpoint-dir", default="checkpoints")
    parser.add_argument("--model-path", help="Explicit trained state_dict for --load_model")
    parser.add_argument("--imputation-split", type=int, nargs="+",
                        help="One existing split ID per repetition; required for imputation")
    parser.add_argument("--num-clusters", type=int, help="Required for clustering; verify sample ground truth")
    args = parser.parse_args()
    if args.task == "cluster" and args.num_clusters is None:
        parser.error("clustering requires --num-clusters")
    if args.task == "imputation" and (args.imputation_split is None or len(args.imputation_split) != args.repetitions):
        parser.error("imputation requires one --imputation-split ID per repetition")
    if args.load_model and not args.model_path:
        parser.error("--load_model requires --model-path")
    return args


def preprocess_cluster(data_path, dataset_name, sample_number):
    
    if dataset_name.lower() != 'starmap' and dataset_name.lower() != 'merfish':
        filter_num = filter_num_calc(data_path, sample_number)
        # filter_num = 0
        print("Optimized Filter Number: ", filter_num)
    else:
        filter_num = 0
    adata = initialize(data_path, sample_number, filter_num)
    return adata

def preprocess_impute(args, data_path, dataset_name, split_version):    
    
    sample_dir = data_path + "/" + str(dataset_name)
    spatial_dir = sample_dir + "/spatial"
    
    split_dir = data_path + "/data_split/" +  str(dataset_name)
    
    test_mask_path = split_dir + "/split_" + str(split_version) + "_test_mask.npz"
    
    test_mask = np.load(test_mask_path)["arr_0"]
    
    adata = ad.read_h5ad(os.path.join(split_dir ,"filtered_adata.h5ad"))
    
    # adata.obs['Region'] = adata.obs['Ground Truth']
    
    
    adata_train = adata.copy()
    '''This should not matter - test mask will be replaced anyway'''
    # adata_train.X[test_mask] = 0    
    # adata = adata.copy()[:, adata.var["highly_variable"]]
    # adata = adata_train.copy()
    
    return adata, adata_train.X.copy(), test_mask

# code from: https://github.com/maiziezhoulab/ADEPT/blob/main/GAAE/utils.py#L55
def create_kNN(adata, rad_cutoff=None, k_cutoff=None, model='Radius', verbose=True):
    """
    Construct the kNN graph.

    Parameters
    ----------
    adata
        AnnData object of scanpy package.
    rad_cutoff
        radius cutoff when model='Radius'
    k_cutoff
        The number of nearest neighbors when model='KNN'
    model
        The network construction model. When model=='Radius', the spot is connected to spots whose distance is less than rad_cutoff. When model=='KNN', the spot is connected to its first k_cutoff nearest neighbors.
    
    Returns
    -------
    The spatial networks are saved in adata.uns['Spatial_Net']
    """

    assert(model in ['Radius', 'KNN'])
    if verbose:
        print('------Calculating spatial graph...')
    coor = pd.DataFrame(adata.obsm['spatial'])
    coor.index = adata.obs.index
    if adata.obsm['spatial'].shape[1] == 2:
        coor.columns = ['imagerow', 'imagecol']
    else:
        coor.columns = ['imagerow', 'imagecol', 'imagez']

    if model == 'Radius':
        nbrs = sklearn.neighbors.NearestNeighbors(radius=rad_cutoff).fit(coor)
        distances, indices = nbrs.radius_neighbors(coor, return_distance=True)
        KNN_list = []
        for it in range(indices.shape[0]):
            KNN_list.append(pd.DataFrame(zip([it]*indices[it].shape[0], indices[it], distances[it])))
    
    if model == 'KNN':
        nbrs = sklearn.neighbors.NearestNeighbors(n_neighbors=k_cutoff+1).fit(coor)
        distances, indices = nbrs.kneighbors(coor)
        KNN_list = []
        for it in range(indices.shape[0]):
            KNN_list.append(pd.DataFrame(zip([it]*indices.shape[1],indices[it,:], distances[it,:])))

    KNN_df = pd.concat(KNN_list)
    KNN_df.columns = ['Cell1', 'Cell2', 'Distance']

    Spatial_Net = KNN_df.copy()
    Spatial_Net = Spatial_Net.loc[Spatial_Net['Distance']>0,]
    id_cell_trans = dict(zip(range(coor.shape[0]), np.array(coor.index), ))
    Spatial_Net['Cell1'] = Spatial_Net['Cell1'].map(id_cell_trans)
    Spatial_Net['Cell2'] = Spatial_Net['Cell2'].map(id_cell_trans)
    if verbose:
        print('The graph contains %d edges, %d cells.' %(Spatial_Net.shape[0], adata.n_obs))
        print('%.4f neighbors per cell on average.' %(Spatial_Net.shape[0]/adata.n_obs))

    adata.uns['Spatial_Net'] = Spatial_Net
    
    
def create_spatial_attn_mask(spatial, k=None, radius=None, mode='Radius'):
    """
    Build boolean attention mask from spatial coordinates.

    Returns:
        attn_mask: [C, C] torch.BoolTensor where True = masked, False = attend
    """
    
    
    if isinstance(spatial, torch.Tensor):
        spatial = spatial.cpu().numpy()
    C = spatial.shape[0]
    attn_mask = torch.ones((C, C), dtype=torch.bool)

    if mode == 'KNN':
        nbrs = NearestNeighbors(n_neighbors=k+1).fit(spatial)
        _, indices = nbrs.kneighbors(spatial)
        knn_idx = indices[:, 1:]  # remove self
        attn_mask[torch.arange(C).unsqueeze(1), torch.tensor(knn_idx)] = False

    elif mode == 'Radius':
        nbrs = NearestNeighbors(radius=radius).fit(spatial)
        _, indices = nbrs.radius_neighbors(spatial)
        for i, nbrs_i in enumerate(indices):
            nbrs_i = nbrs_i[nbrs_i != i]  # exclude self
            attn_mask[i, torch.tensor(nbrs_i)] = False
    
    elif mode == 'Hybrid':
        # Step 1: radius neighbors
        nbrs_r = NearestNeighbors(radius=radius).fit(spatial)
        _, indices_r = nbrs_r.radius_neighbors(spatial)

        # Step 2: kNN neighbors
        nbrs_k = NearestNeighbors(n_neighbors=k+1).fit(spatial)
        _, indices_k = nbrs_k.kneighbors(spatial)

        for i, (nbrs_i_r, nbrs_i_k) in enumerate(zip(indices_r, indices_k)):
            nbrs_i_r = nbrs_i_r[nbrs_i_r != i]
            nbrs_i_k = nbrs_i_k[1:]  # remove self

            # combine: radius neighbors + extra kNN to guarantee min k
            if len(nbrs_i_r) >= k:
                final_nbrs = nbrs_i_r
            else:
                needed = k - len(nbrs_i_r)
                extra = [x for x in nbrs_i_k if x not in nbrs_i_r][:needed]
                final_nbrs = list(nbrs_i_r) + extra

            if len(final_nbrs) > 0:
                attn_mask[i, torch.tensor(final_nbrs)] = False

    return attn_mask

def filter_adj_by_pca(
    adj: torch.Tensor,
    pca_embeddings: torch.Tensor,
    pca_threshold: float,
    allow_self: bool = True,
):
    """
    Filter an existing attention mask using PCA cosine similarity.

    An edge remains allowed only when:
      1. it was already allowed in adj, and
      2. PCA cosine similarity >= pca_threshold.

    Mask convention:
      True  = blocked
      False = allowed
    """
     
    if adj.ndim != 2:
        raise ValueError(
            f"adj must be 2D, got {adj.shape}."
        )

    if pca_embeddings.ndim != 2:
        raise ValueError(
            "pca_embeddings must have shape [C, pca_dim]."
        )

    C = adj.shape[0]

    if adj.shape != (C, C):
        raise ValueError("adj must be square.")

    if pca_embeddings.shape[0] != C:
        raise ValueError(
            "adj and pca_embeddings must contain "
            "the same number of cells."
        )
    device = pca_embeddings.device

    adj = adj.to(
        device=device,
        dtype=torch.bool,
    )

    pca_embeddings = pca_embeddings.float()

    pca_norm = F.normalize(
        pca_embeddings,
        p=2,
        dim=1,
        eps=1e-8,
    )
    
    pca_similarity = pca_norm @ pca_norm.T

    pca_allowed = (
        pca_similarity >= pca_threshold
    )

    spatial_allowed = adj

    final_allowed = (
        spatial_allowed
        & pca_allowed
    )

    filtered_pca_adj = final_allowed

    diagonal = torch.arange(
        C,
        device=device,
    )

    if allow_self:
        filtered_pca_adj[diagonal, diagonal] = True
    else:
        filtered_pca_adj[diagonal, diagonal] = False

    return filtered_pca_adj


def measure_pca_quantile(X_pca, adj):  
    
                  
    pca_norm = F.normalize(
        X_pca.float(),
        p=2,
        dim=1,
        eps=1e-8,
    )

    pca_similarity = pca_norm @ pca_norm.T

    self_mask = torch.eye(
        X_pca.shape[0],
        dtype=torch.bool,
        device=X_pca.device,
    )

    spatial_no_self = adj.bool() & ~self_mask
    spatial_pca_values = pca_similarity[spatial_no_self]

    q10, q20, q25, q50, q75 = torch.quantile(
        spatial_pca_values,
        torch.tensor(
            [0.10, 0.20, 0.25, 0.50, 0.75],
            device=X_pca.device,
        ),
    )
    print(f"10%: {q10:.3f}")
    print(f"20%: {q20:.3f}")
    print(f"25%: {q25:.3f}")
    print(f"50%: {q50:.3f}")
    print(f"75%: {q75:.3f}")
    
    
def measure_pca(X_pca, adjacency_matrix):
    
    self_mask = torch.eye(
    adjacency_matrix.shape[0],
    dtype=torch.bool,
    device=adjacency_matrix.device,
    )   

    spatial_no_self = (adjacency_matrix.bool()) & (~self_mask)
    
    for d in [20, 30, 50, 75, 100]:
        X_d = X_pca[:, :d]

        X_d = F.normalize(
            X_d.float(),
            p=2,
            dim=1,
            eps=1e-8,
        )

        sim_d = X_d @ X_d.T
        spatial_values = sim_d[spatial_no_self]

        print(
            d,
            torch.quantile(
                spatial_values,
                torch.tensor(
                    [0.10, 0.20, 0.25, 0.50],
                    device=spatial_values.device,
                ),
            ),
        )
    

'''Temporary function for use'''
def adj_to_edge_index(adj, include_self=False):
    """
    Input:

    Output:
        edge_index: [2, E]
            edge_index[0] = dst
            edge_index[1] = src
    """
    
    if not isinstance(adj, torch.Tensor):
        adj = torch.tensor(adj, dtype=torch.bool)
        
    # adj.fill_diagonal_(False)
    

    if include_self:
        adj = adj.clone()
        adj.fill_diagonal_(True)
        
    '''True implies edge exists'''
    idx = adj.nonzero(as_tuple=False)  # [E, 2]
    dst = idx[:, 0]
    src = idx[:, 1]

    edge_index = torch.stack([dst, src], dim=0)  # [2, E]
    return edge_index


def compute_gaussian_spatial_bias(coords, sigma=None):
    """
    Computes -||x_i - x_j||^2 / (2 * sigma^2) for all cell pairs.

    Args:
        coords: torch.Tensor of shape [N, 2] or [B, N, 2]
        sigma: Optional float — bandwidth for Gaussian decay.
               If None, use mean pairwise distance.

    Returns:
        bias: [N, N] or [B, N, N] — to be added to attention logits
    """
    
    if isinstance(coords, np.ndarray):
        coords = torch.from_numpy(coords).float()
    elif not torch.is_tensor(coords):
        raise TypeError("coords must be a numpy array or torch tensor")
    
    if coords.ndim == 2:
        # [N, 2]
        dists = torch.cdist(coords, coords, p=2)  # [N, N]
    elif coords.ndim == 3:
        # [B, N, 2]
        dists = torch.cdist(coords, coords, p=2)  # [B, N, N]
    else:
        raise ValueError("coords must be of shape [N, 2] or [B, N, 2]")
    
    
    return dists


# code modified from: https://github.com/liuwei-cs/SpaGIC/blob/main/SpaGIC/utils.py
def Noise_Cross_Entropy(emb, adj):  
    
    N = adj.shape[0]
    mask_self = torch.eye(N, dtype=torch.bool, device=adj.device)
    pos_mask = adj.bool() & (~mask_self)        # positives only
    neg_mask = (~adj) & (~mask_self)       # negatives (no self)
    
    cos_sim = cosine_sim_tensor(emb)
    sim_exp = torch.exp(cos_sim)

    # negative pairs
    n = torch.mul(sim_exp, neg_mask).sum(axis=1)
    # positive pairs
    p = torch.mul(sim_exp, pos_mask).sum(axis=1)

    ave = torch.div(p, n)
    loss = -torch.log(ave).mean()

    return loss

# code modified from: https://github.com/liuwei-cs/SpaGIC/blob/main/SpaGIC/utils.py
# Calculate cosine similarity.
def cosine_sim_tensor(emb):
    
    N = emb.shape[0]
    
    M = torch.matmul(emb, emb.T)
    length = torch.norm(emb, p=2, dim=1)
    
    Norm = torch.matmul(length.reshape(N, 1), length.reshape(1, N)) + 1e-12
    M = M / Norm
    M = torch.nan_to_num(M, nan=0.0)

    return M

def Noise_Cross_Entropy_anchor_only(emb_pool, len_anchor_idx, pos_mask, tau=1.0, eps=1e-12):
    """
    emb_pool : [K, D]      K=35 (anchors + neighbors + other candidates)
    anchor_idx: [B]        B=5 indices of anchors inside the pool (0..K-1)
    pos_mask : [B, K]      True where candidate j is a positive (neighbor) of anchor i
    """
    
    
    K, D = emb_pool.shape
    B = len_anchor_idx
    device = emb_pool.device
    anchor_idx = torch.arange(B, device=device) 

    # cosine similarity logits between anchors and candidates -> [B,K]
    pool = F.normalize(emb_pool, p=2, dim=-1, eps=1e-8)   # [K,D]
    anc  = pool[:len_anchor_idx]                                # [B,D]
    logits = (anc @ pool.T) / tau                          # [B,K]
    

    # exclude self candidate (anchor compared to itself in the pool)
    self_mask = torch.zeros((B, K), dtype=torch.bool, device=device)
    self_mask[torch.arange(B, device=device), anchor_idx] = True

    pos = pos_mask & (~self_mask)
    neg = (~pos_mask) & (~self_mask)

    # per-anchor sums
    exp_logits = torch.exp(logits)
    p = (exp_logits * pos).sum(dim=1)   # [B]
    n = (exp_logits * neg).sum(dim=1)   # [B]

    # valid anchors must have at least one positive and one negative
    valid = pos.any(dim=1) & neg.any(dim=1)
    if valid.sum() == 0:
        return torch.tensor(0.0, device=device)

    ave = (p[valid] + eps) / (n[valid] + eps)
    loss = -torch.log(ave).mean()
    return loss


def weighted_spatial_contrastive(
    emb_pool,        # [K, D] cell embeddings (base+neighbors+optional others)
    C_dst,           # int, number of anchors (first C_dst in pool)
    adj_sub,         # [B, K] bool adjacency on pool
    epoch,
    tau=1.0,
    alpha=5.0,       # weight sharpness (bigger => closer to hard gating)
    eps=1e-12,
):
    """
    Positives = spatial neighbors (adj_sub), weighted by cosine similarity.
    Denominator = all non-self candidates (standard multi-positive InfoNCE denominator).
    """
    
    device = emb_pool.device
    K, D = emb_pool.shape
    B = C_dst

    z = F.normalize(emb_pool, dim=-1, eps=1e-8)      # [K, D]
    anc = z[:B]                                      # [B, D]
    logits = (anc @ z.T) / tau                       # [B, K]

    # masks
    pos_mask = adj_sub[:B, :].bool()                 # [B, K]
    self_mask = torch.zeros((B, K), dtype=torch.bool, device=device)
    self_mask[torch.arange(B), torch.arange(B)] = True

    pos_mask = pos_mask & (~self_mask)
    denom_mask = ~self_mask

    valid = pos_mask.any(dim=1)
    if valid.sum() == 0:
        return torch.tensor(0.0, device=device, dtype=emb_pool.dtype)
    

    # cosine similarity (without /tau) for weights
    cos = anc @ z.T                                  # [B, K], in [-1,1]

    # turn cosine into weights in [0,1]
    # Option: sigmoid centered at 0: w = sigmoid(alpha * cos)
    w = torch.sigmoid(alpha * cos)

    # only weight positives; non-positives get weight 0 in numerator
    w = w * pos_mask.float()

    # stable log numerator: log sum_j w_ij * exp(logits_ij)
    # compute as logsumexp(logits + log(w)) over positives
    neg_inf = torch.finfo(logits.dtype).min
    logw = torch.log(w + eps)                        # [B, K]
    num_terms = (logits + logw).masked_fill(~pos_mask, neg_inf)
    log_num = torch.logsumexp(num_terms[valid], dim=1)

    # denominator over all non-self
    den_terms = logits.masked_fill(~denom_mask, neg_inf)
    log_den = torch.logsumexp(den_terms[valid], dim=1)

    loss = -(log_num - log_den).mean()
    return loss


def weighted_gene_column_contrastive(
    emb_pool_gene,   # [K, G, D]
    C_dst,           # number of anchor/base cells
    adj_sub,         # [K, K] or [B, K] bool adjacency
    epoch,
    tau=1.0,
    alpha=5.0,
    eps=1e-12,
):
    """
    Apply weighted spatial contrastive loss column-wise for every gene.

    For each gene g:
        anchors = first C_dst cells
        positives = spatial neighbors from adj_sub
        denominator = all non-self cells
    Then average over anchors and genes.
    """
    

    device = emb_pool_gene.device
    dtype = emb_pool_gene.dtype

    K, G, D = emb_pool_gene.shape
    B = C_dst

    # [K, G, D]
    z = F.normalize(emb_pool_gene, dim=-1, eps=1e-8)

    # anchors: [B, G, D]
    anc = z[:B]

    # logits[b, g, k] = similarity between anchor cell b and candidate cell k for gene g
    logits = torch.einsum("bgd,kgd->bgk", anc, z) / tau   # [B, G, K]

    # adjacency mask
    pos_mask = adj_sub[:B, :].bool().to(device)           # [B, K]

    self_mask = torch.zeros((B, K), dtype=torch.bool, device=device)
    self_mask[torch.arange(B), torch.arange(B)] = True

    pos_mask = pos_mask & (~self_mask)                    # [B, K]
    denom_mask = ~self_mask                               # [B, K]

    valid = pos_mask.any(dim=1)                           # [B]

    if valid.sum() == 0:
        return torch.tensor(0.0, device=device, dtype=dtype)

    # expand masks over genes
    pos_mask_g = pos_mask[:, None, :]                     # [B, 1, K]
    denom_mask_g = denom_mask[:, None, :]                 # [B, 1, K]

    # cosine similarity without tau
    cos = torch.einsum("bgd,kgd->bgk", anc, z)            # [B, G, K]

    # weights only for positives
    w = torch.sigmoid(alpha * cos)                        # [B, G, K]
    w = w * pos_mask_g.float()

    neg_inf = torch.finfo(logits.dtype).min

    logw = torch.log(w + eps)

    # numerator: weighted positives
    num_terms = logits + logw
    num_terms = num_terms.masked_fill(~pos_mask_g, neg_inf)

    # denominator: all non-self cells
    den_terms = logits.masked_fill(~denom_mask_g, neg_inf)

    # keep only valid anchors
    num_terms = num_terms[valid]                          # [B_valid, G, K]
    den_terms = den_terms[valid]                          # [B_valid, G, K]

    log_num = torch.logsumexp(num_terms, dim=-1)          # [B_valid, G]
    log_den = torch.logsumexp(den_terms, dim=-1)          # [B_valid, G]

    loss = -(log_num - log_den).mean()                    # average over cells and genes

    return loss

 
def Contrastive_Loss_Spatial(
    emb,
    attn_mask,
    tau: float = 0.2,
    eps: float = 1e-8,
    gaussian_weights: torch.Tensor = None,
):
    """
    Spatial contrastive loss (neighbor-based InfoNCE, compatible with attention mask convention).

    Args:
        emb: [N, D] tensor of embeddings.
        attn_mask: [N, N] bool tensor where True = masked (non-neighbor), False = attend (neighbor).
        tau: temperature parameter (smaller -> stronger contrast).
        eps: numerical stability.
        gaussian_weights: optional [N, N] float tensor with 0..1 weights for positives (e.g., Gaussian by distance).

    Returns:
        loss: scalar tensor.
    """

    # Normalize embeddings to use cosine similarity
    z = F.normalize(emb, dim=1)
    sim = torch.matmul(z, z.T) / tau            # [N, N]
    sim = sim - torch.eye(len(z), device=z.device) * 1e9  # mask self-similarity

    # Exponentiate similarities
    sim_exp = torch.exp(sim)

    # Positives = unmasked (neighbors) = ~attn_mask
    pos_mask = ~attn_mask

    if gaussian_weights is not None:
        # multiply weights only on positive pairs
        sim_exp_pos = sim_exp * pos_mask * gaussian_weights
    else:
        sim_exp_pos = sim_exp * pos_mask

    pos_sum = sim_exp_pos.sum(dim=1) + eps
    all_sum = sim_exp.sum(dim=1) + eps

    # Loss = -log( sum_pos / sum_all )
    loss = -torch.log(pos_sum / all_sum)
    loss = torch.nanmean(loss)

    return loss



import torch
import numpy as np
from sklearn.neighbors import NearestNeighbors

def create_gene_attn_mask(gt, mode="KNN", k=None, radius=None):
    """
    Build boolean attention mask for a gene-gene graph based on HVG expression.

    Args:
        adata: AnnData object (already subset to HVGs or with HVG mask available).
        mode: 'KNN', 'Radius', or 'Hybrid'
        k: number of neighbors (for KNN/Hybrid)
        radius: radius cutoff (for Radius/Hybrid)

    Returns:
        attn_mask: torch.BoolTensor [G, G] where True=masked, False=attend
        edge_index: torch.LongTensor [2, E] for graph libs
    """

    # ---- use HVG expression data ----
    if isinstance(gt, np.ndarray):
        feats = gt.T  # genes × cells
    else:
        feats = gt.toarray().T
    
    G = feats.shape[0]
    attn_mask = torch.ones((G, G), dtype=torch.bool)
    edge_index = []
    
    feats = ad.AnnData(X=feats)
    
    
    feats.var_names = [f"gene{i}" for i in range(feats.n_vars)]
    sc.pp.highly_variable_genes(feats, n_top_genes=100)
    hvg_indices = feats.var["highly_variable"]
    hvg_data = (feats.X.astype(np.float32))[:, hvg_indices]
    
    if mode == "KNN":
        nbrs = NearestNeighbors(n_neighbors=k+1).fit(hvg_data)
        _, indices = nbrs.kneighbors(hvg_data)
        for i in range(G):
            nbrs_i = indices[i, 1:]  # exclude self
            attn_mask[i, torch.tensor(nbrs_i)] = False
            edge_index += [[i, j] for j in nbrs_i]
    
    return attn_mask


# '''Creates attn mask using KNN with euclidean distance'''
def create_knn_attn_mask(processed_X, pairwise):
    k = 6
    
    C, G = processed_X.shape
    knn_dist, knn_idx = torch.topk(pairwise, k=k+1, largest=False)
    knn_idx = knn_idx[:, 1:] 
    knn_dist = knn_dist[:, 1:]                                   
    
    # attn_mask = processed_X.new_full((C, C), True, dtype=torch.bool)
    # attn_mask[torch.arange(C).unsqueeze(1), knn_idx] = False
    
    sigma = knn_dist.mean()
    
    # Step 2: Compute weights (Gaussian kernel)
    weights = torch.exp(-knn_dist ** 2 / (2 * sigma ** 2))  # shape [C, K]
    
    # Step 3: Initialize zero attention weights
    attn_weights = torch.zeros(C, C, device=processed_X.device)
    weights = weights.to(dtype=attn_weights.dtype)


    # Step 4: Fill in the attention weights for KNN neighbors
    attn_weights[torch.arange(C).unsqueeze(1), knn_idx] = weights
    
    attn_mask = attn_weights
    
    
    return attn_mask  


def filter_outer_positives(coords, z, r_small=150, r_large=450, sim_threshold=0.5):  
    coords_np = coords.cpu().numpy()
    
    # Step 1: Compute pairwise distances (Euclidean)
    dists = cdist(coords_np, coords_np, metric='euclidean')  # [N, N]

    # Step 2: Get pairwise cosine similarity
    z_np = z.cpu().numpy()
    cos_sim = cosine_similarity(z_np)  # [N, N]

    # Step 3: Identify pairs within r_large but not within r_small
    within_large = (dists <= r_large)
    within_small = (dists <= r_small)
    outer_ring = np.logical_and(within_large, ~within_small)

    # Step 4: Select pairs with high similarity
    sim_mask = cos_sim > sim_threshold
    final_mask = np.logical_and(outer_ring, sim_mask)

    # Step 5: Extract index pairs (i, j)
    i_idx, j_idx = np.where(final_mask)
    pairs = list(zip(i_idx, j_idx))
    
    return pairs, final_mask  


def filter_adjacency_by_similarity(adj_450: torch.Tensor, z: torch.Tensor, sim_thresh: float = 0.9):
    """
    Args:
        adj_450: [N, N] torch.BoolTensor or torch.FloatTensor with 1.0/0.0 indicating neighbors (radius=450).
        z: [N, D] latent embeddings (L2-normalized or not).
        sim_thresh: cosine similarity threshold to keep an edge.

    Returns:
        filtered_adj: [N, N] BoolTensor or FloatTensor — filtered adjacency matrix.
    """   
    N = z.shape[0]

    # Step 1: Normalize embeddings for cosine sim
    z = F.normalize(z, dim=1)  # unit norm

    # Step 2: Cosine similarity matrix
    cos_sim = torch.matmul(z, z.T)  # [N, N], symmetric

    # Step 3: Filter adjacency
    mask = (adj_450 > 0) & (cos_sim > sim_thresh)

    return mask.float() if adj_450.dtype == torch.float32 else mask


def filter_adj_topk_sim(adj_450, z, top_k=15):  
    z = F.normalize(z, dim=1)
    cos_sim = torch.matmul(z, z.T)  # [N, N]
    
    # Mask out non-neighbors by setting sim to -inf
    masked_sim = cos_sim.masked_fill(~adj_450.bool(), float('-inf'))

    # Top-k for each row
    topk_vals, topk_idx = torch.topk(masked_sim, k=top_k, dim=1)

    # Create new adjacency
    new_adj = torch.zeros_like(adj_450)
    row_idx = torch.arange(z.size(0)).unsqueeze(1).expand_as(topk_idx)
    new_adj[row_idx, topk_idx] = 1.0

    return new_adj


def gene_sim_graph(gex, k=10):
    from sklearn.metrics.pairwise import cosine_similarity
    from sklearn.preprocessing import StandardScaler
    
    
    gex_std = StandardScaler().fit_transform(gex)
    S = cosine_similarity(gex_std.T)

    # build top-k adjacency
    # adj = torch.zeros_like(torch.tensor(S))
    # for i in range(S.shape[0]):
    #     idx = np.argsort(S[i])[-k-1:-1]
    #     adj[i, idx] = torch.tensor(S[i, idx])
    
    S_torch = torch.tensor(S, dtype=torch.float32)

    # build top-k boolean mask
    _, topk_idx = torch.topk(S_torch, k=k+1, dim=-1)  # +1 to include itself
    mask = torch.ones_like(S_torch, dtype=torch.bool)

    # unmask the top-k neighbors (set to False where connection exists)
    for i in range(S_torch.shape[0]):
        mask[i, topk_idx[i]] = False

    # Optional: make sure diagonal is always unmasked or masked as you prefer
    mask.fill_diagonal_(True)
    
    return mask

def _digitize(x: np.ndarray, bins: np.ndarray, side: str = "both") -> np.ndarray:
    """
    Same helper you had before: spreads ties across bins when edges repeat.
    This is to spread values that equal plateau edges to uniformly distributed among plateaued value bins
    ex. [0,0,0,5,5,5,10,10] -> for 0 we have choice 1,2,3
    """
    
    assert x.ndim == 1 and bins.ndim == 1

    left_digits = np.digitize(x, bins)  # right=False
    if side == "one":
        return left_digits

    right_digits = np.digitize(x, bins, right=True)

    rands = np.random.rand(len(x))  # uniform [0, 1)
    digits = rands * (right_digits - left_digits) + left_digits
    digits = np.ceil(digits).astype(np.int64)
    return digits


def compute_z_and_bins_from_logX(
    X_log: np.ndarray,
    n_bins: int,
    eps: float = 1e-6,
):
    """
    Given an already log-normalized HVG matrix X_log (cells x genes),
    compute:

      - mu_j, sigma_j (per-gene statistics),
      - Z_ij = (X_log_ij - mu_j) / sigma_j,
      - per-gene quantile bins bin_ids_ij (0..n_bins-1),
      - per-gene bin_edges[j, :] = [0, q1, ..., q_{n_bins-1}].

    Assumptions:
      - X_log >= 0 (zeros correspond to no expression).
      - n_bins >= 2 and bin 0 is reserved for zeros.
    """
    # import pdb 
    
    X_log = np.asarray(X_log, dtype=np.float32)
    if X_log.min() < 0:
        raise ValueError(f"X_log must be non-negative, got min={X_log.min()}.")

    if not isinstance(n_bins, int) or n_bins < 2:
        raise ValueError(f"`n_bins` must be an integer >= 2, got {n_bins}.")

    n_cells, n_genes = X_log.shape

    # ---- Per-gene mu_j, sigma_j, Z_ij ----
    mu = X_log.mean(axis=0)              # (n_genes,)
    sigma = X_log.std(axis=0)      # (n_genes,)
    Z = (X_log - mu[None, :]) / sigma[None, :]   # (n_cells, n_genes)

    # ---- Per-gene quantile binning ----
    bin_ids = np.zeros((n_cells, n_genes), dtype=np.int64)
    '''We have bin_edges as well'''
    bin_edges = np.zeros((n_genes, n_bins), dtype=np.float32)

    zero_gene_count = 0

    for j in range(n_genes):
        col = X_log[:, j]           # log-normalized values for gene j

        # Zero entries (assumed "no expression")
        zero_mask = col == 0.0
        non_zero_mask = ~zero_mask

        if not non_zero_mask.any():
            # All-zero gene
            zero_gene_count += 1
            bin_edges[j, :] = 0.0
            # bin_ids[:, j] already 0
            continue

        # We only pick non zero values
        non_zero_vals = col[non_zero_mask]

        # Gives bins with continuous numbers with its quantiles
        bins = np.quantile(non_zero_vals, np.linspace(0, 1, n_bins - 1))
        

        # Digitize non-zero entries to {1, ..., n_bins - 1}
        digits = _digitize(non_zero_vals, bins)
        assert digits.min() >= 1
        assert digits.max() <= n_bins - 1

        col_bins = np.zeros(n_cells, dtype=np.int64)
        col_bins[non_zero_mask] = digits

        bin_ids[:, j] = col_bins
        bin_edges[j, :] = np.concatenate([[0.0], bins.astype(np.float32)])

    if zero_gene_count > 0:
        print(
            f"Warning: {zero_gene_count} genes are all zero and were assigned "
            f"bin_ids=0 and bin_edges=[0,...,0]."
        )

    return Z, bin_ids                 # (n_cells, n_genes) z_ij  # (n_cells, n_genes) int64


def compute_z_and_bins_from_logX_mask(
    X_log: np.ndarray,
    n_bins: int,
    exclude_mask: np.ndarray = None,
    eps: float = 1e-6,
):
    
    X_log = np.asarray(X_log, dtype=np.float32)

    if exclude_mask is None:
        '''For clustering'''
        exclude_mask = np.zeros_like(X_log, dtype=bool)
    else:
        exclude_mask = np.asarray(exclude_mask, dtype=bool)

    observed = ~exclude_mask

    n_cells, n_genes = X_log.shape
    Z = np.zeros_like(X_log, dtype=np.float32)
    bin_ids = np.zeros((n_cells, n_genes), dtype=np.int64)

    for j in range(n_genes):
        col = X_log[:, j]
        observed_j = observed[:, j]

        # Fit z-score parameters only on fixed training-observed entries.
        observed_values = col[observed_j]

        if observed_values.size == 0:
            continue

        mu_j = observed_values.mean()
        sigma_j = max(observed_values.std(), eps)

        # Assign magnitudes only to observed entries.
        Z[observed_j, j] = (
            observed_values - mu_j
        ) / sigma_j

        # Real observed zeros remain ordinary bin 0.
        observed_nonzero_j = observed_j & (col != 0)

        if not observed_nonzero_j.any():
            continue

        nonzero_values = col[observed_nonzero_j]

        edges = np.quantile(
            nonzero_values,
            np.linspace(0, 1, n_bins - 1),
        )

        digits = _digitize(nonzero_values, edges)
        bin_ids[observed_nonzero_j, j] = digits

    # Excluded entries remain placeholder z=0 and bin=0.
    # EntryTokenizer replaces bin=0 with the special masked-bin ID
    # wherever mask_in is True.
    return Z, bin_ids


def iter_cell_batches(
    adjacency_full: torch.Tensor,   # [n_cells, n_cells], CPU bool - This is actually tensor
    batch_size: int,
    mode: str = "simple",        # "simple" or "neighbors"
    shuffle: bool = True,
    num_extra_neg: int = 0,
    generator: Optional[torch.Generator] = None,
    device : str = ""
):
    """
    Yields cell_ids_cpu for each batch.

    - mode="simple":
        all_ids = base_ids (just chunked cells).
    - mode="neighbors":
        all_ids = base_ids ∪ global neighbors from adjacency_full.
    """
    
    adjacency_full = adjacency_full.bool()
    n_cells = adjacency_full.shape[0]
    

    if shuffle:
        idx = torch.randperm(n_cells, generator=generator, device=device)
    else:
        idx = torch.arange(n_cells, device=device)
    
    for start in range(0, n_cells, batch_size):
        base_ids = idx[start : start + batch_size]  # [<= batch_size]
        if base_ids.numel() == 0:
            continue

        if mode == "simple":
            all_ids = base_ids
        elif mode == "neighbors":
            # neighbors of ANY base cell
            neighbor_mask_global = adjacency_full[base_ids].any(dim=0)  # [n_cells]
            neighbor_ids = neighbor_mask_global.nonzero(as_tuple=False).squeeze(-1)
            
            
            in_base = torch.isin(neighbor_ids, base_ids)
            neighbor_ids = neighbor_ids[~in_base]
            all_ids = torch.cat([base_ids, neighbor_ids], dim=0)
            
            '''introduces extra negatives'''
            # if num_extra_neg > 0:
            #     # Exclude current batch nodes
            #     # import pdb
            #     # pdb.set_trace()
                
            #     exclude_mask = torch.zeros(n_cells, dtype=torch.bool, device=adjacency_full.device)
            #     exclude_mask[all_ids] = True

            #     # Exclude neighbors of any base (so they are true non-neighbors wrt anchors)
            #     # neighbor_mask_global already marks neighbors of any base
            #     neg_candidate_mask = (~exclude_mask) & (~neighbor_mask_global)

            #     neg_candidate_ids = neg_candidate_mask.nonzero(as_tuple=False).squeeze(-1)

            #     if neg_candidate_ids.numel() > 0:
            #         m = min(num_extra_neg, neg_candidate_ids.numel())
            #         perm = torch.randperm(neg_candidate_ids.numel(), device=adjacency_full.device)[:m]
            #         extra_neg_ids = neg_candidate_ids[perm]

            #         all_ids = torch.cat([all_ids, extra_neg_ids], dim=0)
            
        else:
            raise ValueError(f"Unknown mode: {mode}")

        yield base_ids, all_ids  # 1D LongTensor of cell indices for this batch


def encoding_mask(gex, pc_train_from_rest=20, eval=False, first_time=False):
        '''
        Introduce mask here.
        gex corresponds to log-transformed input. Nothing else.
        Here M_in and M_train are equal
        As for clustering we do not need eval_mask
        '''
        
        print("mask zero scheme used")
        
        '''For real dataset'''
        # allowed = ~M_eval
        draw = (torch.rand(gex.shape, device=gex.device) < pc_train_from_rest/100)
        M_train = draw
        # M_in = M_eval | M_train
        M_in = M_train
        
        fully_hidden = M_in.all(dim=1)
        
        # fix_rows = fully_hidden & allowed.any(dim=1)
        
        fix_rows = fully_hidden
        
        '''We want to retrieve fully_hidden and allowed none'''
        # notify_rows = fully_hidden & ~(allowed.any(dim=1))
        
        
        notify_rows = fully_hidden
        
        if notify_rows.any():
            print("M_Eval covered entire row")
        
        '''This code needs to be verified - at most - we need to fix it for all rows not just single row'''
        if fix_rows.any():
            
            
            rows = fix_rows.nonzero(as_tuple=False).squeeze(1)
            '''We use argmax - meaning highest value - but we will think of making it to random later on'''
            # first_allowed = allowed[rows].float().argmax(dim=1)
            # M_in[rows, first_allowed] = False
            # M_train[rows, first_allowed] = False
        
        # M_in = torch.tensor(M_in, dtype=torch.bool, device=gex.device)
        # M_train = torch.tensor(M_train, dtype=torch.bool, device=gex.device)
        
        # import pdb
        # pdb.set_trace()
        
        '''Think about this part'''
        # assert not (M_train).any()  
        
        '''Check: 
            Technically we can apply M_train and the result should be same as
            gex already applies M_Eval and zero-ed out
        '''
        
        # M_in = M_in.to(device=gex_zero_out.device, dtype=torch.bool)    
        
        # train_gex = gex_zero_out.masked_fill(M_train, 0.0)
        # check_gex = gex_zero_out.masked_fill(M_in, 0.0)
        
        # M_in = M_in.to(device=gex_zero_out.device, dtype=torch.float)
        
        return M_in, M_train
    
def encoding_mask_nonzeros(gex_zero_out, M_eval, pc_train_from_rest=20, eval=False, first_time=False):
    '''
    Introduce mask here.
    gex corresponds to log-transformed input. Nothing else.
    '''
    
    print("mask non zero scheme used")
    
    '''For real dataset'''
    allowed = ~M_eval
    nonzero_mask = (gex_zero_out != 0)
    candidates = nonzero_mask & allowed
    
    candidate_indices = candidates.nonzero(as_tuple=False)
    
    num_to_pick = int(pc_train_from_rest / 100 * candidate_indices.shape[0])
    
    selected_indices = candidate_indices[torch.randperm(candidate_indices.shape[0])[:num_to_pick]]
    
    draw = torch.zeros_like(gex_zero_out, dtype=torch.bool)
    draw[selected_indices[:, 0], selected_indices[:, 1]] = True
    '''This should not change anything'''
    M_train = allowed & draw
    M_in = M_eval | M_train
    
    fully_hidden = M_in.all(dim=1)
    
    fix_rows = fully_hidden & allowed.any(dim=1)
    
    '''We want to retrieve fully_hidden and allowed none'''
    notify_rows = fully_hidden & ~(allowed.any(dim=1))
    
    if notify_rows.any():
        print("M_Eval covered entire row")
    
    '''This code needs to be verified - at most - we need to fix it for all rows not just single row'''
    if fix_rows.any():
        
        
        rows = fix_rows.nonzero(as_tuple=False).squeeze(1)
        '''We use argmax - meaning highest value - but we will think of making it to random later on'''
        first_allowed = allowed[rows].float().argmax(dim=1)
        M_in[rows, first_allowed] = False
        M_train[rows, first_allowed] = False
    
    # M_in = torch.tensor(M_in, dtype=torch.bool, device=gex.device)
    # M_train = torch.tensor(M_train, dtype=torch.bool, device=gex.device)
    
    assert not (M_eval & M_train).any()  
    
    '''Check: 
        Technically we can apply M_train and the result should be same as
        gex already applies M_Eval and zero-ed out
    '''
    
    
    M_in = M_in.to(device=gex_zero_out.device, dtype=torch.bool)
      
    '''I think this is not required?'''
    train_gex = gex_zero_out.masked_fill(M_train, 0.0)
    # check_gex = gex_zero_out.masked_fill(M_in, 0.0)
    
    M_in = M_in.to(device=gex_zero_out.device, dtype=torch.float)
    
    return train_gex, M_in, M_train

    
def prepare_dataset(args, dataset_name, sample_number, pc, split_version=0, data_path='data', cache_path='./preprocessed'):
    spatial = []
    loc = []
    adj = []
    I = []
    gt = []
    ps = []
    msk = []
    bin_val = []
    z_val = []
    ratio = 0.2
    
    print("is_cls: ", args.cls)
    print("is_gepc:", args.gepc)
    print("is_contrast: ", args.contrast)
    print("num_layers: ", args.num_layers)
    print("batch_size: ", args.batch_size)
    print("max_epoch: ", args.max_epoch)
    
    task = args.task
    num_bins = args.num_bins    
    
    if not os.path.exists(cache_path):
        os.makedirs(cache_path, exist_ok=True) 
    if args.task == "imputation":
        adata, masked, masks = preprocess_impute(args, data_path, sample_number, split_version)
        if not isinstance(masked,np.ndarray):
            masked = masked.toarray()
        if not isinstance(adata.X, np.ndarray):
            adata.X = adata.X.toarray()
    elif args.task == "cluster":
        '''This is for clustering task'''
        adata = preprocess_cluster(data_path, dataset_name, sample_number)
    

    
    
    gex = adata.X
    
    spa = adata.obsm["spatial"]
    if args.task == "cluster":
        masks = np.full((gex.shape), False)
        if dataset_name.lower() != 'starmap' and dataset_name.lower() != 'merfish':
            masked = np.where(masks, 0.0, gex.todense())
            gex = gex.toarray().astype(float)
        else:
            masked = np.where(masks, 0.0, gex)
            gex = gex.astype(float) 
                    
     
    '''This will change for cosMX dataset with multiple FOVs'''
    adata.obs['fov'] = 0
    fovlist = adata.obs['fov']
      
    '''Imputation Use'''
    if args.task == "imputation":               
        label = None
    elif args.task == "cluster":
        label = adata.obs['Region'].astype('str')

    '''This is processing spatial data'''
    for fov in tqdm(fovlist.unique()):
        x = pairwise_distances(spa[fovlist==fov], metric='euclidean')
        spatial.append(x)
        loc.append(spa[fovlist==fov])
    
        groundtruth_fov = gex[fov==fovlist]
        maskedgex_fov = masked[fov==fovlist]
        mask_fov = masks[fov==fovlist]
        
        
        '''This is where the code for bin should be'''
        if args.task == "imputation":
            Z, bins = compute_z_and_bins_from_logX_mask(groundtruth_fov, num_bins, exclude_mask=mask_fov)
        if args.task == "cluster":
            Z, bins = compute_z_and_bins_from_logX(groundtruth_fov, num_bins)

        gt.append(groundtruth_fov)
        ps.append(maskedgex_fov)
        msk.append(mask_fov)
        z_val.append(Z)
        bin_val.append(bins)
        
    return gt, ps, adj, spatial, msk, loc, bin_val, z_val, label



def graph_construction(args, ground_truth, masked, similarity_graph, spatial_graph, graph_type, standardscale, mask, pos, bin_val, z_val, label, sample_number, cache_path):
    
    total_sum = 0
    edge_count = 0
    graphs = []
    mask_new = []
    lbl = []
     
    logging.info("Loading preprocessed datasets.")
    for i in tqdm(range(len(ground_truth))):        
    
        '''For now, processed_X and gt is same - for our model'''
        processed_X = torch.from_numpy(masked[i]).float()
        gt = torch.from_numpy(ground_truth[i]).float()
        bin_val = torch.from_numpy(bin_val[i])
        z_val = torch.from_numpy(z_val[i]).float()
        
        spa = spatial_graph[i]
        
        '''Here we adjust, different radius to process spatial neighbors
            decoder_neigh, attn_mask goes in pair
        '''
        
        '''Visium - radius=150, BC - radius=150, Starmap - 450'''
        # decoder_neigh = create_spatial_attn_mask(pos[i], k=1, radius=450, mode='Hybrid')
        # attn_mask = create_spatial_attn_mask(pos[i], k=50, radius=0, mode='KNN')
        '''Setting for clustering and imputation'''
        # decoder_neigh = create_spatial_attn_mask(pos[i], k=1, radius=150, mode='Hybrid')
        # # True - blocked, False - adj
        # attn_mask = create_spatial_attn_mask(pos[i], k=1, radius=150, mode='Hybrid')
        
        '''setting for clustering - StarMap'''
        decoder_neigh = create_spatial_attn_mask(pos[i], k=1, radius=args.spatial_radius, mode='Hybrid')
        attn_mask = create_spatial_attn_mask(pos[i], k=1, radius=args.spatial_radius, mode='Hybrid')
        
        
        '''Oirignally this is what I try'''
        # True - adj, False - not adj
        adj = (~decoder_neigh).float()
        adj_I = np.eye(adj.shape[0])
        adj = adj + adj_I
        '''PCA components'''
        X_pca = build_gene_pca_embeddings(gt, n_components=args.expression_pca_dim)
        # measure_pca(X_pca, adj)
        # measure_pca_quantile(X_pca, adj)
        mask_new.append(torch.from_numpy(mask[i]).bool())
        
        adj_I = np.eye(gt.shape[0])
        
        adj_matrix = (~decoder_neigh).float().detach().numpy()
        graph = dgl.from_scipy(sparse.csr_matrix(adj_matrix))
        graph = dgl.add_self_loop(graph)
               
        coords = torch.from_numpy(pos[i]).float()
        
        graph.ndata["adj"] = adj
        graph.ndata["attn_mask"] = attn_mask
        graph.ndata["feat"] = processed_X
        graph.ndata["input"] = processed_X
        graph.ndata["bin_val"] = bin_val
        graph.ndata["z_val"] = z_val
        graph.ndata['processed_X'] = processed_X
        
        
        
        graph.ndata['gt'] = gt
        graph.ndata["X_pca"] = X_pca
        graph.ndata['pos'] = coords
        
        '''I think this should be deleted - as we do not use it'''
        
        # ''''This is for LPE positional encoding'''
        # if os.path.exists(f'{cache_path}/lap/{sample_number}_{i}.pkl'):
        #     with open(f'{cache_path}/lap/{sample_number}_{i}.pkl', 'rb') as f:
        #         lpe = pickle.load(f)
        # else:
        #     if not os.path.exists(f'{cache_path}/lap'):
        #         os.mkdir(f'{cache_path}/lap')
        #     logging.info(f'Laplacian positional encoding not found for fov {i}. Start calculatng.')
        #     lpe = dgl.laplacian_pe(graph, 10)
        #     with open(f'{cache_path}/lap/{sample_number}_{i}.pkl', 'wb') as f:
        #         pickle.dump(lpe, f)
        # lpe[torch.isnan(lpe)] = 0 
        
        '''For now we do not use it'''
        # graph.ndata['eigvec'] = lpe
        
        num_cells = graph.ndata["feat"].shape[0]
        num_features = graph.ndata["feat"].shape[1]
        num_classes = num_features
        graphs.append(graph)
        
    return graphs, num_cells, num_features, num_classes, mask_new, label


def load_dataset(args, dataset_name, sample_number, pc, split_version=0, standardscale=250, data_path='data', graph_type='homo', cache_path='./preprocessed'):
    gt, masked, adj, spatial, msk, loc, bin_val, z_score, label = prepare_dataset(args, dataset_name, sample_number, pc, split_version, data_path, cache_path)

    return graph_construction(args, gt, masked, adj, spatial, graph_type, standardscale, msk, loc, bin_val, z_score, label, sample_number, cache_path)

def build_gene_pca_embeddings(X_hvg, n_components=20):
    
    pca = PCA(n_components=n_components)
    gene_emb = pca.fit_transform(X_hvg)
    
    return torch.from_numpy(gene_emb).float()

def batch_gene_positive_index(gene_emb_batch, adj_sub):
    '''
    gene_emb_batch:
        pca gene embeddings for current batch pool, shape: [K, D_pca]
    adj_sub:
        Boolean KNN mask between anchor and anchor+ its neighbors, shape: [B, K]
    '''
    
    B, K = adj_sub.shape

    gene_emb_batch = F.normalize(gene_emb_batch, dim=1)

    anchor_gene = gene_emb_batch[:B]          # [B, D]
    candidate_gene = gene_emb_batch           # [K, D]

    gene_sim = anchor_gene @ candidate_gene.T   # [B, K]

    # exclude self for anchors
    self_mask = torch.zeros_like(adj_sub, dtype=torch.bool)
    self_mask[:, :B] = torch.eye(B, dtype=torch.bool, device=adj_sub.device)

    candidate_mask = adj_sub.bool() & ~self_mask
    
    valid = candidate_mask.any(dim=1)

    gene_sim = gene_sim.masked_fill(~candidate_mask, -1e9)

    pos_idx = torch.argmax(gene_sim, dim=1)  # [B], local index inside pool

    return pos_idx, valid


def batch_gene_positive_indices(gene_emb_batch, adj_sub, alpha=5.0, k=3):
    '''
    gene_emb_batch:
        pca gene embeddings for current batch pool, shape: [K, D_pca]
    adj_sub:
        Boolean KNN mask between anchor and anchor+ its neighbors, shape: [B, K]
    '''
    B, K = adj_sub.shape

    gene_emb_batch = F.normalize(gene_emb_batch, dim=1)

    anchor_gene = gene_emb_batch[:B]          # [B, D]
    candidate_gene = gene_emb_batch           # [K, D]

    gene_sim = anchor_gene @ candidate_gene.T   # [B, K]
    
    w = torch.sigmoid(alpha * gene_sim)
    
    # exclude self for anchors
    self_mask = torch.zeros_like(adj_sub, dtype=torch.bool)
    self_mask[:, :B] = torch.eye(B, dtype=torch.bool, device=adj_sub.device)

    candidate_mask = adj_sub.bool() & ~self_mask
    
    valid = candidate_mask.any(dim=1)
    
    gene_sim = gene_sim.masked_fill(~candidate_mask, -1e9)
      
    w = w.masked_fill(~candidate_mask, -1e9)

    topk_vals, topk_idx = torch.topk(gene_sim, k=k, dim=1)

    return topk_idx, valid



def create_activation(name):
    if name == "relu":
        return nn.ReLU()
    elif name == "gelu":
        return nn.GELU()
    elif name == "prelu":
        return nn.PReLU()
    elif name is None:
        return nn.Identity()
    elif name == "elu":
        return nn.ELU()
    else:
        raise NotImplementedError(f"{name} is not implemented.")


def create_norm(name):
    if name == "layernorm":
        return nn.LayerNorm
    elif name == "batchnorm":
        return nn.BatchNorm1d
    elif name == "graphnorm":
        return partial(NormLayer, norm_type="groupnorm")
    else:
        return nn.Identity


def create_optimizer(opt, model, lr, weight_decay, get_num_layer=None, get_layer_scale=None):
    opt_lower = opt.lower()

    parameters = model.parameters()
    opt_args = dict(lr=lr, weight_decay=weight_decay)

    opt_split = opt_lower.split("_")
    opt_lower = opt_split[-1]
    if opt_lower == "adam":
        optimizer = optim.Adam(parameters, **opt_args)
    elif opt_lower == "adamw":
        optimizer = optim.AdamW(parameters, **opt_args)
    elif opt_lower == "adadelta":
        optimizer = optim.Adadelta(parameters, **opt_args)
    elif opt_lower == "radam":
        optimizer = optim.RAdam(parameters, **opt_args)
    elif opt_lower == "sgd":
        opt_args["momentum"] = 0.9
        return optim.SGD(parameters, **opt_args)
    else:
        assert False and "Invalid optimizer"

    return optimizer


# -------------------
def mask_edge(graph, mask_prob):
    E = graph.num_edges()

    mask_rates = torch.FloatTensor(np.ones(E) * mask_prob)
    masks = torch.bernoulli(1 - mask_rates)
    mask_idx = masks.nonzero().squeeze(1)
    return mask_idx


def drop_edge(graph, drop_rate, return_edges=False):
    if drop_rate <= 0:
        return graph

    n_node = graph.num_nodes()
    edge_mask = mask_edge(graph, drop_rate)
    src = graph.edges()[0]
    dst = graph.edges()[1]

    nsrc = src[edge_mask]
    ndst = dst[edge_mask]

    ng = dgl.graph((nsrc, ndst), num_nodes=n_node)
    ng = ng.add_self_loop()

    dsrc = src[~edge_mask]
    ddst = dst[~edge_mask]

    if return_edges:
        return ng, (dsrc, ddst)
    return ng


def load_best_configs(args, path):
    with open(path, "r") as f:
        configs = yaml.load(f, yaml.FullLoader)

    if args.dataset not in configs:
        logging.info("Best args not found")
        return args

    logging.info("Using best configs")
    configs = configs[args.dataset]

    for k, v in configs.items():
        if "lr" in k or "weight_decay" in k:
            v = float(v)
        setattr(args, k, v)
    print("------ Use best configs ------")
    return args


# ------ logging ------

class TBLogger(object):
    def __init__(self, log_path="./logging_data", setting='default', seed=None):
        super(TBLogger, self).__init__()

        if not os.path.exists(log_path):
            os.makedirs(log_path, exist_ok=True)

        self.last_step = 0
        self.log_path = log_path
        self.seed = seed
        name = os.path.join(log_path, setting, f'seed_{seed}')
        self.writer = SummaryWriter(logdir=name)

    def note(self, metrics, step=None):
        if step is None:
            step = self.last_step
        for key, value in metrics.items():
            self.writer.add_scalar(key, value, step)
        self.last_step = step

    def finish(self):
        self.writer.close()

class Logger(object):
    def __init__(self, args):
        super(Logger, self).__init__()

        log_path = args.log_path
        
        if not os.path.exists(log_path):
            os.makedirs(log_path, exist_ok=True)

        self.last_step = 0
        self.log_path = log_path
        self.setting = f'{args.dataset}_{args.noise_pc}pc_{args.encoder}'
        self.seed = args.seed
        self.args = args

    def note(self, metrics_dict):
        metrics_dict['seed'] = self.seed
        metrics_dict['num_hidden'] = self.args.num_hidden
        metrics_dict['objective'] = self.args.objective
        metrics_dict['num_layers'] = self.args.num_layers
        metrics_dict['num_heads'] = self.args.num_heads
        metrics_dict['norm'] = self.args.norm
        metrics_dict['mask_node_rate'] = self.args.mask_node_rate
        metrics_dict['mask_feature_rate'] = self.args.mask_feature_rate
        metrics_dict['drop_node_rate'] = self.args.drop_node_rate
        metrics_dict['latent_dim'] = self.args.latent_dim
        metrics_dict['pe'] = self.args.pe
        metrics_dict['cat_pe'] = self.args.cat_pe
        metrics_dict['lr'] = self.args.lr
        metrics_dict['weight_decay'] = self.args.weight_decay
        
        for k in metrics_dict:
            metrics_dict[k] = [metrics_dict[k]]
        if not os.path.exists(os.path.join(self.log_path, self.setting+'.csv')):
            df = pd.DataFrame(metrics_dict)
        else:
            df = pd.read_csv(os.path.join(self.log_path, self.setting+'.csv'))
            df = pd.concat([df, pd.DataFrame(metrics_dict)])
        df.to_csv(os.path.join(self.log_path, self.setting+'.csv'), index=False)


class NormLayer(nn.Module):
    def __init__(self, hidden_dim, norm_type):
        super().__init__()
        if norm_type == "batchnorm":
            self.norm = nn.BatchNorm1d(hidden_dim)
        elif norm_type == "layernorm":
            self.norm = nn.LayerNorm(hidden_dim)
        elif norm_type == "graphnorm":
            self.norm = norm_type
            self.weight = nn.Parameter(torch.ones(hidden_dim))
            self.bias = nn.Parameter(torch.zeros(hidden_dim))

            self.mean_scale = nn.Parameter(torch.ones(hidden_dim))
        else:
            raise NotImplementedError
        
    def forward(self, graph, x):
        tensor = x
        if self.norm is not None and type(self.norm) != str:
            return self.norm(tensor)
        elif self.norm is None:
            return tensor

        batch_list = graph.batch_num_nodes
        batch_size = len(batch_list)
        batch_list = torch.Tensor(batch_list).long().to(tensor.device)
        batch_index = torch.arange(batch_size).to(tensor.device).repeat_interleave(batch_list)
        batch_index = batch_index.view((-1,) + (1,) * (tensor.dim() - 1)).expand_as(tensor)
        mean = torch.zeros(batch_size, *tensor.shape[1:]).to(tensor.device)
        mean = mean.scatter_add_(0, batch_index, tensor)
        mean = (mean.T / batch_list).T
        mean = mean.repeat_interleave(batch_list, dim=0)

        sub = tensor - mean * self.mean_scale

        std = torch.zeros(batch_size, *tensor.shape[1:]).to(tensor.device)
        std = std.scatter_add_(0, batch_index, sub.pow(2))
        std = ((std.T / batch_list).T + 1e-6).sqrt()
        std = std.repeat_interleave(batch_list, dim=0)
        return self.weight * sub / std + self.bias

'''load different ST dataset'''
def load_DLPFC(root_dir='data', section_id='151507'):  
    ad = sc.read_visium(path=os.path.join(root_dir, str(section_id)), count_file=section_id+'_filtered_feature_bc_matrix.h5', load_images=True)
    ad.var_names_make_unique()  
    Ann_df = pd.read_csv(os.path.join(root_dir, section_id,
            section_id + '_truth.txt'), sep='\t', header=None, index_col=0)
    
    Ann_df.columns = ['Ground Truth']
    ad.obs['Region'] = Ann_df.loc[ad.obs_names, 'Ground Truth'].astype('category')
    keep_bcs = ad.obs.dropna().index
    ad = ad[keep_bcs].copy()    
    return ad

'''BC(Breast Cancer) data - 20 clusters'''
def load_BC(root_dir='data', section_id='section1'):
    
    ad = sc.read_visium(path=os.path.join(root_dir, section_id), count_file=section_id+'_filtered_feature_bc_matrix.h5', load_images=True)
    ad.var_names_make_unique()
    
    gt_dir = os.path.join(root_dir, section_id, 'gt')
    gt_df = pd.read_csv(os.path.join(gt_dir, 'tissue_positions_list_GTs.txt'), sep=',', header=None, index_col=0)
    ad.obs['Region'] = gt_df.loc[:, 6].astype(int)
    ad.obs['Region'] += 1
    keep_bcs = ad.obs.dropna().index
    ad = ad[keep_bcs].copy()
    ad.obs['Region'] = ad.obs['Region'].astype('category')
    return ad

'''STARmap data - 7 clusters'''
def load_mVC(root_dir='data', section_id = 'STARmap_20180505_BY3_1k.h5ad'):
    
    ad = sc.read(os.path.join(root_dir, section_id))
    ad.var_names_make_unique()
    ad.obs.columns = ['Total_counts', 'imagerow', 'imagecol', 'original_clusters']
    ad.obs['original_clusters'] = ad.obs['original_clusters'].astype('category')
    return ad

def load_Merfish(root_dir='data', section_id='MERFISH_0.04.h5ad'):
    
    ad = sc.read(os.path.join(root_dir, section_id))
    ad.var_names_make_unique()
    ad.obs['Region'] = ad.obs['Region'].astype('category')
    
    return ad


def filter_num_calc(data_path, slice_num):
    '''Here we autoset filter_nzr = 0.14'''
    
    if str(slice_num) in ['151507', '151508', '151509', '151510', '151669', '151670', '151671', '151672', '151673', '151674', '151675', '151676', 'MA']:
        adata = load_DLPFC(root_dir=data_path, section_id=slice_num)
    elif str(slice_num) in ['section1']:
        adata = load_BC(root_dir=data_path, section_id=slice_num)
    else:
        print("exception in dataname")
        exit(-1)
    adata.var_names_make_unique()
    
    for temp_count in range(5, 150):
        sc.pp.filter_genes(adata, min_counts=temp_count)
        try:
            X = adata.X.todense()
        except:
            X = adata.X
        if np.count_nonzero(X)/(X.shape[0]*X.shape[1]) > 0.14:
            return temp_count
    return 150

def initialize(data_path, data_name, gene_min_count, preprocessing=True, hvgs = 3000):
    
    
    if data_name in ['151507', '151508', '151509', '151510', '151669', '151670', '151671', '151672', '151673', '151674', '151675', '151676']:
        adata_ = load_DLPFC(root_dir=data_path, section_id=data_name)
        
        if preprocessing:
            # Normalization
            sc.pp.filter_genes(adata_, min_counts=gene_min_count)
            if hvgs != 0:
                sc.pp.highly_variable_genes(adata_, flavor="seurat_v3", n_top_genes=hvgs)
            sc.pp.normalize_total(adata_, target_sum=1e4)
            sc.pp.log1p(adata_)
            if hvgs != 0:
                adata_ = adata_[:, adata_.var['highly_variable']]
        else:
            sc.pp.filter_genes(adata_, min_counts=gene_min_count)
    elif data_name in ['section1']:
        
        adata_ = load_BC(root_dir=data_path, section_id=data_name)
        if preprocessing:
            # Normalization
            sc.pp.filter_genes(adata_, min_counts=gene_min_count)
            if hvgs != 0:
                sc.pp.highly_variable_genes(adata_, flavor="seurat_v3", n_top_genes=hvgs)
            sc.pp.normalize_total(adata_, target_sum=1e4)
            sc.pp.log1p(adata_)
            if hvgs != 0:
                adata_ = adata_[:, adata_.var['highly_variable']]
        else:
            sc.pp.filter_genes(adata_, min_counts=gene_min_count)
            
    elif data_name in ['STARmap_20180505_BY3_1k.h5ad']:
        adata_ = load_mVC(root_dir=data_path, section_id=data_name)
        adata_.obs['Region'] = adata_.obs['original_clusters']
        sc.pp.normalize_total(adata_, target_sum=1e4)
        sc.pp.log1p(adata_)
    elif data_name in ['MERFISH_0.04.h5ad','MERFISH_0.09.h5ad','MERFISH_0.14.h5ad','MERFISH_0.19.h5ad','MERFISH_0.24.h5ad']:
        adata_ = load_Merfish(root_dir=data_path, section_id=data_name)
        sc.pp.normalize_total(adata_, target_sum=1e4)
        sc.pp.log1p(adata_)
    else:
        print("exception in data name")
        exit(-1)
        
    return adata_