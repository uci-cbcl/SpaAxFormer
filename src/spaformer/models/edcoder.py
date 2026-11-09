from typing import Optional
from itertools import chain
from functools import partial

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import math
import pickle


from .spaaxformer_encoder import SpaAxFormerEncoder

from ..evaluation import comprehensive_evaluate
from ..evaluation import comprehensive_evaluate_test
from ..utils import Noise_Cross_Entropy, Contrastive_Loss_Spatial, filter_outer_positives, filter_adjacency_by_similarity, gene_sim_graph, iter_cell_batches, Noise_Cross_Entropy_anchor_only, weighted_spatial_contrastive, adj_to_edge_index, weighted_gene_column_contrastive, build_gene_pca_embeddings, batch_gene_positive_index,batch_gene_positive_indices
from ..utils import objective_metadata

    
def setup_module(m_type, enc_dec, in_dim, d_emb_size, num_hidden, out_dim, num_layers, dropout, activation, norm, nhead, nhead_out, attn_drop, pe=None, negative_slope=0.2, concat_out=True, latent_dim=20, cat_pe=False, pe_aug=False, aggr='SAGE') -> nn.Module:
    if m_type != "spaaxformer":
        raise ValueError("Release supports only spaaxformer")
    mod = SpaAxFormerEncoder(
        d_model = d_emb_size,
        n_heads = nhead,
        num_layers=num_layers,
        dropout = dropout
    )
    return mod

class EntryTokenizer(nn.Module):
    def __init__(
        self,
        num_cells: int,
        num_genes: int,
        num_bins: int,
        d_model: int,
    ):
        
        super().__init__()
        self.num_cells = num_cells
        self.num_genes = num_genes
        self.num_bins = num_bins
        self.d_model = d_model

        # Embeddings
        self.cell_emb = nn.Embedding(num_cells, d_model)
        self.gene_emb = nn.Embedding(num_genes, d_model)

        # +1 for special "masked bin" token at index num_bins
        self.mask_bin_id = num_bins
        self.bin_emb = nn.Embedding(num_bins + 1, d_model)

        # Magnitude (z_ij) -> embedding
        # Simple 2-layer MLP
        self.mag_mlp = nn.Sequential(
            nn.Linear(1, d_model//2),
            nn.GELU(),
            nn.Linear(d_model//2, d_model),
        )
 
    def forward(
    self,
    cell_ids: torch.Tensor,   # [C_batch]
    bin_ids:  torch.Tensor,   # [C_batch, n_genes]
    z_scores: torch.Tensor,   # [C_batch, n_genes]
    mask:     torch.Tensor,   # [C_batch, n_genes], bool (True = masked)
) -> torch.Tensor:
        """
        Returns tokens: [C_batch, n_genes, d_model]
        """ 
        device = bin_ids.device
        C, G = bin_ids.shape
        assert G == self.num_genes
        mask = mask.bool()
        

        # ---- Cell embeddings ----
        cell_emb = self.cell_emb(cell_ids.to(device))        # [C, d_model]
        cell_emb = cell_emb.unsqueeze(1).expand(-1, G, -1)   # [C, G, d_model]

        # ---- Gene embeddings ----
        gene_idx = torch.arange(G, device=device)            # [G]
        gene_emb = self.gene_emb(gene_idx).unsqueeze(0)      # [1, G, d_model]
        gene_emb = gene_emb.expand(C, -1, -1)                # [C, G, d_model]

        # ---- Bin embeddings ----
        bin_ids_used = bin_ids.clone()
        bin_ids_used[mask] = self.mask_bin_id                # masked entries get special bin
        bin_emb = self.bin_emb(bin_ids_used)                 # [C, G, d_model]

        # ---- z-score embeddings ----
        z_emb = self.mag_mlp(z_scores.unsqueeze(-1))         # [C, G, d_model]
        '''If masked -> embedding is [0,...,0]. No contribution'''
        z_emb = torch.where(
            mask.unsqueeze(-1),
            torch.zeros_like(z_emb),
            z_emb,
        )
        tokens = cell_emb + gene_emb + bin_emb + z_emb       # [C, G, d_model]
        iden_emb = cell_emb + gene_emb
        
        return iden_emb, tokens


class GeneDecoder(nn.Module):
    def __init__(
        self, d_model: int, hidden:int = None):
        super().__init__()
        hidden = hidden or (2*d_model)
        
        self.MLP = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.LeakyReLU(),
            nn.Linear(hidden, hidden*2),
            nn.LeakyReLU(),
            nn.Linear(hidden*2, 1),
        )
        
    def forward(self, x):
        """x is the output of the transformer, (batch, seq_len, d_model)"""
        pred_value = self.MLP(x).squeeze(-1)  # (batch, seq_len)
        return pred_value


# class GeneCLSDecoder(nn.Module):
#     def __init__(self, d_model: int, hidden: int = None):
#         super().__init__()
#         hidden = hidden or (2 * d_model)
#         self.mlp = nn.Sequential(
#             nn.Linear(2 * d_model, hidden),
#             nn.GELU(),
#             nn.Linear(hidden, 1),
#         )

#     def forward(self, genes_out: torch.Tensor, cls_out: torch.Tensor) -> torch.Tensor:
#         """
#         genes_out: [B, C, G, D]
#         cls_out:   [B, C, D]
#         returns:   [C, G]
#         """
        
#         # import pdb
#         # pdb.set_trace()
        
#         B, C, G, D = genes_out.shape
#         cls_expand = cls_out.unsqueeze(2).expand(B, C, G, D)        # [B,C,G,D]
#         pair = torch.cat([genes_out, cls_expand], dim=-1)           # [B,C,G,2D]
#         pred = self.mlp(pair).squeeze(-1)                           # [B,C,G]
#         pred = pred.squeeze(0)                                      # [C,G]
#         return pred
    
class CellCLSDecoder(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        
        
        
        self.gene2query = nn.Linear(d_model, d_model)
        # d_out = 30
        # d_model = 128
        # self.gene2query = nn.Linear(d_model, d_out*2)
        # self.W = nn.Linear(d_out*2, d_out, bias=False)
        
        self.query_activation = nn.ReLU()
        self.W = nn.Linear(d_model, d_model, bias=False)
    
    
    def forward(self, iden_emb, cls_e):
        '''
        iden_emb: []
        cell_emb: []
        cell_emb corresponds to cell embedding
        '''
        '''Do I need to detach?'''
        # iden_emb = iden_emb.detach() if self.do_detach else iden_emb
        
        
        query_vecs = self.query_activation(self.gene2query(iden_emb))
        cls_e = cls_e.squeeze(0).unsqueeze(-1) #(batch, embsize, 1)
        # the pred gene expr values, # (batch, seq_len)
        pred_value = torch.bmm(self.W(query_vecs), cls_e).squeeze(-1)
        return pred_value

@DeprecationWarning
class GlobalCLSNegativeCache(nn.Module):
    def __init__(self, num_cells, hidden_dim, num_neg=128, device="cuda"):       
        
        super().__init__()
        self.num_cells = num_cells
        self.hidden_dim = hidden_dim
        self.num_neg = num_neg

        self.register_buffer("cls_cache", torch.zeros(num_cells, hidden_dim, device=device))
        self.register_buffer("cache_valid", torch.zeros(num_cells, dtype=torch.bool, device=device))

    @torch.no_grad()
    def update(self, cls_out, cell_ids, B):
        
        anchor_ids = cell_ids[:B]
        anchor_cls = cls_out[:B]

        self.cls_cache[anchor_ids] = anchor_cls.detach()
        self.cache_valid[anchor_ids] = True

    def ready(self):
        return self.cache_valid.all()

    def sample(self, cell_ids, adj_sub):
        """
        cell_ids: [K]
        adj_sub:  [B, K]

        returns:
            neg_z:   [B, num_neg, D]
            neg_ids: [B, num_neg]
        """
        
        
        device = cell_ids.device
        B, K = adj_sub.shape
        M = self.num_neg

        anchor_ids = cell_ids[:B]

        neg_ids_list = []

        for b in range(B):
            valid = self.cache_valid.clone()

            # exclude anchor itself
            valid[anchor_ids[b]] = False

            # exclude positives
            pos_ids = cell_ids[adj_sub[b].bool()]
            valid[pos_ids] = False

            candidate_ids = torch.where(valid)[0]

            if candidate_ids.numel() < M:
                raise RuntimeError(
                    f"Not enough negatives for anchor {int(anchor_ids[b])}: "
                    f"needed {M}, got {candidate_ids.numel()}."
                )

            perm = torch.randperm(candidate_ids.numel(), device=device)
            neg_ids = candidate_ids[perm[:M]]

            neg_ids_list.append(neg_ids)

        neg_ids = torch.stack(neg_ids_list, dim=0)  # [B, M]
        neg_z = self.cls_cache[neg_ids]             # [B, M, D]

        return neg_z, neg_ids
    
class PreModel(nn.Module):
    def __init__(
            self,
            # Number of cells
            in_cell: int,
            # Number of features
            in_dim: int, 
            num_bins: int,
            num_hidden: int,
            num_layers: int,
            nhead: int,
            nhead_out: int,
            activation: str,
            feat_drop: float,
            attn_drop: float,
            negative_slope: float,
            norm: Optional[str],
            mask_node_rate: float = 0.5,
            mask_feature_rate: float = 0.5,
            encoder_type: str = "gat",
            decoder_type: str = "zinb",
            d_emb_size = 128,        
            loss_fn: str = "sce",
            alpha_l: float = 2,
            concat_hidden: bool = False,
            # latent_dim : int = 70,
            # latent_dim : int = 64,
            latent_dim = 16,
            pe = None,
            drop_node_rate : float = 0.,
            objective : str = 'mask', 
            standardscale : int = 250, 
            cat_pe = False,
            pe_aug = False,
            aggr = 'SAGE',
         ):
        
        
        super(PreModel, self).__init__()
        self._encoder_type = encoder_type
        self._decoder_type = decoder_type
        
        self._d_emb_size = d_emb_size
        self._objective = objective
        
        
        # create cell token, gene token here
        
        '''Here we set r: 20'''
        mask_e = nn.Parameter(torch.randn(self._d_emb_size))        
        self.entry_token = EntryTokenizer(in_cell, in_dim, num_bins, d_emb_size)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, 1, d_emb_size))
        nn.init.normal_(self.cls_token, mean=0.0, std=0.02)
        
        '''This is for multiple heads if needed'''
        assert d_emb_size % nhead == 0
        assert d_emb_size % nhead_out == 0
        if encoder_type in ("gat", "dotgat", 'hgt'):
            enc_num_hidden = d_emb_size // nhead
            enc_nhead = nhead
        elif encoder_type == 'dhgcn':
            enc_num_hidden = d_emb_size
            enc_nhead = nhead
        else:
            enc_num_hidden = d_emb_size
            enc_nhead = nhead
       
        '''Build Encoder'''
        self.encoder = setup_module(
            m_type=encoder_type,
            enc_dec="encoding",
            in_dim=in_dim,
            d_emb_size = d_emb_size, 
            num_hidden = enc_num_hidden,
            out_dim = latent_dim,
            num_layers=num_layers,
            nhead = enc_nhead,
            nhead_out = enc_nhead,
            concat_out=True,
            activation=activation,
            dropout=feat_drop,
            attn_drop=attn_drop,
            negative_slope=negative_slope,
            norm=norm,
            latent_dim = latent_dim,
            pe = pe,
            cat_pe = cat_pe,
            pe_aug = pe_aug,
            aggr = aggr,
        )
            
        self.decoderE = nn.Linear(d_emb_size, 1)
        self.decoderGene = GeneDecoder(d_emb_size, num_hidden)
        self.decoderCell = CellCLSDecoder(d_emb_size)
                
        if objective in ['mask', 'ae']:
            self.encoder_to_decoder = nn.Identity()

    @property
    def output_hidden_dim(self):
        return self._output_hidden_size
    
    def save_fixed_mask(self, masks, mask_name, mask_folder="data/masks"):
        np.save(os.path.join(mask_folder, mask_name), masks)
    
    
    def load_fixed_mask(self, mask_name, mask_folder="data/masks"):     
        mask = np.load(os.path.join(mask_folder,f"{mask_name}.npy"))
        return mask
        
    def laplacian_loss(self, H, adj):
        """
        H: [N, C] - cell embeddings
        adj: [N, N] - adjacency matrix (torch.Tensor or torch.sparse)
        """
        
        mask = torch.eye(adj.size(0), dtype=torch.bool, device=adj.device)
        adj_masked = adj.masked_fill(mask, 0)
        
        D = torch.diag(adj_masked.sum(dim=1))
        L = D - adj_masked  # Unnormalized Laplacian
        return torch.trace(H.T @ L @ H) / H.size(0)  # scalar loss
    
    def gaussian_weighted_laplacian_loss(self, Z, g, sigma=2000.0, k=8):
        """
        Gaussian-weighted Laplacian (spatial smoothness) loss.

        Args:
            Z: [N, C] latent embeddings (torch.Tensor)
            coords: [N, 2] spatial coordinates (torch.Tensor)
            sigma: float, Gaussian bandwidth in the same scale as coords
            k: int, number of nearest neighbors to consider
        Returns:
            scalar torch.Tensor loss
        """
        from sklearn.neighbors import NearestNeighbors
        
        coords = g.ndata["pos"]

        # detach coordinates to avoid autograd overhead
        coords_np = coords.detach().cpu().numpy()
        nbrs = NearestNeighbors(n_neighbors=k + 1, algorithm='auto').fit(coords_np)
        distances, indices = nbrs.kneighbors(coords_np)

        # remove self (distance=0)
        distances = torch.tensor(distances[:, 1:], device=Z.device)
        indices = torch.tensor(indices[:, 1:], device=Z.device, dtype=torch.long)

        # Gaussian weights based on spatial distance
        weights = torch.exp(- (distances ** 2) / (2 * sigma ** 2))  # [N, k]

        # gather neighbor embeddings
        Z_neighbors = Z[indices]  # [N, k, C]

        # compute pairwise diff squared norm
        diff = Z.unsqueeze(1) - Z_neighbors  # [N, k, C]
        diff_sq = (diff ** 2).sum(-1)  # [N, k]

        # weighted mean loss
        loss = (weights * diff_sq).sum() / weights.sum().clamp(min=1e-8)
        return loss
    
    
    def weighted_contrastive_loss(self,Z, g, sigma=2000.0, tau=0.2):
        
        
        coords = g.ndata["pos"]
        adj = g.ndata["adj"]
        
        # cosine similarity
        sim = F.normalize(Z, dim=1) @ F.normalize(Z, dim=1).T
        sim = sim / tau
        sim_exp = torch.exp(sim)

        # compute Gaussian distance weights for positives
        dists = torch.cdist(coords, coords)
        weight = torch.exp(- (dists ** 2) / (2 * sigma ** 2))

        # weighted positive and negative terms
        pos = (sim_exp * adj * weight).sum(dim=1)
        neg = (sim_exp * (~adj)).sum(dim=1)
        ratio = pos / (neg + 1e-8)

        loss = -torch.log(ratio + 1e-8).mean()
        return loss
    
    def spatial_contrastive_loss(
    self, Z, adj, coords=None, sigma=None, tau=0.2, use_weight=True
):
        """
        Improved SpaGIC-style contrastive loss.
        Enforces high similarity among spatial neighbors, low among others.

        Args:
            Z: [N, C] latent embeddings (torch.Tensor)
            adj: [N, N] binary adjacency (1 = neighbor / positive)
            coords: [N, 2] spatial coordinates (optional, for Gaussian weighting)
            sigma: float, Gaussian bandwidth in same unit as coords
            tau: float, temperature for scaling cosine similarity
            use_weight: bool, whether to apply Gaussian distance weights

        Returns:
            scalar torch.Tensor loss
        """
        # 1. Normalize embeddings for cosine similarity
        Z = F.normalize(Z, dim=1)

        # 2. Compute cosine similarity matrix with temperature
        sim = torch.matmul(Z, Z.T) / tau
        sim_exp = torch.exp(sim)

        # 3. Build masks
        N = adj.shape[0]
        mask_self = torch.eye(N, dtype=torch.bool, device=adj.device)
        adj = adj.bool() & (~mask_self)        # positives only
        neg_mask = (~adj) & (~mask_self)       # negatives (no self)

        # 4. Optional Gaussian weighting (aligns with spatial bias)
        if use_weight and coords is not None and sigma is not None:
            dists = torch.cdist(coords, coords, p=2)
            weight = torch.exp(- (dists ** 2) / (2 * sigma ** 2))
            weight = weight.to(Z.device)
        else:
            weight = torch.ones_like(adj, dtype=torch.float, device=Z.device)

        # 5. Weighted positive and negative sums
        pos = (sim_exp * adj * weight).sum(dim=1)
        neg = (sim_exp * neg_mask).sum(dim=1)

        # 6. Ratio and safe log
        ratio = pos / (neg + 1e-8)
        loss = -torch.log(ratio + 1e-8).mean()

        return loss
    
    def orthogonal_loss(self, z):
        z = F.normalize(z, dim=1)
        G = torch.matmul(z.T, z) / z.size(0)  # [D x D]
        I = torch.eye(G.size(0), device=z.device)
        return ((G - I) ** 2).sum()
    
    def batch_pca_knn_InfoNCE(self, cls_out, X_pca, adj_sub, tau=1):
        
        B, K = adj_sub.shape

        pos_idx, valid = batch_gene_positive_index(X_pca, adj_sub)
        
        if valid.sum() == 0:
            return torch.tensor(0.0, device=device, dtype=cls_out.dtype)

        
        cls_out = F.normalize(cls_out, dim=1)

        anchor_cls = cls_out[:B]          # [B, D]
        candidate_cls = cls_out           # [K, D]

        logits = anchor_cls @ candidate_cls.T  # [B, K]
        logits = logits / tau

        # exclude anchor self-comparison
        # self_mask = torch.zeros((B, K), dtype=torch.bool, device=cls_out.device)
        # self_mask[:, :B] = torch.eye(B, dtype=torch.bool, device=cls_out.device)
        
        self_mask = adj_sub.clone()
        self_mask[torch.arange(B), pos_idx] = False

        logits = logits.masked_fill(self_mask, -1e9)

        loss = F.cross_entropy(logits, pos_idx.to(cls_out.device))

        return loss 
    
    def weighted_spatial_gene_sim_contrastive(
        self,
        emb_pool,        # [K, D] cell embeddings (base+neighbors+optional others)
        X_pca_batch,
        cell_ids_cpu,
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
        anchor_z = z[:B]                                      # [B, D]
        logits = (anchor_z @ z.T) / tau                       # [B, K]
        
        gene_sim_batch = F.normalize(X_pca_batch, dim=1)
        gene_sim_anc = gene_sim_batch[:B]
        
        gene_sim = gene_sim_anc @ gene_sim_batch.T
        
        # masks
        pos_mask = adj_sub[:B, :].bool()                 # [B, K]
        self_mask = torch.zeros((B, K), dtype=torch.bool, device=device)
        self_mask[torch.arange(B), torch.arange(B)] = True

        pos_mask = pos_mask & (~self_mask)
        denom_mask = ~self_mask

        valid = pos_mask.any(dim=1)
        if valid.sum() == 0:
            return torch.tensor(0.0, device=device, dtype=emb_pool.dtype)
           
        w = torch.sigmoid(alpha * gene_sim)

        # only weight positives; non-positives get weight 0 in numerator
        w = w * pos_mask.float()

        # stable log numerator: log sum_j w_ij * exp(logits_ij)
        # compute as logsumexp(logits + log(w)) over positives
        neg_inf = torch.finfo(logits.dtype).min
        logw = torch.log(w + eps)                        # [B, K]
        num_terms = (logits + logw).masked_fill(~pos_mask, neg_inf)
        log_num = torch.logsumexp(num_terms[valid], dim=1)
        
        '''Using negative cache'''
        # if self.negative_cache.ready():
        #     # print("cache entered")
        #     # import pdb
        #     # pdb.set_trace()
            
        #     neg_z, neg_ids = self.negative_cache.sample(cell_ids_cpu, adj_sub)
            
        #     neg_z = neg_z.to(device)
        #     neg_z = F.normalize(neg_z, dim=2)
            
        #     neg_logits = torch.einsum("bd, bmd->bm", anchor_z, neg_z) / tau
            
        #     pos_den_terms = logits.masked_fill(~pos_mask, neg_inf)    
        #     den_terms = torch.cat([pos_den_terms[valid], neg_logits[valid]], dim=1)
            
        #     log_den = torch.logsumexp(den_terms, dim=1)
        # else:
        #     return 0.0
                   
        # loss = - (log_num - log_den).mean()
        # return loss

        
        '''This is for denominator in a batch'''    
        # denominator over all non-self
        den_terms = logits.masked_fill(~denom_mask, neg_inf)
        log_den = torch.logsumexp(den_terms[valid], dim=1)

        loss = -(log_num - log_den).mean()
        return loss
    
    
    def weighted_spatial_gene_sim_contrastive_neg_not_knn(
    self,
    emb_pool,        # [K, D] cell embeddings (base+neighbors+optional others)
    X_pca_batch,
    cell_ids_cpu,
    C_dst,           # int, number of anchors (first C_dst in pool)
    adj_sub,         # [B, K] bool adjacency on pool
    adj_filtered,
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
        anchor_z = z[:B]                                      # [B, D]
        logits = (anchor_z @ z.T) / tau                       # [B, K]
        
        gene_sim_batch = F.normalize(X_pca_batch, dim=1)
        gene_sim_anc = gene_sim_batch[:B]
        
        gene_sim = gene_sim_anc @ gene_sim_batch.T
        
        # masks
        pos_candidate_mask = adj_sub.bool()
        pos_mask = adj_filtered.bool()                 # [B, K]
        self_mask = torch.zeros((B, K), dtype=torch.bool, device=device)
        self_mask[torch.arange(B), torch.arange(B)] = True
        

        pos_mask = pos_mask & (~self_mask)
        neg_mask = (~pos_candidate_mask) & (~self_mask) 
        
        denom_mask = pos_mask | neg_mask

        valid = pos_mask.any(dim=1)
        if valid.sum() == 0:
            return torch.tensor(0.0, device=device, dtype=emb_pool.dtype)
        
        w = torch.sigmoid(alpha * gene_sim)

        # only weight positives; non-positives get weight 0 in numerator
        w = w * pos_mask.float()

        # stable log numerator: log sum_j w_ij * exp(logits_ij)
        # compute as logsumexp(logits + log(w)) over positives
        neg_inf = torch.finfo(logits.dtype).min
        logw = torch.log(w + eps)                        # [B, K]
        num_terms = (logits + logw).masked_fill(~pos_mask, neg_inf)
        log_num = torch.logsumexp(num_terms[valid], dim=1)
        den_terms = logits.masked_fill(~denom_mask, neg_inf)
        log_den = torch.logsumexp(den_terms[valid], dim=1)

        loss = -(log_num - log_den).mean()
        return loss
    
    
    
    def weighted_spatial_gene_sim_contrastive_neg_no_knn_thres(
        self,
        emb_pool,        # [K, D] cell embeddings (base+neighbors+optional others)
        X_pca_batch,
        cell_ids_cpu,
        C_dst,           # int, number of anchors (first C_dst in pool)
        adj_sub,         # [B, K] bool adjacency on pool
        adj_filtered,
        epoch,
        tau=1.0,
        alpha=5.0,       # weight sharpness (bigger => closer to hard gating)
        negative_pca_threshold=0,
        eps=1e-12,
        ):
        
        device = emb_pool.device
        K, D = emb_pool.shape
        B = C_dst

        z = F.normalize(emb_pool, dim=-1, eps=1e-8)      # [K, D]
        anchor_z = z[:B]                                      # [B, D]
        logits = (anchor_z @ z.T) / tau                       # [B, K]
        
        
        gene_sim_batch = F.normalize(X_pca_batch, dim=1)
        gene_sim_anc = gene_sim_batch[:B]
        
        gene_sim = gene_sim_anc @ gene_sim_batch.T
        
        # masks
        pos_candidate_mask = adj_sub.bool()
        pos_mask = adj_filtered.bool()                 # [B, K]
        self_mask = torch.zeros((B, K), dtype=torch.bool, device=device)
        self_mask[torch.arange(B), torch.arange(B)] = True
        

        pos_mask = pos_mask & (~self_mask)
        neg_mask = (~pos_candidate_mask) & (~self_mask) & (gene_sim < negative_pca_threshold)
        
        denom_mask = pos_mask | neg_mask

        valid = pos_mask.any(dim=1) & neg_mask.any(dim=1)
        if not valid.any():
            return torch.tensor(0.0, device=device, dtype=emb_pool.dtype)
        
        w = torch.sigmoid(alpha * gene_sim)

        # only weight positives; non-positives get weight 0 in numerator
        w = w * pos_mask.float()

        # stable log numerator: log sum_j w_ij * exp(logits_ij)
        # compute as logsumexp(logits + log(w)) over positives
        neg_inf = torch.finfo(logits.dtype).min
        logw = torch.log(w + eps)                        # [B, K]
        num_terms = (logits + logw).masked_fill(~pos_mask, neg_inf)
        log_num = torch.logsumexp(num_terms[valid], dim=1)
        den_terms = logits.masked_fill(~denom_mask, neg_inf)
        log_den = torch.logsumexp(den_terms[valid], dim=1)

        loss = -(log_num - log_den).mean()
        return loss
    
    def weighted_spatial_gene_sim_contrastive_pca_thres_neg(
        self,
        emb_pool,        # [K, D]
        X_pca_batch,     # [K, P]
        cell_ids_cpu,
        C_dst,           # number of anchors; anchors are first C_dst cells
        adj_sub,         # [B, K] bool adjacency
        epoch,
        tau=1.0,
        alpha=5.0,
        negative_pca_threshold=0,
        eps=1e-12,
    ):
        
        """
        Positives:
            Spatial neighbors, weighted by PCA cosine similarity.

        Negatives:
            Non-positive cells whose PCA cosine similarity to the anchor is
            < negative_pca_threshold.

        Other cells:
            Ignored in the denominator.
        """
        device = emb_pool.device
        K, _ = emb_pool.shape
        B = C_dst
        
        
        # ------------------------------------------------------------------
        # CLS cosine similarities
        # ------------------------------------------------------------------
        z = F.normalize(
            emb_pool,
            p=2,
            dim=-1,
            eps=1e-8,
        )

        anchor_z = z[:B]
        logits = (anchor_z @ z.T) / tau                    # [B, K]

        # ------------------------------------------------------------------
        # PCA cosine similarities
        # ------------------------------------------------------------------
        X_pca_batch = X_pca_batch.to(device)

        gene_z = F.normalize(
            X_pca_batch,
            p=2,
            dim=-1,
            eps=1e-8,
        )

        gene_sim = gene_z[:B] @ gene_z.T                   # [B, K]

        # ------------------------------------------------------------------
        # Masks
        # ------------------------------------------------------------------
        self_mask = torch.zeros(
            (B, K),
            dtype=torch.bool,
            device=device,
        )
        self_mask[
            torch.arange(B, device=device),
            torch.arange(B, device=device),
        ] = True

        # Spatial neighbors are positives.
        pos_mask = adj_sub[:B, :].to(device=device, dtype=torch.bool)
        pos_mask = pos_mask & ~self_mask

        # A negative must:
        #   1. not be the anchor itself,
        #   2. not already be a positive,
        #   3. have sufficiently low PCA cosine similarity.
        neg_mask = (
            (~self_mask)
            & (~pos_mask)
            & (gene_sim < negative_pca_threshold)
        )

        # InfoNCE denominator contains selected positives and selected negatives.
        denom_mask = pos_mask | neg_mask

        # An anchor must have at least one positive.
        # Optionally, also require at least one selected negative.
        valid = pos_mask.any(dim=1) & neg_mask.any(dim=1)

        if not valid.any():
            return emb_pool.sum() * 0.0

        # ------------------------------------------------------------------
        # PCA-based positive weights
        # ------------------------------------------------------------------
        positive_weights = torch.sigmoid(alpha * gene_sim)
        positive_weights = positive_weights * pos_mask.to(logits.dtype)

        neg_inf = torch.finfo(logits.dtype).min

        # Weighted multi-positive numerator:
        # log sum_{j in P(i)} w_ij exp(logit_ij)
        log_weights = torch.log(positive_weights.clamp_min(eps))

        numerator_terms = (logits + log_weights).masked_fill(
            ~pos_mask,
            neg_inf,
        )

        log_num = torch.logsumexp(
            numerator_terms[valid],
            dim=1,
        )

        # Denominator:
        # positives + only PCA-dissimilar negatives
        denominator_terms = logits.masked_fill(
            ~denom_mask,
            neg_inf,
        )

        log_den = torch.logsumexp(
            denominator_terms[valid],
            dim=1,
        )

        loss = -(log_num - log_den).mean()
        return loss
    

    def batch_supcon_loss(
        self,
        cls_out,        # [K, D]
        adj_sub,        # [B, K] bool, True = positive
        tau=1.0,
        eps=1e-12,
        ):
        
        device = cls_out.device
        B, K = adj_sub.shape

        z = F.normalize(cls_out, dim=1, eps=1e-8)

        anchor_z = z[:B]          # [B, D]
        candidate_z = z           # [K, D]

        logits = anchor_z @ candidate_z.T   # [B, K]
        logits = logits / tau

        # self mask: anchor i should not compare to itself
        self_mask = torch.zeros((B, K), dtype=torch.bool, device=device)
        self_mask[torch.arange(B, device=device), torch.arange(B, device=device)] = True

        # positives = KNN positives, excluding self
        pos_mask = adj_sub.bool().to(device) & ~self_mask

        # denominator = all non-self candidates
        denom_mask = ~self_mask

        # valid anchors must have at least one positive
        valid = pos_mask.any(dim=1)

        if valid.sum() == 0:
            return torch.tensor(0.0, device=device, dtype=cls_out.dtype)

        neg_inf = torch.finfo(logits.dtype).min

        # log denominator
        logits_denom = logits.masked_fill(~denom_mask, neg_inf)
        log_den = torch.logsumexp(logits_denom, dim=1)  # [B]

        # log probability for every candidate
        log_prob = logits - log_den[:, None]            # [B, K]

        # SupCon: average log-prob over positives for each anchor
        pos_log_prob = (log_prob * pos_mask.float()).sum(dim=1)
        num_pos = pos_mask.float().sum(dim=1)

        loss_per_anchor = -pos_log_prob / (num_pos + eps)

        loss = loss_per_anchor[valid].mean()

        return loss
    
    def batch_supcon_gene_sim_loss(
        self,
        cls_out,        # [K, D]
        X_pca_batch,
        adj_sub,        # [B, K] bool, True = positive
        tau=1.0,
        eps=1e-12,
        ):
        
        device = cls_out.device
        B, K = adj_sub.shape

        z = F.normalize(cls_out, dim=1, eps=1e-8)
        
        pos_indices, valid = batch_gene_positive_indices(X_pca_batch, adj_sub)
        
        if valid.sum() == 0:
            return torch.tensor(0.0, device=device, dtype=cls_out.dtype)

        anchor_z = z[:B]          # [B, D]
        candidate_z = z           # [K, D]

        logits = anchor_z @ candidate_z.T   # [B, K]
        logits = logits / tau
        
        # self mask: anchor i should not compare to itself
        self_mask = torch.zeros((B, K), dtype=torch.bool, device=device)
        self_mask[torch.arange(B, device=device), torch.arange(B, device=device)] = True
        
        
        top_k_mask = torch.zeros((B, K), dtype=torch.bool, device=device)
        top_k_mask.scatter_(1, pos_indices, True)
        # positives = KNN positives, excluding self
        pos_mask = adj_sub.bool().to(device) & ~self_mask
        pos_mask = pos_mask & top_k_mask

        # denominator = all non-self candidates
        denom_mask = ~(self_mask | top_k_mask)
        neg_inf = torch.finfo(logits.dtype).min

        # log denominator
        logits_denom = logits.masked_fill(~denom_mask, neg_inf)
        log_den = torch.logsumexp(logits_denom, dim=1)  # [B]

        # log probability for every candidate
        log_prob = logits - log_den[:, None]            # [B, K]

        # SupCon: average log-prob over positives for each anchor
        pos_log_prob = (log_prob * pos_mask.float()).sum(dim=1)
        num_pos = pos_mask.float().sum(dim=1)

        loss_per_anchor = -pos_log_prob / (num_pos + eps)

        loss = loss_per_anchor[valid].mean()

        return loss
    
    def batch_supcon_loss_pca_negatives(
        self,
        cls_out,                # [K, D] CLS embeddings
        X_pca_batch,            # [K, P] PCA embeddings, same cell order as cls_out
        adj_sub,                # [B, K] bool, True = KNN positive
        tau=1.0,
        negative_pca_threshold=0.0,
        require_negative=True,
        eps=1e-12,
    ):
        """
        Positives:
            KNN/spatial neighbors specified by adj_sub.

        Negatives:
            Non-positive candidates with PCA cosine similarity
            < negative_pca_threshold.

            Negatives are not required to be non-KNN separately. However,
            positive membership takes precedence, so a pair cannot be both
            positive and negative.

        Ignored:
            Non-positive candidates whose PCA cosine similarity is
            >= negative_pca_threshold.

        Mask convention:
            adj_sub[i, j] == True means candidate j is a positive for anchor i.
        """
        device = cls_out.device
        B, K = adj_sub.shape

        if cls_out.ndim != 2:
            raise ValueError(
                f"cls_out must have shape [K, D], got {tuple(cls_out.shape)}."
            )

        if X_pca_batch.ndim != 2:
            raise ValueError(
                "X_pca_batch must have shape [K, P]."
            )

        if cls_out.shape[0] != K:
            raise ValueError(
                f"adj_sub has K={K} candidates, but cls_out has "
                f"{cls_out.shape[0]} rows."
            )

        if X_pca_batch.shape[0] != K:
            raise ValueError(
                f"adj_sub has K={K} candidates, but X_pca_batch has "
                f"{X_pca_batch.shape[0]} rows."
            )

        # ------------------------------------------------------------
        # CLS cosine-similarity logits
        # ------------------------------------------------------------
        z = F.normalize(
            cls_out,
            p=2,
            dim=1,
            eps=1e-8,
        )

        anchor_z = z[:B]                              # [B, D]
        logits = (anchor_z @ z.T) / tau              # [B, K]

        # ------------------------------------------------------------
        # PCA cosine similarities
        # ------------------------------------------------------------
        pca_z = F.normalize(
            X_pca_batch.to(device=device).float(),
            p=2,
            dim=1,
            eps=1e-8,
        )

        anchor_pca = pca_z[:B]                        # [B, P]
        pca_similarity = anchor_pca @ pca_z.T         # [B, K]

        # ------------------------------------------------------------
        # Masks
        # ------------------------------------------------------------
        self_mask = torch.zeros(
            (B, K),
            dtype=torch.bool,
            device=device,
        )

        anchor_indices = torch.arange(B, device=device)
        self_mask[anchor_indices, anchor_indices] = True

        # Positives remain the original adjacency/KNN positives.
        pos_mask = (
            adj_sub.to(device=device, dtype=torch.bool)
            & ~self_mask
        )

        # PCA-negative candidates.
        # ~pos_mask gives positive pairs precedence. Therefore, a spatial
        # positive with PCA similarity < 0 remains positive rather than being
        # assigned to both sets.
        neg_mask = (
            ~self_mask
            & ~pos_mask
            & (pca_similarity < negative_pca_threshold)
        )

        # Only selected positives and PCA-negative candidates participate
        # in the denominator.
        denom_mask = pos_mask | neg_mask

        has_positive = pos_mask.any(dim=1)
        has_negative = neg_mask.any(dim=1)

        if require_negative:
            valid = has_positive & has_negative
        else:
            valid = has_positive

        if not valid.any():
            # Return zero while preserving connection to the computation graph.
            return cls_out.sum() * 0.0

        # ------------------------------------------------------------
        # SupCon denominator
        # ------------------------------------------------------------
        neg_inf = torch.finfo(logits.dtype).min

        denominator_logits = logits.masked_fill(
            ~denom_mask,
            neg_inf,
        )

        log_denominator = torch.logsumexp(
            denominator_logits,
            dim=1,
        )                                               # [B]

        # Log probability for each selected candidate.
        log_probability = logits - log_denominator[:, None]

        # ------------------------------------------------------------
        # Average log probability over all positives per anchor
        # ------------------------------------------------------------
        positive_log_probability = (
            log_probability.masked_fill(~pos_mask, 0.0)
        ).sum(dim=1)

        number_of_positives = pos_mask.sum(dim=1).to(
            dtype=logits.dtype
        )

        loss_per_anchor = -positive_log_probability / (
            number_of_positives + eps
        )

        return loss_per_anchor[valid].mean()
    
    
    def batch_supcon_loss_pca_filter_negatives_thres(
        self,
        cls_out,                # [K, D] CLS embeddings
        X_pca_batch,            # [K, P] PCA embeddings, same cell order as cls_out
        adj_sub,                # [B, K] bool, True = KNN positive
        adj_filtered,
        tau=1.0,
        negative_pca_threshold=0.0,
        require_negative=True,
        eps=1e-12,
    ):
        """
        Positives:
            KNN/spatial neighbors specified by adj_sub.

        Negatives:
            Non-positive candidates with PCA cosine similarity
            < negative_pca_threshold.

            Negatives are not required to be non-KNN separately. However,
            positive membership takes precedence, so a pair cannot be both
            positive and negative.

        Ignored:
            Non-positive candidates whose PCA cosine similarity is
            >= negative_pca_threshold.

        Mask convention:
            adj_sub[i, j] == True means candidate j is a positive for anchor i.
        """
        device = cls_out.device
        B, K = adj_sub.shape

        if cls_out.ndim != 2:
            raise ValueError(
                f"cls_out must have shape [K, D], got {tuple(cls_out.shape)}."
            )

        if X_pca_batch.ndim != 2:
            raise ValueError(
                "X_pca_batch must have shape [K, P]."
            )

        if cls_out.shape[0] != K:
            raise ValueError(
                f"adj_sub has K={K} candidates, but cls_out has "
                f"{cls_out.shape[0]} rows."
            )

        if X_pca_batch.shape[0] != K:
            raise ValueError(
                f"adj_sub has K={K} candidates, but X_pca_batch has "
                f"{X_pca_batch.shape[0]} rows."
            )

        # ------------------------------------------------------------
        # CLS cosine-similarity logits
        # ------------------------------------------------------------
        z = F.normalize(
            cls_out,
            p=2,
            dim=1,
            eps=1e-8,
        )

        anchor_z = z[:B]                              # [B, D]
        logits = (anchor_z @ z.T) / tau              # [B, K]

        # ------------------------------------------------------------
        # PCA cosine similarities
        # ------------------------------------------------------------
        pca_z = F.normalize(
            X_pca_batch.to(device=device).float(),
            p=2,
            dim=1,
            eps=1e-8,
        )

        anchor_pca = pca_z[:B]                        # [B, P]
        pca_similarity = anchor_pca @ pca_z.T         # [B, K]

        # ------------------------------------------------------------
        # Masks
        # ------------------------------------------------------------
        self_mask = torch.zeros(
            (B, K),
            dtype=torch.bool,
            device=device,
        )

        anchor_indices = torch.arange(B, device=device)
        self_mask[anchor_indices, anchor_indices] = True
        
        adj_sub = adj_sub.to(device=device, dtype=torch.bool)
        adj_filtered = adj_filtered.to(device=device, dtype=torch.bool)

        # Positives remain the original adjacency/KNN positives.
        pos_mask = (
            adj_filtered.to(device=device, dtype=torch.bool)
            & ~self_mask
        )

        # PCA-negative candidates.
        # ~pos_mask gives positive pairs precedence. Therefore, a spatial
        # positive with PCA similarity < 0 remains positive rather than being
        # assigned to both sets.
        neg_mask = (
            ~self_mask
            & ~adj_sub
            & (pca_similarity < negative_pca_threshold)
        )

        # Only selected positives and PCA-negative candidates participate
        # in the denominator.
        denom_mask = pos_mask | neg_mask

        has_positive = pos_mask.any(dim=1)
        has_negative = neg_mask.any(dim=1)

        if require_negative:
            valid = has_positive & has_negative
        else:
            valid = has_positive

        if not valid.any():
            # Return zero while preserving connection to the computation graph.
            return cls_out.sum() * 0.0

        # ------------------------------------------------------------
        # SupCon denominator
        # ------------------------------------------------------------
        neg_inf = torch.finfo(logits.dtype).min

        denominator_logits = logits.masked_fill(
            ~denom_mask,
            neg_inf,
        )

        log_denominator = torch.logsumexp(
            denominator_logits,
            dim=1,
        )                                               # [B]

        # Log probability for each selected candidate.
        log_probability = logits - log_denominator[:, None]

        # ------------------------------------------------------------
        # Average log probability over all positives per anchor
        # ------------------------------------------------------------
        positive_log_probability = (
            log_probability.masked_fill(~pos_mask, 0.0)
        ).sum(dim=1)

        number_of_positives = pos_mask.sum(dim=1).to(
            dtype=logits.dtype
        )

        loss_per_anchor = -positive_log_probability / (
            number_of_positives + eps
        )

        return loss_per_anchor[valid].mean()
    
    
    @torch.no_grad()
    def update_negative_cache(self, cache_payload):
        self.negative_cache.update(
            cls_out=cache_payload["cls_out"],
            cell_ids=cache_payload["cell_ids"],
            B=cache_payload["B"],)

    def forward(self, args, g, X, X_pca, X_bins, z_val, mask_in, mask_train, cell_ids_cpu, target_idx, epoch, count): 
        requires_cls = args.encoder == "spaaxformer"

        if requires_cls and not args.cls:
            raise ValueError(
            "spaaxformer requires CLS for the retained-CLS configuration. "
            "For GEP-only, remove --no_cls and use --no_gepc without --contrast.")    
        
        is_cls = args.cls
        is_contrast = args.contrast
        is_gepc = args.gepc
        
        '''This is where we set the training objective'''
        
        # pc = 50
        
        mse1 = nn.MSELoss()
        mse2 = nn.MSELoss()
        self.BCE_loss = nn.BCEWithLogitsLoss()
                
        adj = g.ndata["adj"]
        pos_mask = g.ndata["attn_mask"]
        # adj_pca_filtered = g.ndata["adj_filtered"]
        
        
        bin_batch   = X_bins[cell_ids_cpu]       # [C_batch, G]
        z_batch     = z_val[cell_ids_cpu]        # [C_batch, G]
        X_batch     = X[target_idx]            # [C_batch, G]
        '''This will change'''
        
        X_pca_batch = X_pca[cell_ids_cpu]
        
        
        '''Few entries are masked'''
        mask_in_batch  = mask_in[cell_ids_cpu]    # [C_batch, G], bool
        mask_train_batch = mask_train[target_idx]
        if not mask_train_batch.any():
            return None, {}

        
        '''only ones in the batch are available'''
        # cell_attn_mask_batch = pos_mask[cell_ids_cpu][:, cell_ids_cpu]
        
        # start from here
        
        
        A_sub = adj[cell_ids_cpu][:, cell_ids_cpu].bool()  # [C_batch, C_batch]
        # A_sub_pca_filtered = adj_pca_filtered[cell_ids_cpu][:, cell_ids_cpu].bool()
        
        A_sub.fill_diagonal_(True)
        # A_sub_pca_filtered.fill_diagonal_(True)
        
        cell_attn_mask_batch = ~A_sub   # True = block attention
        # cell_attn_mask_batch_pca_filter = ~A_sub_pca_filtered
        
        '''This pertains to SpaAxFormerEncoder module'''
        A_hat = adj[target_idx][:, cell_ids_cpu].float()
        deg = A_hat.sum(dim=1, keepdim=True).clamp_min(1e-8)
        A_hat = A_hat / deg
        
                
        entry_iden_emb, tokens = self.entry_token(cell_ids_cpu, bin_batch, z_batch, mask_in_batch)    # C, G, D_model
        B = 1   
        C,G,D = tokens.shape
        C_dst = target_idx.numel()
        
        A_sub_pre_edge = A_sub[:C_dst, :]
        '''We let diagonal true for now'''
        edge_idx = adj_to_edge_index(A_sub_pre_edge)
        
        
        
        '''There has been change here'''
        # cls = self.cls_token.expand(B, C, 1, D) + cell_e
        
        if is_cls:
            cls = self.cls_token.expand(B, C, 1, D)
            x = torch.cat([cls, tokens.view(B, C, G, D)], dim=2)  # B, C, G+1, D_model
        else:
            '''non CLS'''
            x = tokens.view(B,C,G,D)
        
        if requires_cls:
            expected_shape = (B, C, G + 1, D)
            if tuple(x.shape) != expected_shape:
                raise ValueError(
                    f"Expected retained-CLS input {expected_shape}, got {tuple(x.shape)}")
        
        x = self.encoder(x, cell_attn_mask_batch, target_idx, A_hat, epoch, edge_idx, count)                                 # [1,C,G+1,D]
        
        if requires_cls and tuple(x.shape) != expected_shape:
            raise ValueError(
        f"Expected encoder output {expected_shape}, got {tuple(x.shape)}")
        
        
        if not is_cls:
            # x = x.squeeze(0)                                                     # [C,G+1,D]
            '''This is without cls'''
            genes_out = x
        else:     
            cls_out = x[:, :, 0, :]                                               # [C,D]
            # Genes output for reconstruction (exclude CLS)
            genes_out = x[:, :, 1:, :]                                          # [C,G,D]
       
        X_pred = self.decoderGene(genes_out[:, :C_dst, :, :]).squeeze(-1).squeeze(0) 
        X_pred_batch = X_pred  
        
        '''Together but unnecessary'''
        # X_pred = self.decoderGene(genes_out).squeeze(-1).squeeze(0)  
        # X_pred_batch = X_pred[:C_dst, :]
        if is_gepc:
            # X_pred = self.decoderCLS(genes_out, cls_out)
            
            '''Together'''
            # X_pred_cell = self.decoderCell(entry_iden_emb, cls_out)
            # X_pred_cell_batch = X_pred_cell[:C_dst, :]
            
            X_pred_cell = self.decoderCell(entry_iden_emb[:C_dst, :, :], cls_out[:, :C_dst, :])
            X_pred_cell_batch = X_pred_cell
        
        '''target(base cell) ground truth'''
        masked_batch_targets = X_batch[mask_train_batch]
        masked_batch_preds = X_pred_batch[mask_train_batch]
        
        loss = F.mse_loss(masked_batch_preds, masked_batch_targets)
        cell_loss = 0
        contrastive_loss = 0
        cache_payload = {}
        
        if is_gepc:
            masked_batch_cls_preds = X_pred_cell_batch[mask_train_batch]
            cell_loss = F.mse_loss(masked_batch_cls_preds, masked_batch_targets)
        if is_contrast:
            
            pos_candidate_mask = A_sub[:C_dst, :]
            contrastive_loss = self.weighted_spatial_gene_sim_contrastive(cls_out.squeeze(0), X_pca_batch, cell_ids_cpu, C_dst, pos_candidate_mask, epoch)
            cache_payload = {"cls_out":cls_out.squeeze(0).detach(),"cell_ids": cell_ids_cpu.detach(), "B": C_dst}
              
        '''our default'''
        # return 10*(0.5*loss + 0.5*cell_loss) + 0.02 * contrastive_loss, cache_payload
        '''StarMap'''
        # return 10*(0.5*loss + 0.5*cell_loss) + 2.5 * contrastive_loss, cache_payload
         
        '''StarMap Used'''
        return 10*(0.5*loss + 0.5*cell_loss) + args.contrastive_weight * contrastive_loss, cache_payload 

    def batch_test(self, args, graphs, eval_mask, lbl,logger=None, slice=None, slice_num=None, save_pt_name=None):
        with torch.no_grad():
            self.eval()
                    
            device = eval_mask.device
            batch_size = args.batch_size
            mode = args.mode
            task = args.task
            eval_mask = eval_mask.bool()
            emb_size = self._d_emb_size 
            is_cls = args.cls
            is_gepc = args.gepc
            
            requires_cls = args.encoder == "spaaxformer"

            if requires_cls and not is_cls:
                raise ValueError(
                    "spaaxformer requires CLS during inference. "
                    "Remove --no_cls. For GEP-only, use --no_gepc "
                    "without --contrast."
                )
            count = 0
            
            '''' Memory Issue: Does not work - cellxgene computation as a whole without using mini-batch'''
            # for i,g in enumerate(graphs):
            #     g = g.to(device)
                
            #     X = g.ndata['gt']
            #     X_bins = g.ndata['bin_val'].to(device)
            #     z_val = g.ndata['z_val'].to(device)
            #     adj = g.ndata['adj'].to(device)
            #     pos = g.ndata["pos"]
                
            #     # Container for all predictions
            #     X_pred_full = torch.zeros_like(X).to(device)
            #     cell_emb_full = torch.zeros((X.shape[0], emb_size), dtype=torch.float32)          
            
            #     cell_attn_mask = ~adj.bool()
                
            #     if task == "imputation":
            #         mask = eval_mask
            #     elif task == "cluster":
            #         mask = torch.zeros_like(eval_mask, dtype=torch.bool)
                    
            #     cell_ids_len = X_bins.shape[0]
            #     cell_ids = torch.arange(cell_ids_len, device=X.device) 
                
                
            #     iden_emb, tokens = self.entry_token(cell_ids, X_bins, z_val, mask)
            #     B = 1
            #     C, G, D = tokens.shape
                
            #     if is_cls:
            #         cls = self.cls_token.expand(B, C, 1, D)
            #         '''This is with CLS'''
            #         x = torch.cat([cls, tokens.view(B, C, G, D)], dim=2)  #B,C, G+1, D_model
            #     else:
            #         ''' Without CLS'''
            #         x = tokens.view(B,C,G,D)
                    
            #     x = self.encoder(x, cell_attn_mask, target_idx=None)
                
                
            #     if is_cls: 
            #         cls_out = x[:, :, 0, :]
            #         genes_out = x[:, :, 1:, :]     #B,C,G,D
            #     else:
            #         '''This is without CLS'''
            #         genes_out = x
            
            #     X_pred = self.decoderGene(genes_out).squeeze(-1).squeeze(0)
            #     X_pred_full = X_pred
                
            #     if not is_cls:
            #         cell_context =  genes_out.max(dim=2)[0].squeeze(0)
            #         cell_emb_full = cell_context
            #     else:
            #         cell_emb_full = cls_out.squeeze(0)
                    
            #     if args.task == "cluster":
            #         '''This is for clustering purpose'''
            #         return comprehensive_evaluate(cell_emb_full.cpu().detach().numpy(), X.detach().cpu().numpy(), eval_mask, pos, lbl, logger, slice=slice, slice_num=slice_num, num_clusters=args.num_clusters, output_dir=args.output_dir)
            #     elif args.task == "imputation":
            #         '''This is for imputation purpose'''
            #         return comprehensive_evaluate_test(X_pred_full.cpu().detach().numpy(), X.detach().cpu().numpy(), eval_mask.cpu().detach().numpy())
                

            for i,g in enumerate(graphs):
                epoch = 0
                g = g.to(device)
                
                X = g.ndata['gt']
                X_bins = g.ndata['bin_val'].to(device)
                z_val = g.ndata['z_val'].to(device)
                adj = g.ndata['adj'].to(device)
                # adj_pca_filtered = g.ndata['adj_filtered'].to(device)
                pos = g.ndata["pos"]
                X_pca = g.ndata["X_pca"]

                # Container for all predictions
                X_pred_full = torch.zeros_like(X).to(device)
                cell_emb_full = torch.zeros((X.shape[0], emb_size), dtype=torch.float32)
                # cell_emb_full = torch.zeros((X.shape[0], 30), dtype=torch.float32)

                for target_idx, cell_ids in iter_cell_batches(
                    adjacency_full=adj,
                    batch_size=batch_size,
                    mode=mode,
                    shuffle=False,     # deterministic order for eval
                    device=device,
                ):
                    
                    # Slice data for this batch
                    bin_batch   = X_bins[cell_ids]    # [C_batch, G]
                    z_batch     = z_val[cell_ids]          # [C_batch, G]

                    if task == "imputation":
                        mask_batch  = eval_mask[cell_ids]  # [C_batch, G], bool
                    elif task == "cluster":
                        
                        mask = torch.zeros_like(eval_mask, dtype=torch.bool) # [all 0s of size: eval_mask]
                        mask_batch = mask[cell_ids]
                        
                    # Build adjacency submatrix for column attention
                    A_sub = adj[cell_ids][:, cell_ids].bool()  # [C_batch, C_batch]
                    
                    A_hat = adj[cell_ids][:, cell_ids].float()
                    deg = A_hat.sum(dim=1, keepdim=True).clamp_min(1e-8)
                    A_hat = A_hat / deg 
                    
                    # Allow self-attention for stability
                    cell_attn_mask_batch = ~A_sub   # True = block
                    # cell_attn_mask_batch_pca_filter = ~A_sub_pca_filtered
                    
                    
                    
                    C_dst = target_idx.numel()
                    
                    
                    A_sub_pre_edge = A_sub[:C_dst, :]
                    edge_idx = adj_to_edge_index(A_sub_pre_edge)
                    
                    iden_emb, tokens = self.entry_token(cell_ids, bin_batch, z_batch, mask_batch)
                    B = 1
                    C, G, D = tokens.shape
                    

                
                    # cls = self.cls_token.expand(B, C, 1, D) + cell_e
                    if is_cls:
                        cls = self.cls_token.expand(B, C, 1, D)
                        '''This is with CLS'''
                        x = torch.cat([cls, tokens.view(B, C, G, D)], dim=2)  #B,C, G+1, D_model
                    else:
                        ''' Without CLS'''
                        x = tokens.view(B,C,G,D)
                    
                    '''encoder step'''                      
                    x = self.encoder(x, cell_attn_mask_batch, target_idx, A_hat, epoch, edge_idx, count)
                    '''Mute this for CLS decoder'''
                    # x = x.squeeze(0)
                    
                    
                
                    # '''This is with CLS''' 
                    if is_cls: 
                        cls_out = x[:, :, 0, :]
                        genes_out = x[:, :, 1:, :]     #B,C,G,D
                    else:
                        '''This is without CLS'''
                        genes_out = x
                    
                    
                    '''decoder step'''
                    X_pred_batch = self.decoderGene(genes_out).squeeze(-1).squeeze(0)
                    if is_cls:
                        pass
                    
                    X_pred_dst = X_pred_batch[:C_dst, :]                 
                    # cls_dst    = cls_out[:, :C_dst, :].squeeze(0)
                    
                    X_pred_full[target_idx] = X_pred_dst
                    # cell_emb_full[target_idx] = cls_dst.detach().cpu()
                    
                    
                    # cell_emb_batch = pool(x)
                
                    if not is_cls:
                        cell_context =  genes_out.max(dim=2)[0].squeeze(0)
                        cell_emb_batch = cell_context[:C_dst, :]
                        cell_emb_full[target_idx] = cell_emb_batch.cpu()
                    else:
                        cls_dst    = cls_out[:, :C_dst, :].squeeze(0)
                        cell_emb_full[target_idx] = cls_dst.cpu()
                
                # cell_context = genes_out.mean(dim=2)[0]
                # cell_context = pool(x)
                # cell_static = self.entry_token.cell_emb(cell_ids_cpu)
                # cell_emb_batch = x.mean(dim=1)
                
                # cell_emb_batch = cell_context[:C_dst, :]
                
                # Store predictions for these cells
                # X_pred_full[cell_ids_cpu] = X_pred_batch
                
                # cell_emb_full[target_idx] = cell_emb_batch.cpu()
                # cell_emb_full[cell_ids_cpu] = cls_out.squeeze(0).cpu()
                
                # import pdb    
                # pdb.set_trace()
            
            if args.replicate_exp:
                if save_pt_name is None:
                    raise ValueError(
                    "save_pt_name is required when replicate_exp is enabled."
                    )

                artifact_dir = os.path.join(args.checkpoint_dir, save_pt_name)
                os.makedirs(artifact_dir, exist_ok=True)

                artifact_path = os.path.join(
                artifact_dir,
                f"{slice}_{args.num_layers}layers_"
                f"{args.max_epoch}epochs_cls.pt",
                )

                torch.save(
                    {
                        "cell_ids": torch.arange(adj.shape[0]),
                        "cls_embeddings": cell_emb_full,
                        "adjacency_mtx": adj,
                        "pca_embedding": X_pca,
                        "task": args.task,
                        "repeat_id": slice_num,
                        **objective_metadata(args),
                    },
                    artifact_path,
                )
            
            if args.task == "cluster":
                '''This is for clustering purpose'''
                return comprehensive_evaluate(cell_emb_full.cpu().detach().numpy(), X.detach().cpu().numpy(), eval_mask, pos, lbl, logger, slice=slice, slice_num=slice_num, num_clusters=args.num_clusters, output_dir=args.output_dir,
                                              dataset=args.dataset, epochs=args.max_epoch, num_layers=args.num_layers)
            elif args.task == "imputation":
                '''This is for imputation purpose'''
                return comprehensive_evaluate_test(X_pred_full.cpu().detach().numpy(), X.detach().cpu().numpy(), eval_mask.cpu().detach().numpy())
            
            count += 1
    
        
    def embed(self, g, x):
        rep = self.encoder(g, x)
        return rep

    @property
    def enc_params(self):
        return self.encoder.parameters()
    
    @property
    def dec_params(self):
        return chain(*[self.encoder_to_decoder.parameters(), self.decoder.parameters()])
    
    
    def row_normalize(X, eps=1e-8):
        return X / (X.sum(axis=1, keepdims=True) + eps)
