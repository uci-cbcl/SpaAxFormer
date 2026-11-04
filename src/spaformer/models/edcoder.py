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

#from sklearn.cluster import KMeans
#from sklearn.metrics.cluster import normalized_mutuscore
#from sklearn.metrics import adjusted_rand_score
#from sklearn.preprocessing import LabelEncoder

# Bring in Model
# This is for our model
from .axialformer_temp_fix import AxialTransformer_temp_fix
from .additional_gat import GAT_cls
# from .axialformer_temp_fix_col_gat import 

# from .gcn_simple import GraphDecoder, ResidualGraphDecoder, GraphDecoder2, GraphDecoder_mid, GraphDecoder_gate, GATDecoder

# Add our model here as well (Look over Performer)

from ..evaluation import comprehensive_evaluate
from ..evaluation import comprehensive_evaluate_test
from ..utils import Noise_Cross_Entropy, Contrastive_Loss_Spatial, filter_outer_positives, filter_adjacency_by_similarity, gene_sim_graph, iter_cell_batches, Noise_Cross_Entropy_anchor_only, weighted_spatial_contrastive, adj_to_edge_index, weighted_gene_column_contrastive, build_gene_pca_embeddings, batch_gene_positive_index,batch_gene_positive_indices
from ..utils import objective_metadata

    
def setup_module(m_type, enc_dec, in_dim, d_emb_size, num_hidden, out_dim, num_layers, dropout, activation, norm, nhead, nhead_out, attn_drop, pe=None, negative_slope=0.2, concat_out=True, latent_dim=20, cat_pe=False, pe_aug=False, aggr='SAGE') -> nn.Module:
    if m_type != "axialformer_temp_fix":
        raise ValueError("Release supports only axialformer_temp_fix")
    mod = AxialTransformer_temp_fix(
        d_model = d_emb_size,
        n_heads = nhead,
        num_layers=num_layers,
        dropout = dropout
    )
    return mod

# class MeanAct(nn.Module):
#     """Mean activation class."""

#     def __init__(self, softmax, standardscale):
#         super().__init__()
#         self.standardscale = standardscale
#         self.softmax = softmax

#     def forward(self, x):
#         if not self.softmax:
#             return torch.clamp(torch.exp(x), min=1e-5, max=1e6)
#         else:
#             return torch.softmax(x, 1) * self.standardscale

# class DispAct(nn.Module):
#     """Dispersion activation class."""

#     def __init__(self):
#         super().__init__()

#     def forward(self, x):
#         return torch.clamp(F.softplus(x), min=1e-4, max=1e4)

# class ZINB(nn.Module):
#     """ZINB Decoder.
#     Parameters
#     ----------
#     input_dim : int
#         dimension of input feature.
#     n_z : int
#         dimension of latent embedding.
#     n_dec_1 : int optional
#         number of nodes of decoder layer 1.
#     n_dec_2 : int optional
#         number of nodes of decoder layer 2.
#     """

#     def __init__(self, n_z, input_dim, standardscale, n_dec_1=128, n_dec_2=128, softmax=True, disp='gene_cell'):
#         super().__init__()
#         self.input_dim = input_dim
#         self.dec_1 = nn.Linear(n_z, n_dec_1)
#         self.dec_2 = nn.Linear(n_dec_1, n_dec_2)
#         self.dec_mean = nn.Sequential(nn.Linear(n_dec_2, self.input_dim), MeanAct(softmax, standardscale))
#         self.dec_pi = nn.Sequential(nn.Linear(n_dec_2, self.input_dim), nn.Sigmoid())
#         self.disp = disp
#         if disp == 'gene':
#             self.dec_disp = nn.Parameter(torch.ones(input_dim))
#         else:
#             self.dec_disp = nn.Sequential(nn.Linear(n_dec_2, self.input_dim), DispAct())

#     def forward(self, z):
#         """Forward propagation.
#         Parameters
#         ----------
#         z :
#             embedding.
#         Returns
#         -------
#         _mean :
#             data mean from ZINB.
#         _disp :
#             data dispersion from ZINB.
#         _pi :
#             data dropout probability from ZINB4
#         """
        
#         h = F.relu(self.dec_1(z))
# #         h = F.relu(self.dec_2(h))
#         _mean = self.dec_mean(h)
#         if self.disp == 'gene':
#             _disp = self.dec_disp.repeat(z.shape[0], 1)
#         else:
#             _disp = self.dec_disp(h)
#         _pi = self.dec_pi(h)
#         return _mean, _disp, _pi

# class LatentModel(nn.Module):
#     def __init__(self, h_dim, z_dim, kl_weight=1e-6, warmup_step=10000):
#         super().__init__()
#         self.hid_2mu = nn.Linear(h_dim, z_dim)
#         self.hid_2sigma = nn.Linear(h_dim, z_dim)
#         self.kl = 0
#         self.kl_weight = kl_weight
#         self.step_count = 0
#         self.warmup_step = warmup_step
    
#     def kl_schedule_step(self):
#         self.step_count += 1
#         if self.step_count < self.warmup_step:
#             self.kl_weight = self.kl_weight + (1e-2 - 1e-6) / self.warmup_step
#         elif self.step_count == self.warmup_step:
#             pass
# #             self.step_count = 0
# #             self.kl_weight = 1e-6
        
#     def forward(self, h):
#         mu = self.hid_2mu(h)
#         log_var = self.hid_2sigma(h)
#         sigma = torch.exp(0.5 * log_var)
#         epsilon = torch.randn_like(sigma)
        
#         if self.training:
#             z = mu + sigma*epsilon
#             self.kl = -0.5 * (1 + log_var - mu ** 2 - log_var.exp()).sum() / z.shape[0]  * self.kl_weight
#             self.kl_schedule_step()
#         else:
#             z = mu
#         return z

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
        # cls_e = cls_e.unsqueeze(2)  # (batch, embsize, 1)
        cls_e = cls_e.squeeze(0).unsqueeze(-1) #(batch, embsize, 1)
        # the pred gene expr values, # (batch, seq_len)
        pred_value = torch.bmm(self.W(query_vecs), cls_e).squeeze(-1)
        # return dict(pred=pred_value)
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
    

# class GeneAttentionPool(nn.Module):
#     def __init__(self, d_model: int, hidden: int = 64):
#         super().__init__()
#         self.score_mlp = nn.Sequential(
#             nn.Linear(d_model, hidden),
#             nn.Tanh(),
#             nn.Linear(hidden, 1),
#         )

#     def forward(self, token_repr: torch.Tensor) -> torch.Tensor:
#         # token_repr: [C, G, D]
#         scores  = self.score_mlp(token_repr).squeeze(-1)  # [C, G]
#         weights = scores.softmax(dim=1)                   # [C, G]
#         cell_emb = torch.einsum("cg,cgd->cd", weights, token_repr)
#         return cell_emb

# class ZINBLoss(nn.Module):
#     """ZINB loss class."""

#     def __init__(self):
#         super().__init__()

#     def forward(self, x, mean, disp, pi, scale_factor, ridge_lambda=0.0):
#         """Forward propagation.
#         Parameters
#         ----------
#         x :
#             input features.
#         mean :
#             data mean.
#         disp :
#             data dispersion.
#         pi :
#             data dropout probability.
#         scale_factor : list
#             scale factor of mean.
#         ridge_lambda : float optional
#             ridge parameter.
#         Returns
#         -------
#         result : float
#             ZINB loss.
#         """
#         eps = 1e-10
#         scale_factor = scale_factor.unsqueeze(-1)
#         mean = mean * scale_factor

#         t1 = torch.lgamma(disp + eps) + torch.lgamma(x + 1.0) - torch.lgamma(x + disp + eps)
#         t2 = (disp + x) * torch.log(1.0 + (mean / (disp + eps))) + (x * (torch.log(disp + eps) - torch.log(mean + eps)))
#         nb_final = t1 + t2

#         nb_case = nb_final - torch.log(1.0 - pi + eps)
#         zero_nb = torch.pow(disp / (disp + mean + eps), disp)
#         zero_case = -torch.log(pi + ((1.0 - pi) * zero_nb) + eps)
#         result = torch.where(torch.le(x, 1e-8), zero_case, nb_case)

#         if ridge_lambda > 0:
#             ridge = ridge_lambda * torch.square(pi)
#             result += ridge
#         result = torch.mean(result)
#         return result
    
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
            
            # d_emb_size : int = 100,
            # d_emb_size : int = 256,
            d_emb_size = 128,
            # d_emb_size = 32,
            # d_emb_size = 64,
            # d_emb_size : int = 80,
            
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
        '''This is used for corrupting nodes - not used in our case'''
        # self._mask_node_rate = mask_node_rate
        # self._mask_feature_rate = mask_feature_rate

        self._encoder_type = encoder_type
        self._decoder_type = decoder_type
        
        '''we need to add this properly - temporarily'''
        self._d_emb_size = d_emb_size
        
        # self._output_hidden_size = num_hidden
        # self._concat_hidden = concat_hidden
        # self._drop_node_rate = drop_node_rate
        # self._drop_model = DropNode(drop_node_rate)
        self._objective = objective
        
        
        # create cell token, gene token here
        
        '''Here we set r: 20'''
        mask_e = nn.Parameter(torch.randn(self._d_emb_size))
        # self.cell_token = CellTokenizer(in_cell, in_dim, d_emb_size, mask_e)
        # self.gene_token = GeneTokenizer(in_cell, in_dim, d_emb_size, mask_e)
        
        '''Here we get all the values needed to create entry tokenizer'''
                
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

        # dec_in_dim = latent_dim
        # dec_num_hidden = num_hidden // nhead_out if decoder_type in ("gat", "dotgat") else num_hidden 
        # if objective in ['vae', 'maskvae']:
        #     enc_out_dim = enc_num_hidden
        # else:
        #     enc_out_dim = latent_dim
        
        # import pdb
        # pdb.set_trace()
            
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
        
        # self.encoder = setup_module(
        #     m_type=encoder_type,
        #     enc_dec="encoding",
        #     in_dim=in_dim,
        #     num_hidden=enc_num_hidden,
        #     out_dim=enc_out_dim,
        #     num_layers=num_layers,
        #     nhead=enc_nhead,
        #     nhead_out=enc_nhead,
        #     concat_out=True,
        #     activation=activation,
        #     dropout=feat_drop,
        #     attn_drop=attn_drop,
        #     negative_slope=negative_slope,
        #     norm=norm,
        #     latent_dim = latent_dim,
        #     pe = pe,
        #     cat_pe = cat_pe,
        #     pe_aug = pe_aug,
        #     aggr = aggr,
        # )
        
        # # build decoder
        # if decoder_type == 'zinb':
        #     self.decoder = ZINB(dec_in_dim, in_dim, standardscale=standardscale)
        # else:
        
        
        # decoder should be MLP X2 if this is for our model else one MLP
        
        '''Build Decoder - in_dim refers to gene dim'''
        self.decoder1 = nn.Sequential(
            nn.Linear(d_emb_size, num_hidden),
            nn.PReLU(),
            nn.Dropout(feat_drop),
            nn.Linear(num_hidden, in_dim),
            nn.ReLU()
        )
        
        self.decoder2 = nn.Sequential(
            nn.Linear(d_emb_size, num_hidden),
            nn.PReLU(),
            nn.Dropout(feat_drop),
            nn.Linear(num_hidden, in_cell),
            nn.ReLU()
        )
        
        self.decoderE = nn.Linear(d_emb_size, 1)
        self.decoderGene = GeneDecoder(d_emb_size, num_hidden)
        # self.decoderGene = GeneDecoder(30, num_hidden)
        self.decoderCell = CellCLSDecoder(d_emb_size)
        # self.gat = AdditiveGAT(d_emb_size, d_emb_size) 
        self.gat = GAT_cls(d_emb_size, d_emb_size)
        
        # import pdb
        # pdb.set_trace()
        
        # self.negative_cache = GlobalCLSNegativeCache(in_cell, d_emb_size) 
    
    
        # self.decoderCell = CellCLSDecoder(30)
        
        # self.decoderCLS = GeneCLSDecoder(d_emb_size, num_hidden)
        
        # self.decoder1 = GraphDecoder(d_emb_size, num_hidden, in_dim)
        # self.decoder1 = ResidualGraphDecoder(d_emb_size, num_hidden, in_dim)
        # self.decoder1 = GraphDecoder2(d_emb_size,num_hidden,in_dim)
        # self.decoder1 = GraphDecoder_mid(d_emb_size, num_hidden, in_dim)
        # self.decoder1 = GraphDecoder_gate(d_emb_size, num_hidden, in_dim)
        # self.decoder1 = GATDecoder(d_emb_size, num_hidden, in_dim)
        # self.decoder1 = ResidualGATDecoder(d_emb_size, num_hidden, in_dim)
        
        # self.decoder2 = GraphDecoder(d_emb_size, num_hidden, in_cell)
        
        # self.decoder1 = nn.Sequential(
            
        # )
        
        # self.mse1 = nn.MSELoss()
        # self.mse2 = nn.MSELoss()
            
            
        # self.enc_mask_token = nn.Parameter(torch.zeros(1, in_dim))
        
        if objective in ['mask', 'ae']:
            self.encoder_to_decoder = nn.Identity()
        # elif objective in ['maskvae', 'vae']:
        #     self.encoder_to_decoder = LatentModel(enc_num_hidden, dec_in_dim)
        
        
        # cell tokenize / gene tokenize
        
        
        
        
#         if concat_hidden:
#             self.encoder_to_decoder = nn.Linear(num_hidden * num_layers, dec_in_dim, bias=False)
#         else:
#             self.encoder_to_decoder = nn.Linear(dec_in_dim, dec_in_dim, bias=False)

        # self.criterion = self.setup_loss_fn(loss_fn, alpha_l)
        # self.zinb_loss = ZINBLoss()

    @property
    def output_hidden_dim(self):
        return self._output_hidden_size

    def focused_mask(self, gex, pc, eval=False, first_time=False): 
        # mask_folders = ["2-spiral", "zebra", "concentric"]       
        cells = gex.shape[0]
        genes = gex.shape[1]
        
        marker_gene_indices = [0,1]
        all_gene_indices = np.arange(gex.shape[1])
        non_marker_indices = np.setdiff1d(all_gene_indices, marker_gene_indices)
        
        total_masked_genes = gex.shape[0] * gex.shape[1] * pc
        # num_marker = int(0.7 * total_masked_genes)
        # num_house_keeping = int(0.3 * total_masked_genes)
        
        # num_marker = 30
        # num_house_keeping = 10
        
        num_marker = 5000
        num_house_keeping = 2500
        
        # num_marker = 2500
        # num_house_keeping = 5000
        
        # num_marker = 3500
        # num_house_keeping = 1500
        
        
        mask_name = f"shared_mask_{num_marker}_{num_house_keeping}"

        marker_cells = np.random.choice(cells, size=num_marker, replace=False)
        marker_genes = np.random.choice(marker_gene_indices, size=num_marker, replace=True)
        
        non_marker_cells = np.random.choice(cells, size=num_house_keeping, replace=True)
        non_marker_genes = np.random.choice(non_marker_indices, size=num_house_keeping, replace=True)
        
        
        
        masks = np.ones((cells, genes))
        
        masks[(marker_cells, marker_genes)]=0
        masks[(non_marker_cells, non_marker_genes)]=0
        
        if eval:
            
            if first_time:
                self.save_fixed_mask(masks, mask_name)
            masks = self.load_fixed_mask(mask_name)
        return masks
    
    def save_fixed_mask(self, masks, mask_name, mask_folder="data/masks"):
        np.save(os.path.join(mask_folder, mask_name), masks)
    
    
    def load_fixed_mask(self, mask_name, mask_folder="data/masks"):     
        mask = np.load(os.path.join(mask_folder,f"{mask_name}.npy"))
        return mask
        
    # def encoding_mask(self, gex_zero_out, M_eval, pc_train_from_rest=20, eval=False, first_time=False):
    #     '''
    #     Introduce mask here.
    #     gex corresponds to log-transformed input. Nothing else.
    #     '''
    #     # import pdb
    #     # pdb.set_trace()
        
    #     '''For real dataset'''
    #     allowed = ~M_eval
    #     draw = (torch.rand(allowed.shape, device=allowed.device) < pc_train_from_rest/100)
    #     M_train = allowed & draw
    #     M_in = M_eval | M_train
        
    #     fully_hidden = M_in.all(dim=1)
        
    #     fix_rows = fully_hidden & allowed.any(dim=1)
        
    #     '''We want to retrieve fully_hidden and allowed none'''
    #     notify_rows = fully_hidden & ~(allowed.any(dim=1))
        
    #     if notify_rows.any():
    #         print("M_Eval covered entire row")
        
    #     '''This code needs to be verified - at most - we need to fix it for all rows not just single row'''
    #     if fix_rows.any():
            
    #         # import pdb
    #         # pdb.set_trace()
            
    #         rows = fix_rows.nonzero(as_tuple=False).squeeze(1)
    #         '''We use argmax - meaning highest value - but we will think of making it to random later on'''
    #         first_allowed = allowed[rows].float().argmax(dim=1)
    #         M_in[rows, first_allowed] = False
    #         M_train[rows, first_allowed] = False
        
    #     # M_in = torch.tensor(M_in, dtype=torch.bool, device=gex.device)
    #     # M_train = torch.tensor(M_train, dtype=torch.bool, device=gex.device)
        
    #     assert not (M_eval & M_train).any()  
        
    #     '''Check: 
    #         Technically we can apply M_train and the result should be same as
    #         gex already applies M_Eval and zero-ed out
    #     '''
        
    #     M_in = M_in.to(device=gex_zero_out.device, dtype=torch.bool)    
        
    #     train_gex = gex_zero_out.masked_fill(M_train, 0.0)
    #     # check_gex = gex_zero_out.masked_fill(M_in, 0.0)
        
    #     M_in = M_in.to(device=gex_zero_out.device, dtype=torch.float)
        
    #     return train_gex, M_in, M_train
    
    # def encoding_mask_nonzeros(self, gex_zero_out, M_eval, pc_train_from_rest=20, eval=False, first_time=False):
    #     '''
    #     Introduce mask here.
    #     gex corresponds to log-transformed input. Nothing else.
    #     '''
    #     # import pdb
    #     # pdb.set_trace()
        
    #     '''For real dataset'''
    #     allowed = ~M_eval
    #     nonzero_mask = (gex_zero_out != 0)
    #     candidates = nonzero_mask & allowed
        
    #     candidate_indices = candidates.nonzero(as_tuple=False)
        
    #     num_to_pick = int(pc_train_from_rest / 100 * candidate_indices.shape[0])
        
    #     selected_indices = candidate_indices[torch.randperm(candidate_indices.shape[0])[:num_to_pick]]
        
    #     draw = torch.zeros_like(gex_zero_out, dtype=torch.bool)
    #     draw[selected_indices[:, 0], selected_indices[:, 1]] = True
    #     '''This should not change anything'''
    #     M_train = allowed & draw
    #     M_in = M_eval | M_train
        
    #     fully_hidden = M_in.all(dim=1)
        
    #     fix_rows = fully_hidden & allowed.any(dim=1)
        
    #     '''We want to retrieve fully_hidden and allowed none'''
    #     notify_rows = fully_hidden & ~(allowed.any(dim=1))
        
    #     if notify_rows.any():
    #         print("M_Eval covered entire row")
        
    #     '''This code needs to be verified - at most - we need to fix it for all rows not just single row'''
    #     if fix_rows.any():
            
    #         # import pdb
    #         # pdb.set_trace()
            
    #         rows = fix_rows.nonzero(as_tuple=False).squeeze(1)
    #         '''We use argmax - meaning highest value - but we will think of making it to random later on'''
    #         first_allowed = allowed[rows].float().argmax(dim=1)
    #         M_in[rows, first_allowed] = False
    #         M_train[rows, first_allowed] = False
        
    #     # M_in = torch.tensor(M_in, dtype=torch.bool, device=gex.device)
    #     # M_train = torch.tensor(M_train, dtype=torch.bool, device=gex.device)
        
    #     assert not (M_eval & M_train).any()  
        
    #     '''Check: 
    #         Technically we can apply M_train and the result should be same as
    #         gex already applies M_Eval and zero-ed out
    #     '''
        
    #     M_in = M_in.to(device=gex_zero_out.device, dtype=torch.bool)    
        
    #     train_gex = gex_zero_out.masked_fill(M_train, 0.0)
    #     # check_gex = gex_zero_out.masked_fill(M_in, 0.0)
        
    #     M_in = M_in.to(device=gex_zero_out.device, dtype=torch.float)
        
    #     return train_gex, M_in, M_train
    
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
        
        import pdb
        pdb.set_trace()

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
            import pdb
            pdb.set_trace()
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
        
        # if epoch == 10:
        #     import pdb
        #     pdb.set_trace()

        # cosine similarity (without /tau) for weights
        # turn cosine into weights in [0,1]
        # Option: sigmoid centered at 0: w = sigmoid(alpha * cos)
        # import pdb
        # pdb.set_trace()
        
        w = torch.sigmoid(alpha * gene_sim)

        # only weight positives; non-positives get weight 0 in numerator
        w = w * pos_mask.float()

        # stable log numerator: log sum_j w_ij * exp(logits_ij)
        # compute as logsumexp(logits + log(w)) over positives
        neg_inf = torch.finfo(logits.dtype).min
        logw = torch.log(w + eps)                        # [B, K]
        num_terms = (logits + logw).masked_fill(~pos_mask, neg_inf)
        log_num = torch.logsumexp(num_terms[valid], dim=1)
        
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
        # denom_mask = ~self_mask

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
    
    
    # @torch.no_grad()
    # def update_negative_cache(self, cache_payload):
    #     self.negative_cache.update(
    #         cls_out=cache_payload["cls_out"],
    #         cell_ids=cache_payload["cell_ids"],
    #         B=cache_payload["B"],)

    def forward(self, args, g, X, X_pca, X_bins, z_val, mask_in, mask_train, cell_ids_cpu, target_idx, epoch, count):
        # if self._objective in ['mask', 'maskvae']:
            # should I introduce mask here?
            
            # use_g, use_x, (mask_nodes, keep_nodes) = self.encoding_mask_noise(g, x)
        # else:
        
        
        requires_cls = args.encoder == "axialformer_temp_fix"

        if requires_cls and not args.cls:
            raise ValueError(
            "axialformer_temp_fix requires CLS for the retained-CLS configuration. "
            "For GEP-only, remove --no_cls and use --no_gepc without --contrast.")    
        
        is_cls = args.cls
        is_contrast = args.contrast
        is_gat = args.gat
        is_gepc = args.gepc
        
        '''This is where we set the training objective'''
        
        # pc = 50
        
        mse1 = nn.MSELoss()
        mse2 = nn.MSELoss()
        self.BCE_loss = nn.BCEWithLogitsLoss()
        
        
        
        # pc = 3
        
        # pc = 5
        '''This is imputation setting - wit zero and nonzero entries'''
        # pc = 4
        
        # pc = 10
        
        # pc = 45
        
        # pc = 16
        
        # pc = 80
        
        # pc = 20
        
        # pc = 20
        
        # pc = 16
        
        # pc = 3
        
        # import pdb
        # pdb.set_trace()
            
        ''''use_x implies masked_gex'''
        '''This is for imputation'''
        
        '''This mask generation could change'''
        
        # if args.task == "imputation":
            # use_x, mask_in, mask_train = self.encoding_mask_nonzeros(X, eval_mask, pc)
        # elif args.task == "cluster":
        #     '''This is for clustering'''
        #     use_x, mask_in, mask_train = self.encoding_mask(X, eval_mask, pc)
            
        '''This is where we call forward function -> to create actual embeddings given mask_in, bin_val, Z_val'''
        
        
        '''adjacency matrix - might be better to pass in adj from main.py'''
        
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
        
        '''This pertains to Axialformer_temp_fix module'''     
        A_hat = adj[target_idx][:, cell_ids_cpu].float()
        deg = A_hat.sum(dim=1, keepdim=True).clamp_min(1e-8)
        A_hat = A_hat / deg
        
                
        entry_iden_emb, tokens = self.entry_token(cell_ids_cpu, bin_batch, z_batch, mask_in_batch)    # C, G, D_model
        B = 1
        # x = tokens.unsqueeze(0) #B, C, G, D_model where B = 1
        
        C,G,D = tokens.shape
        C_dst = target_idx.numel()
        
        '''check'''
        # if is_gat:
        #     A_sub.fill_diagonal_(False)
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
       
        '''This is tested module'''
        if is_gat:
            x = self.gat(cls_out.squeeze(0), edge_idx, C_dst, epoch)
            x = x.unsqueeze(0)
            
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
            
        '''This needs fix'''
        # if  (epoch >=100) and is_contrast:
        # if  (epoch >=125) and is_contrast:
        # if  (epoch >=150) and is_contrast:
        # if epoch >= 50 and is_contrast:
        # if (epoch >= 25) and is_contrast:
        # if (epoch >=100) and is_contrast:  
        # if (epoch >= 75) and is_contrast:
        if is_contrast:
            
            pos_candidate_mask = A_sub[:C_dst, :]
            # pos_pca_candidate_mask = A_sub_pca_filtered[:C_dst, :]
            
            # contrastive_loss = self.batch_pca_knn_InfoNCE(cls_out.squeeze(0), X_pca_batch, pos_candidate_mask, tau=1)
            # pos_candidate_mask
            contrastive_loss = self.weighted_spatial_gene_sim_contrastive(cls_out.squeeze(0), X_pca_batch, cell_ids_cpu, C_dst, pos_candidate_mask, epoch)
            # pos_candidiate_mask
            # contrastive_loss = self.weighted_spatial_gene_sim_contrastive_pca_thres_neg(cls_out.squeeze(0), X_pca_batch, cell_ids_cpu, C_dst, pos_pca_candidate_mask, epoch, negative_pca_threshold=0)
            
            # contrastive_loss = self.weighted_spatial_gene_sim_contrastive_neg_not_knn(cls_out.squeeze(0), X_pca_batch, cell_ids_cpu, C_dst, pos_candidate_mask, pos_pca_candidate_mask, epoch)
            
            # contrastive_loss = self.weighted_spatial_gene_sim_contrastive_neg_no_knn_thres(cls_out.squeeze(0), X_pca_batch, cell_ids_cpu, C_dst, pos_candidate_mask, pos_pca_candidate_mask, epoch, negative_pca_threshold=0)
            
            # contrastive_loss = self.batch_supcon_loss_pca_filter_negatives_thres(cls_out.squeeze(0), X_pca_batch, pos_candidate_mask, pos_pca_candidate_mask, negative_pca_threshold=0)
            
            cache_payload = {"cls_out":cls_out.squeeze(0).detach(),"cell_ids": cell_ids_cpu.detach(), "B": C_dst}
            
            # contrastive_loss = self.batch_supcon_loss(cls_out.squeeze(0), pos_candidate_mask)
            
            # contrastive_loss = self.batch_supcon_gene_sim_loss(cls_out.squeeze(0), X_pca_batch, pos_candidate_mask)
            
            # contrastive_loss = weighted_spatial_contrastive(cls_out.squeeze(0), C_dst, pos_candidate_mask, epoch)
                        
            # contrastive_loss_1 = weighted_gene_column_contrastive(genes_out.squeeze(0), C_dst, pos_mask, epoch)
            
            # contrastive_loss_1 = weighted_gene_column_contrastive(x.squeeze(0), C_dst, pos_mask, epoch)
            # contrastive_loss = weighted_spatial_contrastive(cls_out.squeeze(0), C_dst, pos_mask, epoch)
            # contrastive_loss = Noise_Cross_Entropy_anchor_only(cls_out.squeeze(0), C_dst , pos_candidate_mask, epoch)
            
            # loss = loss + 0.005 * contrastive_loss
            # loss = loss + 0.05 * contrastive_loss
            # loss = loss + 0.0005 * contrastive_loss
            
        # return 10 * (0.1 * loss + 0.9 * cell_loss) + 0.05 * contrastive_loss
    
        # return 10 * (0.1 * loss + 0.9 * cell_loss) + 0.05 * contrastive_loss
        
        # if epoch == 10:
        #     import pdb
        #     pdb.set_trace()
        
        # return 10*(0.5*loss + 0.5*cell_loss)
        # print("cell_loss: ", cell_loss)
        # print("contrast_loss", contrastive_loss)
        '''This is what we use'''
        # return 10*(0.5*loss + 0.5*cell_loss) + 0.01 * contrastive_loss
        # return 10*(0.5*loss + 0.5*cell_loss) + 0.02 * contrastive_loss
        
    
        # return 10*(0.5*loss + 0.5*cell_loss) + 0.02 * (0.5 * contrastive_loss_1 + 0.5 * contrastive_loss_2)
        # return 10*(0.5*loss + 0.5*cell_loss) + 0.02 * (0.1 * contrastive_loss_1 + 0.9 * contrastive_loss_2)
        '''our default'''
        # return 10*(0.5*loss + 0.5*cell_loss) + 0.02 * contrastive_loss, cache_payload
        '''StarMap?'''
        # return 10*(0.5*loss + 0.5*cell_loss) + 1 * contrastive_loss, cache_payload
        
        # return 10*(0.5*loss + 0.5*cell_loss) + 0.05 * contrastive_loss, cache_payload
        '''Could be BC? check result first and see'''
        # return 10*(0.5*loss + 0.5*cell_loss) + 0.1* contrastive_loss, cache_payload 
        
        # return 10*(0.5*loss + 0.5*cell_loss) + 2* contrastive_loss, cache_payload 
        
        # return 10*(0.5*loss + 0.5*cell_loss) + 1.5* contrastive_loss, cache_payload 
        
        '''StarMap Used'''
        return 10*(0.5*loss + 0.5*cell_loss) + args.contrastive_weight * contrastive_loss, cache_payload 
        
        # return 10*(0.5*loss + 0.5*cell_loss) + 3 * contrastive_loss, cache_payload 

        # return 10*(0.5*loss + 0.5*cell_loss) + 4 * contrastive_loss, cache_payload 
    
    
        # return 10*(0.5*loss + 0.5*cell_loss) + 0.1 * contrastive_loss
        # return 10*(0.5*loss + 0.5*cell_loss) + 0.05 * contrastive_loss
        # return 10*(0.5*loss + 0.5*cell_loss) + 1 * contrastive_loss
        
        # return 10*loss
        
        # '''original code - biformer'''
        # # import pdb
        # # pdb.set_trace()
        
        # gene_mask = g.attn_gene_mask
        
        # adj_gene = (~gene_mask)
        
        # coords = g.ndata["pos"]
        
        # # adj_ = torch.mm(C_enc, C_enc.T)
        
        # # import pdb
        # # pdb.set_trace()
        
        # # adj = filter_outer_positives(coords, C_enc)
        
        
        # # adj_filtered = filter_adjacency_by_similarity(adj, C_enc)
        # # import pdb
        # # pdb.set_trace()
        
        # '''Working best solution'''
        # loss_NCE_cell = Noise_Cross_Entropy(C_enc, adj)
        
        # # loss_NCE_gene = Noise_Cross_Entropy(G_enc, adj_gene)
        
        # # loss_adj = self.BCE_loss(adj_, adj.float())
        # # orth_loss_cell = self.orthogonal_loss(C_enc)
        # # orth_loss_gene = self.orthogonal_loss(G_enc)
        # # loss_lap = self.laplacian_loss(C_enc, adj)
        
        
        # # loss_NCE = Contrastive_Loss_Spatial(C_enc, pos_mask)
        
        #  # loss_NCE = self.spatial_contrastive_loss(C_enc, adj)
        
        
        # '''
        # We do not need this part for our model
        # but for extension incorporate this part.
        # '''
        # # rep = self.encoder_to_decoder(enc_rep)
        
        # # import pdb
        # # pdb.set_trace()
        
        
        # recon1 = self.decoder1(C_enc)
        # recon2 = self.decoder2(G_enc)
        # recon2 = recon2.T
        
        # # '''Uses graph decoder'''
        # # recon1 = self.decoder1(g, C_enc)
        # # recon2 = self.decoder2(G_enc)
        
        # # import pdb
        # # pdb.set_trace()
        
        # '''One before masking applied: use_g.ndata['gt']'''
        # loss1 = mse1(recon1[mask_train], g.ndata['gt'][mask_train])
        # loss2 = mse2(recon2[mask_train], g.ndata['gt'][mask_train])
        
        # # loss = math.sqrt(F.mse_loss(recon[mask], use_g.ndata['gt'][mask]))
        
        # # return 0.5 *(loss1) + 0.5 * (loss2)
        
        # # return 0.2 * (loss1) + 0.8 * (loss2)
        
        # # import pdb
        # # pdb.set_trace()
        
        # # return 0.8 * (loss1) + 0.2 * (loss2) + 0.01 * loss_adj_1
        
        # # return 60 * loss1 + 0.01 * loss_NCE
        
        # # return loss1
        
        # '''Preserve this'''
        # # return loss1 * 0.9 + 0.1 * loss2
        
        # # return 10*(loss1 * 0.9 + 0.1 * loss2) + 0.01 * self.gaussian_weighted_laplacian_loss(C_enc, g)
        
        # # import pdb
        # # pdb.set_trace()
        
        # '''This could change'''
        # if epoch > 1000:
        #     '''best worked - Visium'''
        #     # return 10*(loss1 * 0.9 + 0.1 * loss2) + 0.05*loss_NCE_cell 
        #     return 10 * loss1 + 0.05 * loss_NCE_cell
        
        # # + 0.05*loss_NCE_gene
        
        #     #+ 0.005*orth_loss_cell + 0.005 * orth_loss_gene
        #     '''best worked - Visium BC'''
        #     # return 10*(loss1 * 0.9 + 0.1 * loss2) + 0.001*loss_NCE_cell 
        #     '''best worked - Starmap'''
        #     # return 10*(loss1 * 0.9 + 0.1 * loss2) + 1*loss_NCE_cell 
        #     '''best-worked = Merfish'''
        #     # return 10*(loss1 * 0.9 + 0.1 * loss2) + 4*loss_NCE_cell
            
        #     # return 10*(loss1 * 0.9 + 0.1 * loss2) + 1.5*loss_NCE 
        #     # return 10*(loss1 * 0.9 + 0.1 * loss2) + 5*loss_NCE 
        #     # return 10*(loss1 * 0.9 + 0.1 * loss2) + 10*loss_NCE 
            
            
            
        #     # return 10*(loss1 * 0.9 + 0.1 * loss2) + 0.005*loss_NCE 
        
        # # # # # + 0.005*orth_loss
        # # # # # # + 0.005 * loss_lap
        # # # else:
        # # #     return 10*(loss1 * 0.9 + 0.1 * loss2)
        
        # return 10*(loss1 * 0.9 + 0.1 * loss2)
        
        # # return 10*(loss1)
            
        # return 10 * (loss2) 
        
        # return 5*(loss1 * 0.9 + 0.1 * loss2)
        
        # return 10*(loss1 * 0.9 + loss2 * 0.1) + 0.05 * loss_NCE
        
        # return 10*(loss1 * 0.9 + loss2 * 0.1) + 0.3 * loss_NCE
        
        # return 5*(loss1 * 0.9 + loss2 * 0.1)
        
        # return 10*(loss1 * 0.9 + loss2 * 0.1) + 0.01 * self.weighted_contrastive_loss(C_enc, g)
        
    
    # 0.01 * loss_adj_1
        
        # return 0.5 * (loss1) + 0.5 * (loss2)
        
        # return loss1 +  0.01 * loss_NCE
        
        # return 60 * loss1 + loss2 + 0.01 * loss_NCE
        # return 60 * loss1 + loss2
        
        # return loss2    
    
    
    '''Related to Training and Evaluation'''
    def evaluate(self, g, x, mode='valid', mask=None, no_pe=False):
        assert mode in ['valid', 'test', 'infer'], 'Invalid evaluation mode.'
            
        # if mode == 'valid' and self._objective in ['mask', 'maskvae']:
        #     use_g, use_x, (mask_nodes, keep_nodes) = self.encoding_mask_noise(g, x)   
        # else:
        
        use_g, use_x = g, x
        mask_nodes = np.arange(g.num_nodes())

        if self._encoder_type == 'performer':
            enc_rep = self.encoder(use_g, use_x, no_pe)
        else:
            assert not no_pe, '"no_pe" only supported for performers.'
            enc_rep = self.encoder(use_g, use_x)
            
        if self._concat_hidden:
            enc_rep = torch.cat(all_hidden, dim=1)
        rep = self.encoder_to_decoder(enc_rep)

        if self._decoder_type == "zinb":
            mu, _, pi = self.decoder(rep)
            recon = (1 - pi) * mu 
            recon = torch.log(recon+1)
        else:
            recon = self.decoder(rep)
        
        if mode=='infer':
            print('Overall RMSE', math.sqrt(F.mse_loss(recon, gt)))
            return recon, rep
        elif mode=='test':
            loss = math.sqrt(F.mse_loss(recon[mask], use_g.ndata['gt'][mask]))
        elif mode=='valid':
            if self._objective in ['mask', 'maskvae']:
                loss = math.sqrt(F.mse_loss(recon[mask_nodes], use_g.ndata['feat'][mask_nodes]))
            elif self._objective in ['vae', 'ae']:
                loss = math.sqrt(F.mse_loss(recon, use_g.ndata['feat']))
        return loss
    
    # '''Evaluation Method'''
    # def batch_test(self, args, graphs, eval_mask, lbl, logger=None, slice=None, slice_num=None):
    #     # import pdb
    #     # pdb.set_trace()
        
    #     with torch.no_grad():
    #         self.eval()
    #         device = next(self.parameters()).device
    #         res1 = []
    #         res2 = []
    #         encres = []
    #         ground_truths = []
    #         all_pos = []
    #         for i, g in enumerate(graphs):
    #             g = g.to(device)
    #             '''
    #             x and gt are not the same - now
    #             x - zero-ed out entries for M_eval
    #             gt - log transformed raw gene expression
    #             '''
    #             '''This is used for clustering'''
    #             if args.task == "cluster":
    #                 x = g.ndata['gt']        
    #             elif args.task == "imputation":
    #                 '''This is used for imputation'''
    #                 x = g.ndata['input']

    #             gt = g.ndata['gt'].cpu()
    #             pos = g.ndata["pos"]
    #             all_pos.append(pos)
    #             ground_truths.append(gt)
                
    #             # use_x, mask = self.encoding_mask(x, pc=10, eval=True, first_time=True)
                
    #             zero_out_x = x.detach().cpu().numpy()
    #             # print(zero_out_x)
                    
    #             C = self.cell_token(x, eval_mask)
    #             G = self.gene_token(x, eval_mask)
    #             C_enc, G_enc = self.encoder(g, C, G)
    #             cell_emb = C_enc.detach().cpu().numpy()
    #             gene_emb = G_enc.detach().cpu().numpy()
                
    #             '''Graph Decoder'''
    #             # recon1 = self.decoder1(g, C_enc).cpu()
    #             ''' MLP Decoder'''
    #             recon1 = self.decoder1(C_enc).cpu()
    #             recon2 = self.decoder2(G_enc).cpu()
    #             recon2 = recon2.T
    #             eval_mask = eval_mask.bool().cpu().numpy()
    #             res1.append(recon1)
    #             res2.append(recon2)
    #         nres1 = torch.cat(res1).numpy()
    #         nres2 = torch.cat(res2).numpy()
    #         # nres1 = None
    #         # nres2 = None
    #         # gene_emb = None
    #         # cell_emb = None
            
    #         # latent_embedding = latent_embedding1 * 0.8 + latent_embedding2 * 0.2
    #         # latent_embedding = latent_embedding2
            
            
    #         if args.task == "cluster":    
    #             return comprehensive_evaluate(nres1, nres2, cell_emb, gene_emb,  torch.cat(ground_truths).numpy(), zero_out_x, eval_mask, pos, lbl, logger, slice=slice, slice_num=slice_num, num_clusters=args.num_clusters, output_dir=args.output_dir)
    #         elif args.task == "imputation":
    #             '''This is for imputation purpose'''
    #             return comprehensive_evaluate_test(nres1, nres2, torch.cat(ground_truths).numpy(), zero_out_x, eval_mask, pos, lbl, logger)
                
                
                
                # rep = self.encoder_to_decoder(enc_rep)
                # encres.append(rep.cpu()) 
                # if self._decoder_type == "zinb":
                #     mu, _, pi = self.decoder(rep)
                #     recon = (1 - pi) * mu
                #     recon = torch.log(recon+1).cpu()
                # else:
                #     recon = self.decoder(rep).cpu()
                # res.append(recon)
            # nres = torch.cat(res).numpy()
                
                # import pdb
                # pdb.set_trace()
                
                
                # '''One before masking applied: use_g.ndata['gt']'''
                # loss1 = mse(recon1[mask], use_g.ndata['gt'][mask])
                
                # loss2 = mse(recon2[mask], use_g.ndata['gt'][mask])
                
                # return loss1 + loss2

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
            
            requires_cls = args.encoder == "axialformer_temp_fix"

            if requires_cls and not is_cls:
                raise ValueError(
                    "axialformer_temp_fix requires CLS during inference. "
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

                    # For evaluation:
                    #   eval_mask_full == True  → entry should be reconstructed (masked at input)
                    #   eval_mask_full == False → entry stays as context
                    if task == "imputation":
                        mask_batch  = eval_mask[cell_ids]  # [C_batch, G], bool
                    elif task == "cluster":
                        
                        mask = torch.zeros_like(eval_mask, dtype=torch.bool) # [all 0s of size: eval_mask]
                        mask_batch = mask[cell_ids]
                        
                    # Build adjacency submatrix for column attention
                    A_sub = adj[cell_ids][:, cell_ids].bool()  # [C_batch, C_batch]
                    # A_sub_pca_filtered = adj_pca_filtered[cell_ids][:, cell_ids].bool() 
                    
                    A_hat = adj[cell_ids][:, cell_ids].float()
                    deg = A_hat.sum(dim=1, keepdim=True).clamp_min(1e-8)
                    A_hat = A_hat / deg 
                    
                    # A_sub = A_sub + A_sub.fill_diagonal_(0)        
                    
                    # Allow self-attention for stability
                    cell_attn_mask_batch = ~A_sub   # True = block
                    # cell_attn_mask_batch_pca_filter = ~A_sub_pca_filtered
                    
                    
                    
                    C_dst = target_idx.numel()
                    
                    
                    A_sub_pre_edge = A_sub[:C_dst, :]
                    edge_idx = adj_to_edge_index(A_sub_pre_edge)
                    
                    
                    
                    # pool = GeneAttentionPool(emb_size).to(device)
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
                        # Todo
                        # X_pred_batch = self.decoderCLS(genes_out, cls_out)
                        # X_pred_cell = self.decoderCell(iden_emb, cls_out)
                        # X_pred_cell_batch = X_pred_cell[:C_dst, :]
                        pass
                    
                    X_pred_dst = X_pred_batch[:C_dst, :]
                    '''This implies cls will be all similar'''
                    
                                    
                    # cls_dst    = cls_out[:, :C_dst, :].squeeze(0)
                    
                    '''I should make this is only for imputation'''
                    X_pred_full[target_idx] = X_pred_dst
                    # cell_emb_full[target_idx] = cls_dst.detach().cpu()
                    
                    
                    # cell_emb_batch = pool(x)
                
                    '''What made this dimension change from d=2 to d=1??? -> we just do not squeeze. But this all needs to be verified'''
                    if not is_cls:
                        cell_context =  genes_out.max(dim=2)[0].squeeze(0)
                        cell_emb_batch = cell_context[:C_dst, :]
                        cell_emb_full[target_idx] = cell_emb_batch.cpu()
                    else:
                        '''cls exists'''
                        
                        '''This needs to be verified'''
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
                
                # import pdb
                # pdb.set_trace()
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
                return comprehensive_evaluate(cell_emb_full.cpu().detach().numpy(), X.detach().cpu().numpy(), eval_mask, pos, lbl, logger, slice=slice, slice_num=slice_num, num_clusters=args.num_clusters, output_dir=args.output_dir)
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
