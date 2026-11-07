import torch
import torch.nn as nn
import torch.nn.functional as F
# from graphmae.utils import create_activation, NormLayer, create_norm
# from performer_pytorch import SelfAttention
# from .layers import SignNetNodeEncoder, positionalencoding2d
import dgl
import copy
# from  .spatial_bias_attention import SpatialBiasMultiheadSelfAttention

from torch.nn.attention import sdpa_kernel, SDPBackend

torch.backends.cuda.enable_flash_sdp(True)
torch.backends.cuda.enable_mem_efficient_sdp(True)
torch.backends.cuda.enable_math_sdp(True)

class CellGraphPropagation(nn.Module):
    def __init__(self, beta=0):
        super().__init__()
        self.beta = beta

    def forward(self, x, A_hat, C_dst):
        """
        x: [C_dst, G, D] 
        A: [C_dst, C_batch] binary or weighted adjacency
        """

        # x_prop = torch.einsum('ij,jgd->igd', A_hat, x)
        
        C, G, D = x.shape    
        x_2d = x.reshape(C, G * D)        # [C, G*D]
        x_prop_2d = A_hat @ x_2d          # [C, G*D]
        x_prop = x_prop_2d.reshape(C_dst, G, D)
         
        x_new = (1.0 - self.beta) * x[:C_dst, :, :] + self.beta * x_prop
        return x_new

class CellWeightGraphPropagation(nn.Module):
    def __init__(self, beta=1.0, tau=0.2, eps=1e-8):
        super().__init__()
        self.beta = beta
        self.tau = tau
        self.eps = eps

    def forward(self, x, A_hat, C_dst):
        """
        x: [C_dst, G, D] 
        A: [C_dst, C_batch] binary or weighted adjacency
        """

        # x_prop = torch.einsum('ij,jgd->igd', A_hat, x)
        # pdb.set_trace()  
        
        C, G, D = x.shape    
        x_norm = F.normalize(x, p=2, dim=-1, eps=self.eps) 
        x_2d = x_norm.reshape(C, G * D)        # [C_batch, G*D]
        sim = x_2d @ x_2d.T          # [C_batch, C_batch]
        sim = sim[:C_dst, :]
        sim = sim / self.tau                     # temperature (e.g. tau=0.5 or 1.0)

        # mask non-neighbors
        sim = sim.masked_fill(~A_hat, -1e9) 

        alpha = torch.softmax(sim, dim=-1)  # row-wise

        x_prop = alpha @ x

        x_prop = x_prop_2d.reshape(C_dst, G, D)
         
        x_new = (1.0 - self.beta) * x[:C_dst, :, :] + self.beta * x_prop
        
        return x_new




'''This is basic function. We do not use this function'''
class AxialBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        
        self.d_model = d_model

        # Row attention: per cell, across genes
        self.row_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,  # [batch, seq, dim]
        )
        self.row_ln = nn.LayerNorm(d_model)

        # Column attention: per gene, across cells
        self.col_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.col_ln = nn.LayerNorm(d_model)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model*4),
            nn.GELU(),
            nn.Linear(d_model*4, d_model),
        )
        self.ffn_ln = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, cell_attn_mask: torch.Tensor, target_idx) -> torch.Tensor:
        """
        x: [B, C, G, D]
        cell_attn_mask: [C, C] bool (True = block) or float mask
        """
        B, C, G, D = x.shape
        assert D == self.d_model

        # ---- Row attention (within each cell across genes) ----
        row_in = x.view(B * C, G, D)  # [B*C, G, D] 
        
        # row_out = F.scaled_dot_product_attention(row_in, row_in, row_in)
        row_out, _ = self.row_attn(row_in, row_in, row_in)
        
        
        row_out = self.row_ln(row_in + row_out)
        x = row_out.view(B, C, G, D)

        # ---- Column attention (within each gene across cells) ----
        col_in = x.permute(0, 2, 1, 3).contiguous()  # [B, G, C, D]
        col_in = col_in.view(B * G, C, D)            # [B*G, C, D]
        
        
        mask_rect = cell_attn_mask[target_idx, :]
        
        q = col_in[:, target_idx, :]
        k = col_in
        v = col_in

        

        col_out, _ = self.col_attn(q,k,v,attn_mask=mask_rect)  # [C, C], broadcast over B*G and heads 
        
        col_updated = col_in.clone()
        col_updated[:, target_idx, :] = col_out
        col_in = col_updated
        
        col_out = self.col_ln(col_in + col_out)

        col_out = col_out.view(B, G, C, D).permute(0, 2, 1, 3).contiguous()
        x = col_out   # [B, C, G, D]
        
        
        '''Let's see if we need to change FFN to accomodate target_idx as well'''
        # ---- FFN ----
        ffn_in = x.view(B * C * G, D)
        ffn_out = self.ffn(ffn_in)
        ffn_out = self.ffn_ln(ffn_in + ffn_out)
        x = ffn_out.view(B, C, G, D)

        return x


class AxialBlockLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        
        self.row_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.row_ln = nn.LayerNorm(d_model)

        self.col_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.col_ln = nn.LayerNorm(d_model)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model),
        )
        self.ffn_ln = nn.LayerNorm(d_model)
        
        
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=True)
        
        self.proj = nn.Linear(d_model, d_model, bias=True)
        
        
        self.cls_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        
        self.cls_ln = nn.LayerNorm(d_model)                

    def forward(self, x: torch.Tensor, cell_attn_mask: torch.Tensor, target_idx, A_hat, epoch, edge_idx, count, tau=1) -> torch.Tensor:
        """
        x: [B, K, G+1, D].
        Position 0 is CLS; positions 1 through G are gene tokens.
        cell_attn_mask: [K, K], with True blocking attention.
        """
        
        B, C, Gp, D = x.shape
        assert D == self.d_model
        assert Gp >= 2, "Need CLS + at least 1 gene token"
        
        C_dst = target_idx.numel()
        dst_idx = torch.arange(C_dst, device=cell_attn_mask.device, dtype=torch.long)
        
        
        Gp = x.size(2)  # CLS + genes
        
        # # # # ---- Row attention over (CLS + genes) within each cell ----
        row_in = x.view(B * C, Gp, D)                    # [B*C, Gp, D]     
        H = 1
        Dh = D // H
        
        # qkv = self.qkv(row_in[:C_dst, :, :])
        qkv = self.qkv(row_in)
        q, k, v = qkv.chunk(3, dim=-1)    
             
        q = q.view(B*C, Gp, H, Dh).transpose(1, 2).contiguous()
        k = k.view(B*C, Gp, H, Dh).transpose(1, 2).contiguous()
        v = v.view(B*C, Gp, H, Dh).transpose(1, 2).contiguous()
        
        
        q_ = q.to(torch.bfloat16)
        k_ = k.to(torch.bfloat16)
        v_ = v.to(torch.bfloat16)      
        
        
        from torch.cuda.amp import autocast    
        try:
            with sdpa_kernel([SDPBackend.FLASH_ATTENTION]):
                with autocast(dtype=torch.bfloat16):
                    row_out = F.scaled_dot_product_attention(q_, k_, v_, attn_mask=None, dropout_p=0.0, is_causal=False)
                    row_out = row_out.transpose(1, 2).contiguous().view(B*C, Gp, D)        
        except Exception as e:
            print("FLASH failed:", type(e).__name__, e)
                           
        # with sdpa_kernel([SDPBackend.FLASH_ATTENTION]):
        # SDPBackend.FLASH_ATTENTION, 
        # This one works - SDPBackend.MATH 
        # # This one works  - SDPBackend.EFFICIENT_ATTENTION
        # with sdpa_kernel([SDPBackend.FLASH_ATTENTION]):
        #     row_out = F.scaled_dot_product_attention(q,k,v, attn_mask=None, dropout_p=0.0, is_causal=False)
                    
        '''This is where flash attention ends'''
        
        
        row_out = self.row_ln(row_in + row_out)
        x = row_out.view(B, C, Gp, D)                    # [B, C, G+1, D]
        
        '''Newly added'''
        # # # # Split CLS and genes
        cls_tok = x[:, :, 0, :]                         # [B, K D]
        genes   = x[:, :, 1:, :]                         # [B, K, G, D]
        G = genes.size(2)
        
        # pdb.set_trace()   

        # ---- Column attention over cells for each gene (CLS excluded) ----
        col_in = genes.permute(0, 2, 1, 3).contiguous()  # [B, G, C, D]
        col_in = col_in.view(B * G, C, D)                # [B*G, C, D]
          
        q = col_in[:, :C_dst, :]
        k = col_in
        v = col_in
        
        
        mask_rect = cell_attn_mask.index_select(0, dst_idx)  # [C_dst, C_src]
     
        col_out, _ = self.col_attn(q, k, v, attn_mask=mask_rect, need_weights=False)
        
        col_out = self.col_ln(col_in[:, :C_dst,:] + col_out)
        
        col_updated = col_in.clone()
        col_updated[:, :C_dst, :] = col_out
        col_out_updated = col_updated
        
        col_out_updated = col_out_updated.view(B, G, C, D).permute(0, 2, 1, 3).contiguous()  # [B, C, G, D]
        x = torch.cat([cls_tok.unsqueeze(2), col_out_updated], dim=2)         # [B, C, G+1, D]
        # x = torch.cat([cls_out_updated.unsqueeze(2), col_out_updated], dim=2)         # [B, C, G+1, D] -> [B, G, C, D]
        x_new = x.clone()   
        # x_graph_prop = x.clone()
        x_dst = x[:, :C_dst, :, :]  # (B, C_dst, Gp, D)

        ffn_in  = x_dst.reshape(B * C_dst * Gp, D)
        ffn_out = self.ffn(ffn_in)
        ffn_out = self.ffn_ln(ffn_in + ffn_out)
        
        ffn_out = ffn_out.reshape(B*C_dst, Gp, D)
        x_new[:, :C_dst, :, :] = ffn_out.reshape(B, C_dst, Gp, D)
            
        return x_new


class AxialTransformer_temp_fix(nn.Module):
    def __init__(self, d_model: int, n_heads: int, num_layers: int, dropout: float = 0.1):
        
        super().__init__()
        self.layers = nn.ModuleList([
            AxialBlockLayer(d_model, n_heads, dropout)
            for _ in range(num_layers)
        ])

    def forward(self, x: torch.Tensor, cell_attn_mask: torch.Tensor, target_idx, A_hat, epoch, edge_idx, count) -> torch.Tensor:
        
        for layer in self.layers:
            x = layer(x, cell_attn_mask, target_idx, A_hat, epoch, edge_idx, count)
        return x
