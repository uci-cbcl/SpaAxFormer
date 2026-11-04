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


# import pdb
# pdb.set_trace()

# try:
#     from flash_attn.flash_attention import FlashMHA

#     flash_attn_available = True
# except ImportError:
#     import warnings

#     warnings.warn("flash_attn is not installed")
#     flash_attn_available = False

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
        # import pdb
        # pdb.set_trace()
        
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
        # import pdb
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
        # import pdb
        # pdb.set_trace()
        
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
        
        # import pdb
        # pdb.set_trace()
        
        mask_rect = cell_attn_mask[target_idx, :]
        
        q = col_in[:, target_idx, :]
        k = col_in
        v = col_in

        
        # import pdb
        # pdb.set_trace()

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

        # self.graph_prop = CellGraphPropagation()
        
        # self.graph_prop = CellWeightGraphPropagation()
        

    def forward(self, x: torch.Tensor, cell_attn_mask: torch.Tensor, target_idx, A_hat, epoch, edge_idx, count, tau=1) -> torch.Tensor:
        """
        x: [B, K, G+1, D].
        Position 0 is CLS; positions 1 through G are gene tokens.
        cell_attn_mask: [K, K], with True blocking attention.
        """
        
        B, C, Gp, D = x.shape
        assert D == self.d_model
        assert Gp >= 2, "Need CLS + at least 1 gene token"
        
        # import pdb
        # pdb.set_trace()
        
        
        '''This is also muted'''
        C_dst = target_idx.numel()
        dst_idx = torch.arange(C_dst, device=cell_attn_mask.device, dtype=torch.long)
        
        # import pdb
        # pdb.set_trace()
        
        Gp = x.size(2)  # CLS + genes
        # row_mask = torch.zeros(Gp, Gp, dtype=torch.bool, device=x.device)
        # row_mask[1:, 0] = True
        
        # # # # ---- Row attention over (CLS + genes) within each cell ----
        row_in = x.view(B * C, Gp, D)                    # [B*C, Gp, D]
        '''This is with CLS'''
        # row_out, _ = self.row_attn(row_in, row_in, row_in, attn_mask=row_mask, need_weights=False) # Not sure about this need_weights component
        '''This is without CLS - we do not use this for flash attention. need_weights=False makes speed faster???'''
        # row_out, _ = self.row_attn(row_in, row_in, row_in, need_weights=False) # Not sure about this need_weights component
        
        # import pdb
        # pdb.set_trace()
        
        # print("Hi")
        '''
        This is how we run flash attention. I need to revise the code little bit
        check if CLS is used or not as well - Flash Attention do not run with mask
        '''
        # Todo: Give Error if attention  mask used
        
        H = 1
        Dh = D // H
        
        # qkv = self.qkv(row_in[:C_dst, :, :])
        qkv = self.qkv(row_in)
        q, k, v = qkv.chunk(3, dim=-1)    
        
        # q = q.view(B*C_dst, Gp, H, Dh).transpose(1, 2).contiguous()
        # k = k.view(B*C_dst, Gp, H, Dh).transpose(1, 2).contiguous()
        # v = v.view(B*C_dst, Gp, H, Dh).transpose(1, 2).contiguous()
        
        q = q.view(B*C, Gp, H, Dh).transpose(1, 2).contiguous()
        k = k.view(B*C, Gp, H, Dh).transpose(1, 2).contiguous()
        v = v.view(B*C, Gp, H, Dh).transpose(1, 2).contiguous()
        
        
        q_ = q.to(torch.bfloat16)
        k_ = k.to(torch.bfloat16)
        v_ = v.to(torch.bfloat16)      
        
        # print("q:", q_.shape, q_.dtype, q_.device, q_.is_contiguous())
        # print("k:", k_.shape, k_.dtype, k_.device, k_.is_contiguous())
        # print("v:", v_.shape, v_.dtype, v_.device, v_.is_contiguous())
        
        from torch.cuda.amp import autocast
        
        # This makes really fast probably: float16 + flash attention
        
        try:
            with sdpa_kernel([SDPBackend.FLASH_ATTENTION]):
                with autocast(dtype=torch.bfloat16):
                    row_out = F.scaled_dot_product_attention(q_, k_, v_, attn_mask=None, dropout_p=0.0, is_causal=False)
                    # print("FLASH worked")
                    '''Check this part'''
                    # row_out = row_out.transpose(1, 2).contiguous().view(B*C_dst, Gp, D)
                    row_out = row_out.transpose(1, 2).contiguous().view(B*C, Gp, D)
                    # row_out = self.proj(row_out)
        
        except Exception as e:
            print("FLASH failed:", type(e).__name__, e)
                           
        # with sdpa_kernel([SDPBackend.FLASH_ATTENTION]):
        # SDPBackend.FLASH_ATTENTION, 
        # This one works - SDPBackend.MATH 
        # # This one works  - SDPBackend.EFFICIENT_ATTENTION
        # with sdpa_kernel([SDPBackend.FLASH_ATTENTION]):
        #     row_out = F.scaled_dot_product_attention(q,k,v, attn_mask=None, dropout_p=0.0, is_causal=False)
                    
        '''This is where flash attention ends'''
        
            # import pdb
            # pdb.set_trace()
        
        row_out = self.row_ln(row_in + row_out)
        x = row_out.view(B, C, Gp, D)                    # [B, C, G+1, D]
        
        # row_updated = row_in.clone()
        # row_updated[:C_dst, :, :] = row_out
        # row_out_updated = row_updated
        
        # row_out_updated = row_out
        
        # row_out_updated = row_out_updated.view(B, C, Gp, D)
        
        # x = row_out_updated
        
        '''Newly added'''
        # # # # Split CLS and genes
        cls_tok = x[:, :, 0, :]                         # [B, K D]
        genes   = x[:, :, 1:, :]                         # [B, K, G, D]
        G = genes.size(2)
        
        # import pdb
        # pdb.set_trace()   

        # ---- Column attention over cells for each gene (CLS excluded) ----
        # if epoch >= 50:
        # col_in = x.permute(0, 2, 1, 3).contiguous()  # [B, G, Cp, D]
        # col_in = col_in.view(B * Gp, C, D) 
        # else:
        col_in = genes.permute(0, 2, 1, 3).contiguous()  # [B, G, C, D]
        col_in = col_in.view(B * G, C, D)                # [B*G, C, D]
        
        # import pdb
        # pdb.set_trace()
        
        # cls_in = cls_tok.permute(0,2,1).contiguous()
        # cls_in = cls_in.view(B*1, C, D)
        
        
        # import pdb
        # pdb.set_trace()
        
        q = col_in[:, :C_dst, :]
        k = col_in
        v = col_in
        
        # import pdb
        # pdb.set_trace()
        
        mask_rect = cell_attn_mask.index_select(0, dst_idx)  # [C_dst, C_src]
        # A_hat = A_hat.index_select(0, dst_idx)
        
        # q = col_in
        # k = col_in
        # v = col_in
        
        # import pdb
        # pdb.set_trace()
        
        # mask_sq = cell_attn_mask
        
        # import pdb
        # pdb.set_trace()
    
        col_out, _ = self.col_attn(q, k, v, attn_mask=mask_rect, need_weights=False)
        
        col_out = self.col_ln(col_in[:, :C_dst,:] + col_out)
        
        # col_out, _ = self.col_attn(q, k, v, attn_mask=mask_sq, need_weights=False) 
        # col_out = self.col_ln(col_in + col_out)
        
        col_updated = col_in.clone()
        col_updated[:, :C_dst, :] = col_out
        col_out_updated = col_updated
        
        # if epoch >= 50:
        # col_out_updated = col_out_updated.view(B, Gp, C, D).permute(0, 2, 1, 3).contiguous()  # [B, C, Gp, D]
        # else:
        col_out_updated = col_out_updated.view(B, G, C, D).permute(0, 2, 1, 3).contiguous()  # [B, C, G, D]
        
        # col_out_updated = col_out.view(B, Gp, C, D).permute(0, 2, 1, 3).contiguous()  # [B, C, Gp, D]
        
        # import pdb
        # pdb.set_trace()
        
        # import pdb
        # pdb.set_trace()
        
        # pick cls token 
        # cls_in = col_out_updated[:, :, 0, :]
        
        # if epoch >= 50:    
        #     q_c = cls_in[:, :C_dst, :]
        #     k_c = cls_in
        #     v_c = cls_in

        #     cls_out, _ = self.cls_attn(q_c, k_c, v_c, attn_mask=mask_rect, need_weights=False)
        #     cls_out = self.cls_ln(cls_in[:, :C_dst, :] + cls_out)
            
        #     cls_updated = cls_in.clone()
        #     cls_updated[:, :C_dst, :] = cls_out
        #     cls_out_updated = cls_updated

        
        # if epoch in {5, 75, 125} and count == 0:
        #     self.col_attn.eval()
        #     with torch.no_grad():
        #         import pdb
        #         pdb.set_trace()
        #         col_out, attn_w = self.col_attn(
        #         q, k, v,
        #         attn_mask=mask_rect,
        #         need_weights=True,
        #         average_attn_weights=False
        #         ) 
        
            # self.cls_attn.eval()
            # with torch.no_grad():
            #     col_out, attn_w = self.cls_attn(
            #     q_c, k_c, v_c,
            #     need_weights=True,
            #     average_attn_weights=False
            #     ) 
                
            #     import pdb
            #     pdb.set_trace()
                
            #     col_out_2, attn_w_2 = self.cls_attn(
            #     q_c, k_c, v_c,
            #     attn_mask=mask_rect,
            #     need_weights=True,
            #     average_attn_weights=False
            #     )   

            #     print(attn_w.shape)  # e.g. [N, H, L, S] or [H, L, S] depending on your input form
                
            #     # import pdb
            #     # pdb.set_trace()

            #     # example: first batch item, first head
            #     w = attn_w[0, 0]          # [L, S]
            #     print(w.sum(dim=-1))      # should be ~1 per query row
        
        # cls_out = self.cls_ln(cls_in[:, :C_dst, :] + cls_out)
        
        # # cls_out = self.cls_ln(cls_in[:, :C_dst, :])
        
        # cls_updated = cls_in.clone()
        # cls_updated[:, :C_dst, :] = cls_out
        # cls_out_updated = cls_updated
        
        # import pdb
        # pdb.set_trace()
        
        
        # cls_out_updated = cls_out_updated.view(B, C, D).permute(0, 2, 1).contiguous()  # [B, C, G, D]
        
        # import pdb
        # pdb.set_trace()

        # Reattach CLS - CLS is already attached now
        # if epoch >= 50:
        #     x = torch.cat([cls_out_updated.unsqueeze(2), col_out_updated], dim=2)
        # else:
        #     x = torch.cat([cls_tok.unsqueeze(2), col_out_updated], dim=2)  
        
        # if epoch >= 50:
        # x = col_out_updated
        # else:
        x = torch.cat([cls_tok.unsqueeze(2), col_out_updated], dim=2)         # [B, C, G+1, D]
        # x = torch.cat([cls_out_updated.unsqueeze(2), col_out_updated], dim=2)         # [B, C, G+1, D] -> [B, G, C, D]
        '''cls is also updated together with col_out'''
        # x = col_out_updated
        
        # row_in = x.view(B * C, Gp, D)                    # [B*C, G+1, D]
        # row_out, _ = self.row_attn(row_in, row_in, row_in) # Not sure about this need_weights component
        # # need_weights=False
        # row_out = self.row_ln(row_in + row_out)
        # x = row_out.view(B, C, Gp, D)                    # [B, C, G+1, D]
        
        '''Was already muted (up to here)'''
        
        '''This is where concat should happen'''
        x_new = x.clone()   
        # x_graph_prop = x.clone()
        
        # x_sub = self.graph_prop(x.reshape(B*C, Gp, D), A_hat, C_dst)
        # x_sub = self.graph_prop(x.reshape(B*C, G, D), A_hat, C_dst)
        
        # x_dst = x_sub
        
        # x_graph_prop[:, :C_dst, :, :] = x_sub.reshape(B, C_dst, Gp, D)
        
        x_dst = x[:, :C_dst, :, :]  # (B, C_dst, Gp, D)

        ffn_in  = x_dst.reshape(B * C_dst * Gp, D)
        ffn_out = self.ffn(ffn_in)
        ffn_out = self.ffn_ln(ffn_in + ffn_out)
        
        # import pdb
        # pdb.set_trace()
        
        '''newly added - we add new experiment here'''
        ffn_out = ffn_out.reshape(B*C_dst, Gp, D)
        x_new[:, :C_dst, :, :] = ffn_out.reshape(B, C_dst, Gp, D)
        
        # ffn_in  = x.reshape(B * C * Gp, D)
        # ffn_out = self.ffn(ffn_in)
        # ffn_out = self.ffn_ln(ffn_in + ffn_out)
        # x = ffn_out.reshape(B, C, Gp, D)
        
        return x_new


class AxialTransformer_temp_fix(nn.Module):
    def __init__(self, d_model: int, n_heads: int, num_layers: int, dropout: float = 0.1):
        
        # import pdb
        # pdb.set_trace()
        
        super().__init__()
        self.layers = nn.ModuleList([
            AxialBlockLayer(d_model, n_heads, dropout)
            for _ in range(num_layers)
        ])

    def forward(self, x: torch.Tensor, cell_attn_mask: torch.Tensor, target_idx, A_hat, epoch, edge_idx, count) -> torch.Tensor:
        # import pdb
        # pdb.set_trace()
        
        for layer in self.layers:
            x = layer(x, cell_attn_mask, target_idx, A_hat, epoch, edge_idx, count)
        return x
