import torch
import torch.nn as nn
import torch.nn.functional as F

class GAT_cls(nn.Module):
    """
    Additive GAT over cell graph, independently for each gene.

    Input:
        x: [C, G, D]
        edge_index: [2, E]
            edge_index[0] = dst
            edge_index[1] = src

    Output:
        out: [C, G, D_out]
    """
    def __init__(
        self,
        d_in,
        d_out,
        include_self=False,
        residual=True,
        use_layernorm=False,
        negative_slope=0.2,
        eps=1e-8,
    ):
        # import pdb
        # pdb.set_trace()
        
        
        super().__init__()
        self.d_in = d_in
        self.d_out = d_out
        self.include_self = include_self
        self.residual = residual
        self.use_layernorm = use_layernorm
        self.negative_slope = negative_slope
        self.eps = eps

        # Shared linear transform W
        self.cls_proj = nn.Linear(d_in, d_out, bias=False)

        # Additive attention vector a
        self.attn = nn.Linear(2 * d_out, 1, bias=False)

        # Residual projection if needed
        if residual and d_in != d_out:
            self.res_proj = nn.Linear(d_in, d_out, bias=False)
        else:
            self.res_proj = None

        if use_layernorm:
            self.ln = nn.LayerNorm(d_out)

    def forward(self, x, edge_index, C_dst, epoch=None, return_attention=False):
        """
        x: [C, D_in]  # CLS embeddings only
        edge_index: [2, E]
            edge_index[0] = dst
            edge_index[1] = src

        returns:
            out: [C_dst, D_out]
        """
    
        # import pdb
        # pdb.set_trace()

        C = x.shape[0]
        device = x.device
        dtype = x.dtype

        dst = edge_index[0].to(device)
        src = edge_index[1].to(device)

        h = self.cls_proj(x)  # [C, D_out]

        h_dst = h[dst]        # [E, D_out]
        h_src = h[src]        # [E, D_out]

        cat = torch.cat([h_dst, h_src], dim=-1)  # [E, 2D_out]

        e = self.attn(
            F.leaky_relu(cat, negative_slope=self.negative_slope)
        ).squeeze(-1)  # [E]

        alpha = self.segment_softmax_cls(e, dst, epoch, num_nodes=C)  # [E]

        msg = alpha[:, None] * h_src  # [E, D_out]

        out = torch.zeros(C_dst, self.d_out, device=device, dtype=dtype)
        out.index_add_(0, dst, msg)

        if self.residual:
            res = x[:C_dst] if self.res_proj is None else self.res_proj(x[:C_dst])
            out = out + res

        if self.use_layernorm:
            out = self.ln(out)

        if return_attention:
            return out, alpha, dst, src

        return out
    
    
    def segment_softmax_cls(self, scores, dst, epoch, num_nodes):
        # import pdb
        # pdb.set_trace()
        
        alpha = torch.zeros_like(scores)

        unique_dst = torch.unique(dst)
        for i in unique_dst:
            # if epoch == 10:
            #     import pdb
            #     pdb.set_trace()
            
            mask = (dst == i)
            alpha[mask] = torch.softmax(scores[mask], dim=0)

        return alpha