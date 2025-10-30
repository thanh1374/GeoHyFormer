import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_add, scatter_softmax, scatter_max
from torch_geometric.nn import global_mean_pool, global_max_pool, GlobalAttention
from torch_geometric.data import Data

# Utilities: numerically-stable ops
def atanh_safe(x):
    # atanh(x) = 0.5 * (log1p(x) - log1p(-x))
    return 0.5 * (torch.log1p(x) - torch.log1p(-x))

def clamp_min_eps(x, eps=1e-15):
    return torch.clamp(x, min=eps)

# Hyperbolic maps (stable)
def expmap_tensor(e, c):

    # avoid zero division/nan
    abs_c = torch.clamp(torch.abs(c), min=1e-15)
    sqrt_c = torch.sqrt(abs_c)
    e_norm = e.norm(dim=-1, keepdim=True)
    e_norm = clamp_min_eps(e_norm)
    # for negative c use tanh, positive c use tan
    tanh_part = torch.tanh(sqrt_c * e_norm) * e / (sqrt_c * e_norm)
    tan_part = torch.tan(sqrt_c * e_norm) * e / (sqrt_c * e_norm)
    cond = (c < 0).view(1, 1) if c.dim() == 0 else (c < 0).unsqueeze(-1)
    out = torch.where(cond, tanh_part, tan_part)
    return out

def proj_tensor(x, c, eps=1e-5):
    """
    Project to ball radius (1-eps)/sqrt(|c|)
    """
    abs_c = torch.clamp(torch.abs(c), min=1e-15)
    maxnorm = (1.0 - eps) / torch.sqrt(abs_c)
    norm = x.norm(dim=-1, keepdim=True)
    scale = torch.where(norm > maxnorm, maxnorm / norm, torch.ones_like(norm))
    return x * scale

# Multi-scale Time Encoding
class MultiScaleTimeEncoding(nn.Module):
    def __init__(self, time_dim=32, scales=(1.0, 24.0, 24.0*30.0)):
        """
        time_dim: base dimension (per-scale will be time_dim)
        scales: list of scale factors (seconds->units). We will log1p(dt / scale)
        Output dim = time_dim * len(scales)
        """
        super().__init__()
        assert time_dim % 2 == 0, "time_dim must be even"
        self.scales = scales
        self.time_dim = time_dim
        self.inv_freq = nn.ParameterList()
        for _ in scales:
            inv_f = 1.0 / (10000 ** (torch.arange(0, time_dim, 2).float() / time_dim))
            # register as buffer-like param but keep on module device
            self.inv_freq.append(nn.Parameter(inv_f, requires_grad=False))

    def forward(self, dt):
        # dt: (E,) or (E,1)
        if dt.dim() == 2 and dt.size(1) == 1:
            dt = dt.view(-1)
        embs = []
        device = dt.device
        for s, inv_f in zip(self.scales, self.inv_freq):
            x = torch.log1p(dt / s).unsqueeze(-1).to(device)  # (E,1)
            freqs = x * inv_f.unsqueeze(0).to(device)  # (E, time_dim/2)
            emb = torch.cat([torch.sin(freqs), torch.cos(freqs)], dim=-1)  # (E, time_dim)
            embs.append(emb)
        return torch.cat(embs, dim=-1)  # (E, time_dim * n_scales)

# Dual-space fusion transformer layer (advanced)
class DualHybridLayer(nn.Module):
    def __init__(
        self,
        d_model=128,
        n_heads=8,
        time_dim=32,
        n_edge_types=2,
        dropout=0.1,
        use_rel_qkv=True,
    ):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.time_dim = time_dim

        # base projections
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)

        # relation embeddings for q/k/v (edge-aware)
        self.use_rel_qkv = use_rel_qkv
        if use_rel_qkv:
            self.rel_q = nn.Embedding(n_edge_types, d_model)
            self.rel_k = nn.Embedding(n_edge_types, d_model)
            self.rel_v = nn.Embedding(n_edge_types, d_model)

        # time projection (input dim equals whatever time encoder outputs)
        self.time_proj = nn.Linear(time_dim, d_model)

        # adaptive decay MLP: input [edge_time_scalar, edge_type_emb_mean] -> per-head decay
        self.decay_mlp = nn.Sequential(
            nn.Linear(1 + 1, 64),  # we'll concat edge_time and edge_type scalar proxy
            nn.ReLU(),
            nn.Linear(64, n_heads),
            nn.Tanh()  # outputs in (-1,1), will be scaled with a learnable factor
        )
        self.decay_scale = nn.Parameter(torch.ones(n_heads) * 0.5)  # scale factor

        # hyperbolic per-layer params
        self.c_mag = nn.Parameter(torch.tensor(1.0))  # positive magnitude; c = -abs(c_mag)
        self.hyp_beta = nn.Parameter(torch.ones(n_heads) * 1.0)  # per-head scaling of manifold distance

        # fusion gate: per-head logits -> sigmoid -> alpha in (0,1)
        self.logit_alpha = nn.Parameter(torch.zeros(n_heads))

        # output + FFN
        self.out_proj = nn.Linear(d_model, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout),
        )
        self.dropout = nn.Dropout(dropout)

        # initialization
        self._reset_parameters()

    def _reset_parameters(self):
        # xavier for linear layers
        for m in [self.W_q, self.W_k, self.W_v, self.out_proj, self.time_proj]:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if getattr(m, "bias", None) is not None:
                    nn.init.zeros_(m.bias)
        if self.use_rel_qkv:
            nn.init.xavier_uniform_(self.rel_q.weight)
            nn.init.xavier_uniform_(self.rel_k.weight)
            nn.init.xavier_uniform_(self.rel_v.weight)
        nn.init.zeros_(self.logit_alpha)
        nn.init.constant_(self.hyp_beta, 1.0)
        nn.init.constant_(self.c_mag, 1.0)
        nn.init.constant_(self.decay_scale, 0.5)

    def forward(self, x, edge_index, time_emb, edge_type, edge_time):
        """
        x: (N, d_model)
        edge_index: (2, E)
        time_emb: (E, time_dim)
        edge_type: (E,) long
        edge_time: (E,) float
        """
        src, dst = edge_index
        device = x.device

        # Project QKV & reshape to heads
        Q = self.W_q(x).view(-1, self.n_heads, self.d_head)  # (N, H, Dh)
        K = self.W_k(x).view(-1, self.n_heads, self.d_head)
        V = self.W_v(x).view(-1, self.n_heads, self.d_head)

        q_dst = Q[dst]  # (E, H, Dh)
        k_src = K[src]
        v_src = V[src]

        # edge aware adjustments
        if self.use_rel_qkv:
            rel_q = self.rel_q(edge_type).view(-1, self.n_heads, self.d_head)
            rel_k = self.rel_k(edge_type).view(-1, self.n_heads, self.d_head)
            rel_v = self.rel_v(edge_type).view(-1, self.n_heads, self.d_head)
            q_dst = q_dst + rel_q
            k_src = k_src + rel_k
            v_src = v_src + rel_v

        # time projection into d_model and split heads
        t_proj = self.time_proj(time_emb).view(-1, self.n_heads, self.d_head)
        k_src = k_src + t_proj  # inject time into keys

        # EUCLIDEAN logits
        euclid_logits = (q_dst * k_src).sum(dim=-1) / math.sqrt(self.d_head)  # (E, H)

        # Adaptive decay per-edge per-head
        # create a small scalar proxy for edge_type (0..n_edge_types-1) normalized
        edge_type_scalar = edge_type.float().unsqueeze(-1) / max(1.0, float(max(1, edge_type.max().item()+1)))
        decay_input = torch.cat([edge_time.unsqueeze(-1), edge_type_scalar.to(device)], dim=-1)  # (E,2)
        decay_raw = self.decay_mlp(decay_input)  # (E, H) in (-1,1)
        decay = torch.exp(self.decay_scale.view(1, -1).to(device) * decay_raw)  # positive scaling
        euclid_logits = euclid_logits * decay

        # HYPERBOLIC logits: map x -> manifold once per layer
        c = -torch.abs(self.c_mag)  # negative curvature
        m = expmap_tensor(x, c)
        m = proj_tensor(m, c)
        m_src = m[src]
        m_dst = m[dst]
        man_dist = (m_src - m_dst).norm(dim=-1).unsqueeze(-1)  # (E,1)
        beta = self.hyp_beta.view(1, -1).to(device)
        hyp_logits = - beta * man_dist  # negative distance

        # optionally modulate hyp_logits with same decay to account for time
        hyp_logits = hyp_logits * decay

        # Per-head fusion alpha
        alpha = torch.sigmoid(self.logit_alpha.view(1, -1).to(device))  # (1,H)
        logits = alpha * euclid_logits + (1.0 - alpha) * hyp_logits  # (E,H)

        # attention softmax grouped by dst
        attn = scatter_softmax(logits, dst, dim=0)
        attn = self.dropout(attn)

        weighted_v = v_src * attn.unsqueeze(-1)  # (E,H,Dh)
        agg = scatter_add(weighted_v, dst, dim=0, dim_size=x.size(0))  # (N,H,Dh)
        agg = agg.view(-1, self.d_model)  # (N, d_model)

        out = self.out_proj(agg)
        x = self.norm1(x + self.dropout(out))
        x = self.norm2(x + self.ffn(x))
        return x

# Stack + Model with robust pooling & root selection
class DualHybridStack(nn.Module):
    def __init__(self, num_layers=3, d_model=128, n_heads=8, time_dim=32, n_edge_types=2, dropout=0.1):
        super().__init__()
        # Use multi-scale time encoder with 3 scales (seconds, hours, days)
        self.time_enc = MultiScaleTimeEncoding(time_dim=time_dim, scales=(1.0, 3600.0, 3600.0*24.0))
        ts_out_dim = time_dim * 3
        self.layers = nn.ModuleList([
            DualHybridLayer(d_model=d_model, n_heads=n_heads, time_dim=ts_out_dim,
                            n_edge_types=n_edge_types, dropout=dropout)
            for _ in range(num_layers)
        ])

    def forward(self, x, edge_index, edge_time, edge_type):
        # compute time embeddings once (expects edge_time float tensor)
        time_emb = self.time_enc(edge_time)
        for layer in self.layers:
            x = layer(x, edge_index, time_emb, edge_type, edge_time)
        return x

class GHP(nn.Module):
    def __init__(self, in_dim, d_model=128, num_layers=3, n_heads=8, time_dim=32,
                 n_edge_types=2, dropout=0.1, num_classes=2):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(in_dim, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.gtrans = DualHybridStack(num_layers=num_layers, d_model=d_model, n_heads=n_heads,
                                      time_dim=time_dim, n_edge_types=n_edge_types, dropout=dropout)
        self.att_pool = GlobalAttention(gate_nn=nn.Sequential(nn.Linear(d_model, 1)))
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_classes)
        )
        self.view_weights = nn.Parameter(torch.ones(4))
        self.root_selector = nn.Linear(d_model, 1)

    def forward(self, data: Data):
        x = data.x
        edge_index = data.edge_index
        edge_time = data.edge_time
        edge_type = data.edge_type
        batch = data.batch

        h = self.input_proj(x)
        h = self.gtrans(h, edge_index, edge_time, edge_type)

        hg_mean = global_mean_pool(h, batch)
        hg_max = global_max_pool(h, batch)
        hg_att = self.att_pool(h, batch)

        scores = self.root_selector(h).squeeze(-1)
        _, argmax = scatter_max(scores, batch, dim=0)
        root_idx = argmax.long()
        h_root = h[root_idx]

        views = torch.stack([h_root, hg_mean, hg_max, hg_att], dim=1)
        weights = torch.softmax(self.view_weights, dim=0)
        final = (views * weights.view(1, -1, 1)).sum(dim=1)
        logits = self.classifier(final)
        return logits