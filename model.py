"""PRISM² model: patient-centric heterogeneous graph network, task heads, losses, EWC and EMA."""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import NUM_LOS, NUM_REL, TASK_KIND, log, to_dev

# relation-id ranges handled by each local (intra-modality) attention block
LOCAL_REL = {'ehr': (0, 1), 'cxr': (2, 3), 'rel': (4, 7)}


# --------------------------------------------------------------------------- #
# Graph network
# --------------------------------------------------------------------------- #
class FeatureEncoder(nn.Module):
    """Projects every modality to a common width d and lays nodes out per patient:
    [patient | T time steps | 6 CXR regions | 1 note | K concepts | 1 demographics]."""

    def __init__(self, d, dims, dropout):
        super().__init__()

        def mlp(i):
            return nn.Sequential(nn.Linear(i, d), nn.ReLU(), nn.Dropout(dropout), nn.Linear(d, d))

        self.ehr = mlp(dims['ehr'])
        self.txt = mlp(dims['text'])
        self.ent = mlp(dims['ent'])
        self.demo = mlp(dims['demo'])
        self.img = nn.Sequential(nn.Linear(dims['cxr'], 768), nn.LayerNorm(768), nn.GELU(),
                                 nn.Dropout(dropout), nn.Linear(768, d))
        self.kind_emb = nn.Embedding(3, d)       # concept kind: symptom / disease / treatment
        self.time_pos = nn.Linear(1, d)
        self.region_emb = nn.Embedding(6, d)
        self.patient = nn.Parameter(torch.zeros(1, d))
        self.d = d

    def forward(self, b):
        B, T = b['B'], b['T']
        dev = b['ehr'].device
        pos = torch.arange(T, device=dev, dtype=torch.float32).view(T, 1) / max(T - 1, 1)
        h_pat = self.patient.expand(B, 1, self.d)
        h_ehr = self.ehr(b['ehr']) + self.time_pos(pos).unsqueeze(0)
        h_img = self.img(b['cxr'].reshape(B * 6, -1)).view(B, 6, -1) \
            + self.region_emb(torch.arange(6, device=dev)).unsqueeze(0)
        h_txt = self.txt(b['txt']).unsqueeze(1)
        h_con = self.ent(b['cfe']) + self.kind_emb(b['ckind'])
        h_dem = self.demo(b['demo']).unsqueeze(1)
        dt = h_ehr.dtype
        X = torch.cat([h_pat.to(dt), h_ehr, h_img.to(dt), h_txt.to(dt),
                       h_con.to(dt), h_dem.to(dt)], 1)
        return X.reshape(B * b['n'], self.d)


class RelationalGAT(nn.Module):
    """Local intra-modality attention: h_i' = sigma(W_s h_i + sum_j a_ij W_m h_j).
    Attention logits use the source/target messages plus a relation embedding and an edge-attribute term.
    Unavailable (missing-modality) source nodes are masked out through `avail`."""

    def __init__(self, d, heads, num_rel, dropout, edge_dim=2):
        super().__init__()
        assert d % heads == 0
        self.h, self.dh, self.d = heads, d // heads, d
        self.W = nn.Linear(d, d, bias=False)          # W_m
        self.self_lin = nn.Linear(d, d)               # W_s
        self.rel = nn.Embedding(num_rel, d)
        self.disp = nn.Linear(edge_dim, d)
        self.att = nn.Parameter(torch.empty(heads, 3 * self.dh))
        nn.init.xavier_uniform_(self.att)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, edge_index, edge_type, edge_attr, avail):
        x = x.float()
        src, dst = edge_index[0], edge_index[1]
        Wx = self.W(x).view(-1, self.h, self.dh)
        r = (self.rel(edge_type) + self.disp(edge_attr)).view(-1, self.h, self.dh)
        e = (torch.cat([Wx[src], Wx[dst], r], -1) * self.att.unsqueeze(0)).sum(-1)
        e = F.leaky_relu(e, 0.2) + torch.log(avail[src].clamp_min(1e-9)).unsqueeze(-1)
        neg = torch.finfo(e.dtype).min / 4
        mx = torch.full((x.size(0), self.h), neg, device=x.device, dtype=e.dtype)
        mx = mx.scatter_reduce(0, dst.view(-1, 1).expand(-1, self.h), e, 'amax', include_self=True)
        a = torch.exp(e - mx[dst])
        den = torch.zeros(x.size(0), self.h, device=x.device, dtype=a.dtype).index_add_(0, dst, a)
        a = self.drop(a / den[dst].clamp_min(1e-16))
        msg = (Wx[src] * a.unsqueeze(-1)).reshape(-1, self.d)
        agg = torch.zeros_like(x).index_add_(0, dst, msg)
        return F.elu(self.self_lin(x) + agg)


class HGTLayer(nn.Module):
    """Global heterogeneous attention: relation-specific Q/K/V (block-diagonal per head),
    softmax over the neighbours of each (target node, relation), summed over relations."""

    def __init__(self, d, heads, num_rel, dropout):
        super().__init__()
        assert d % heads == 0
        self.h, self.dh, self.d, self.R = heads, d // heads, d, num_rel
        self.Wq = nn.Parameter(torch.randn(num_rel, heads, self.dh, self.dh) * self.dh ** -0.5)
        self.Wk = nn.Parameter(torch.randn(num_rel, heads, self.dh, self.dh) * self.dh ** -0.5)
        self.Wv = nn.Parameter(torch.randn(num_rel, heads, self.dh, self.dh) * self.dh ** -0.5)
        self.out = nn.Linear(d, d)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, edge_index, edge_type, avail):
        x = x.float()
        src, dst = edge_index[0], edge_index[1]
        E, N = src.numel(), x.size(0)
        xh = x.view(N, self.h, self.dh)
        q = x.new_zeros(E, self.h, self.dh)
        k = x.new_zeros(E, self.h, self.dh)
        v = x.new_zeros(E, self.h, self.dh)
        for r in range(self.R):
            idx = (edge_type == r).nonzero(as_tuple=True)[0]
            if idx.numel() == 0:
                continue
            xs, xd = xh[src[idx]], xh[dst[idx]]
            q[idx] = torch.einsum('ehd,hdf->ehf', xd, self.Wq[r]).float()
            k[idx] = torch.einsum('ehd,hdf->ehf', xs, self.Wk[r]).float()
            v[idx] = torch.einsum('ehd,hdf->ehf', xs, self.Wv[r]).float()
        s = (q * k).sum(-1) / math.sqrt(self.dh)
        s = s + torch.log(avail[src].clamp_min(1e-9)).unsqueeze(-1)
        g = dst * self.R + edge_type                       # group = (target node, relation)
        G = N * self.R
        neg = torch.finfo(s.dtype).min / 4
        mx = torch.full((G, self.h), neg, device=x.device, dtype=s.dtype)
        mx = mx.scatter_reduce(0, g.view(-1, 1).expand(-1, self.h), s, 'amax', include_self=True)
        a = torch.exp(s - mx[g])
        den = torch.zeros(G, self.h, device=x.device, dtype=a.dtype).index_add_(0, g, a)
        a = a / den[g].clamp_min(1e-16)
        a = self.drop(a) * avail[src].unsqueeze(-1)        # missing source nodes send nothing
        msg = (v * a.unsqueeze(-1)).reshape(-1, self.d)
        agg = torch.zeros_like(x).index_add_(0, dst, msg)
        return self.out(agg)


class Backbone(nn.Module):
    def __init__(self, cfg, dims):
        super().__init__()
        d = cfg.HIDDEN_DIM
        self.enc = FeatureEncoder(d, dims, cfg.DROPOUT)
        self.local = nn.ModuleDict({k: RelationalGAT(d, cfg.NUM_HEADS, NUM_REL, cfg.DROPOUT)
                                    for k in LOCAL_REL})
        self.local_norm = nn.ModuleDict({k: nn.LayerNorm(d) for k in LOCAL_REL})
        self.type_emb = nn.Embedding(6, d)
        self.layers = nn.ModuleList([HGTLayer(d, cfg.NUM_HEADS, NUM_REL, cfg.DROPOUT)
                                     for _ in range(cfg.NUM_LAYERS)])
        self.norms = nn.ModuleList([nn.LayerNorm(d) for _ in range(cfg.NUM_LAYERS)])
        self.dropout = cfg.DROPOUT
        self.edge_dropout = cfg.GRAPH_DROPOUT

    def forward(self, b):
        x = self.enc(b).float()
        ei, et, ea, av = b['edge_index'], b['edge_type'], b['edge_attr'], b['avail']
        if self.training and self.edge_dropout > 0:
            keep = torch.rand(ei.size(1), device=ei.device) > self.edge_dropout
            ei, et, ea = ei[:, keep], et[keep], ea[keep]
        # (1) local intra-modality propagation (temporal EHR chain, CXR regions, concept relations)
        for key, (lo, hi) in LOCAL_REL.items():
            m = (et >= lo) & (et <= hi)
            if not bool(m.any()):
                continue
            h = self.local[key](x, ei[:, m], et[m], ea[m], av)
            upd = torch.zeros(x.size(0), 1, dtype=torch.bool, device=x.device)
            upd[ei[1, m]] = True
            x = torch.where(upd, self.local_norm[key](
                x + F.dropout(h, p=self.dropout, training=self.training)), x)
        # (2) global heterogeneous reasoning over the patient-centric graph
        x = x + self.type_emb(b['node_type'])
        for lay, nrm in zip(self.layers, self.norms):
            h = lay(x, ei, et, av)
            x = nrm(x + F.dropout(h, p=self.dropout, training=self.training))
        return x


class GatedFusion(nn.Module):
    def __init__(self, in_dim, out_dim, dropout):
        super().__init__()
        self.gate = nn.Sequential(nn.Linear(in_dim, out_dim), nn.Sigmoid())
        self.map = nn.Sequential(nn.Linear(in_dim, out_dim), nn.LayerNorm(out_dim),
                                 nn.ReLU(), nn.Dropout(dropout))

    def forward(self, x):
        return self.map(x) * self.gate(x)


class TaskHead(nn.Module):
    def __init__(self, d, dropout, out):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, d), nn.ReLU(), nn.Dropout(dropout), nn.Linear(d, out))

    def forward(self, x):
        return self.net(x)


class Heads(nn.Module):
    """Availability-masked pooling of each modality's nodes -> gated fusion -> six task heads."""

    def __init__(self, cfg, dims_out):
        super().__init__()
        d = cfg.HIDDEN_DIM
        self.fuse = GatedFusion(d * 5, d, cfg.DROPOUT)       # [ehr || cxr || txt || rel || demo]
        self.IHM = TaskHead(d, cfg.DROPOUT, 1)
        self.REA = TaskHead(d, cfg.DROPOUT, 1)
        self.DEC = TaskHead(d, cfg.DROPOUT, 1)
        self.LOS = TaskHead(d, cfg.DROPOUT, NUM_LOS)
        self.PHE = TaskHead(d, cfg.DROPOUT, dims_out['PHE'])
        self.DIA = TaskHead(d, cfg.DROPOUT, dims_out['DIA'])

    def pool(self, x, b):
        B, n, off, T, K = b['B'], b['n'], b['off'], b['T'], b['K']
        X = x.view(B, n, -1)
        av = b['avail'].view(B, n, 1)

        def mean(lo, hi):
            return (X[:, lo:hi] * av[:, lo:hi]).sum(1) / av[:, lo:hi].sum(1).clamp_min(1.0)

        h_ehr = mean(off['T'], off['T'] + T)
        h_cxr = mean(off['I'], off['I'] + 6)
        h_rel = mean(off['C'], off['C'] + K)
        h_txt = X[:, off['N']] * av[:, off['N']]
        h_demo = X[:, off['D']] * av[:, off['D']]
        return torch.cat([h_ehr, h_cxr, h_txt, h_rel, h_demo], -1)

    def forward(self, x, b):
        z = self.fuse(self.pool(x, b))
        return {'IHM': self.IHM(z).squeeze(-1), 'REA': self.REA(z).squeeze(-1),
                'DEC': self.DEC(z).squeeze(-1), 'LOS': self.LOS(z),
                'PHE': self.PHE(z), 'DIA': self.DIA(z)}


class PatientGraphNet(nn.Module):
    def __init__(self, cfg, dims, dims_out):
        super().__init__()
        self.backbone = Backbone(cfg, dims)
        self.heads = Heads(cfg, dims_out)

    def forward(self, b):
        return self.heads(self.backbone(b), b)


# --------------------------------------------------------------------------- #
# Losses
# --------------------------------------------------------------------------- #
class AsymmetricLoss(nn.Module):
    def __init__(self, gp=1.0, gn=3.0, pos_weights=None, eps=1e-8):
        super().__init__()
        self.gp, self.gn, self.pw, self.eps = gp, gn, pos_weights, eps

    def forward(self, logits, t):
        p = torch.sigmoid(logits.float())
        t = t.float()
        pos = -t * torch.pow(1 - p, self.gp) * torch.log(p + self.eps)
        if self.pw is not None:
            pos = pos * self.pw.to(logits.device)
        neg = -(1 - t) * torch.pow(p, self.gn) * torch.log(1 - p + self.eps)
        return (pos + neg).mean()


class DiceLoss(nn.Module):
    def forward(self, logits, t):
        p = torch.sigmoid(logits.float())
        t = t.float()
        return 1 - (2 * (p * t).sum() + 1.0) / (p.sum() + t.sum() + 1.0)


class FocalLoss(nn.Module):
    """Binary / multi-label focal loss with class weight alpha (scalar or per-label tensor)."""

    def __init__(self, gamma=2.0, alpha=None):
        super().__init__()
        self.gamma, self.alpha = gamma, alpha

    def forward(self, logits, t):
        x, t = logits.float(), t.float()
        ce = F.binary_cross_entropy_with_logits(x, t, reduction='none')
        loss = (1 - torch.exp(-ce)) ** self.gamma * ce
        if self.alpha is not None:
            a = torch.as_tensor(self.alpha, device=x.device, dtype=torch.float32)
            loss = (a * t + (1 - a) * (1 - t)) * loss
        return loss.mean()


class CombinedLoss(nn.Module):
    """IHM: lambda_f * Focal + lambda_d * Dice."""

    def __init__(self, fw, dw, gamma, alpha):
        super().__init__()
        self.f, self.d, self.fw, self.dw = FocalLoss(gamma, alpha), DiceLoss(), fw, dw

    def forward(self, logits, t):
        return self.fw * self.f(logits, t) + self.dw * self.d(logits, t)


class SoftOrdinalCE(nn.Module):
    """Cross-entropy against a Gaussian-smoothed ordinal target (sigma = smoothing across bins)."""

    def __init__(self, num_classes=NUM_LOS, sigma=1.0, weights=None):
        super().__init__()
        self.k, self.sigma, self.w = num_classes, sigma, weights

    def forward(self, logits, t):
        k = torch.arange(self.k, device=logits.device).float().view(1, -1)
        soft = torch.exp(-0.5 * ((k - t.view(-1, 1).float()) / self.sigma) ** 2)
        soft = soft / soft.sum(1, keepdim=True)
        l = -(soft * F.log_softmax(logits.float(), -1)).sum(-1)
        if self.w is not None:
            l = l * self.w.to(logits.device)[t.long()]
        return l.mean()


def make_criterion(task, cfg, y=None):
    """y: numpy label array of the training split (used only for class-imbalance weights)."""
    if task == 'IHM':
        alpha = 0.75
        if y is not None:
            alpha = float(1.0 - np.mean(y))
            log(f'  IHM focal alpha {alpha:.3f} (pos {int(np.sum(y))}/{len(y)})')
        return CombinedLoss(cfg.MORT_FOCAL_WEIGHT, cfg.MORT_DICE_WEIGHT, cfg.FOCAL_GAMMA, alpha)
    if task == 'REA':
        return AsymmetricLoss(cfg.GAMMA_POS, cfg.GAMMA_NEG)
    if task == 'DEC':
        return nn.BCEWithLogitsLoss()
    if task == 'PHE':
        alpha = None
        if y is not None:
            alpha = torch.tensor(1.0 - y.mean(0), dtype=torch.float32)
        return FocalLoss(cfg.FOCAL_GAMMA, alpha)
    if task == 'DIA':
        return AsymmetricLoss(cfg.GAMMA_POS, cfg.GAMMA_NEG)
    w = None                                   # LOS: class-balanced weights (effective number of samples)
    if y is not None:
        c = torch.bincount(torch.from_numpy(y).long(), minlength=NUM_LOS).float()
        if (c == 0).any():
            log(f'  WARNING: empty LOS classes {(c == 0).nonzero().view(-1).tolist()}')
        c = c.clamp_min(1.0)
        w = (1 - 0.9999) / (1 - torch.pow(0.9999, c))
        w = w / w.sum() * NUM_LOS
        log(f'  LOS counts {c.numpy().astype(int).tolist()}')
    return SoftOrdinalCE(NUM_LOS, cfg.LOS_SIGMA, w)


# --------------------------------------------------------------------------- #
# Continual learning (EWC) and weight averaging (EMA)
# --------------------------------------------------------------------------- #
def log_lik(task, logits, y):
    kind, logits = TASK_KIND[task], logits.float()
    if kind == 'binary':
        return -F.binary_cross_entropy_with_logits(logits.view(-1), y.view(-1).float(), reduction='sum')
    if kind == 'multilabel':
        return -F.binary_cross_entropy_with_logits(logits, y.float(), reduction='sum')
    return F.log_softmax(logits, -1)[torch.arange(logits.size(0)), y.long()].sum()


class EWC:
    """Elastic Weight Consolidation: diagonal empirical Fisher + anchor parameters of a finished task."""

    def __init__(self, model, loader, task, cfg, n_samples=256):
        """`loader` must have batch_size=1 so each gradient is a per-sample gradient."""
        self.fisher = {n: torch.zeros_like(p) for n, p in model.named_parameters() if p.requires_grad}
        self.params = {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad}
        model.eval()
        used = 0
        for b, y, _ in loader:
            if used >= n_samples:
                break
            b, y = to_dev(b, cfg.DEVICE), y.to(cfg.DEVICE)
            model.zero_grad(set_to_none=True)
            log_lik(task, model(b)[task], y).backward()
            for n, p in model.named_parameters():
                if p.requires_grad and p.grad is not None:
                    self.fisher[n] += p.grad.detach() ** 2
            used += 1
        for n in self.fisher:
            self.fisher[n] /= max(used, 1)
        model.zero_grad(set_to_none=True)
        log(f'  Fisher estimated from {used} samples')

    def merge(self, other):
        """Online EWC: accumulate Fisher matrices, anchor at the newest parameters."""
        for n in self.fisher:
            self.fisher[n] += other.fisher[n]
        self.params = other.params
        return self

    @torch.no_grad()
    def add_grad(self, model, lam):
        """Add the penalty gradient lam * F * (theta - theta*) to the existing gradients."""
        for n, p in model.named_parameters():
            if p.requires_grad and p.grad is not None and n in self.fisher:
                p.grad.add_(self.fisher[n] * (p.detach() - self.params[n]), alpha=lam)

    @torch.no_grad()
    def value(self, model, lam):
        s = torch.zeros((), device=next(model.parameters()).device)
        for n, p in model.named_parameters():
            if p.requires_grad and n in self.fisher:
                s += (self.fisher[n] * (p.detach() - self.params[n]) ** 2).sum()
        return (lam / 2.0) * s


class EMA:
    def __init__(self, model, decay=0.999):
        self.model, self.decay = model, decay
        self.shadow = {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad}
        self.backup = {}
        self.names = list(self.shadow)

    @torch.no_grad()
    def update(self):
        params = dict(self.model.named_parameters())
        p = [params[n].detach() for n in self.names]
        s = [self.shadow[n] for n in self.names]
        torch._foreach_mul_(s, self.decay)
        torch._foreach_add_(s, p, alpha=1 - self.decay)

    @torch.no_grad()
    def apply_shadow(self):
        self.backup = {}
        for n, p in self.model.named_parameters():
            if p.requires_grad:
                self.backup[n] = p.detach().clone()
                p.data.copy_(self.shadow[n])

    @torch.no_grad()
    def restore(self):
        for n, p in self.model.named_parameters():
            if p.requires_grad and n in self.backup:
                p.data.copy_(self.backup[n])
        self.backup = {}
