"""Utilities for PRISM²: constants, label parsing, EHR caching, graph batching,
metrics, logging and configuration."""
from __future__ import annotations

import ast
import contextlib
import csv
import json
import os
import random
import time

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (average_precision_score, cohen_kappa_score, f1_score,
                             mean_absolute_error, precision_recall_curve, roc_auc_score)
from torch.utils.data import DataLoader, Dataset

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #
SEED = 42
EWC_LAMBDA = 10.0

TASK_ORDER = ['IHM', 'LOS', 'PHE', 'REA', 'DEC', 'DIA']
TASK_DIR = {'IHM': 'in_hospital_mortality', 'LOS': 'length_of_stay', 'PHE': 'phenotyping',
            'REA': 'readmission', 'DEC': 'decompensation', 'DIA': 'diagnosis'}
TASK_WINDOW = {'IHM': 48, 'LOS': 200, 'PHE': 200, 'REA': 200, 'DEC': 200, 'DIA': 200}
TASK_KIND = {'IHM': 'binary', 'REA': 'binary', 'DEC': 'binary', 'LOS': 'ordinal',
             'PHE': 'multilabel', 'DIA': 'multilabel'}
PRIMARY = {'binary': 'auroc', 'ordinal': 'macro_f1', 'multilabel': 'macro_auroc'}

MODALITIES = ['ehr', 'cxr', 'notes', 'concepts', 'demo']

# node types
NT_PATIENT, NT_TIME, NT_IMAGE, NT_NOTE, NT_CONCEPT, NT_DEMO = range(6)

# edge (relation) types
REL = {'forward_temporal': 0, 'backward_temporal': 1, 'ipsilateral': 2, 'bilateral': 3,
       'symptom_indicates_disease': 4, 'disease_manifests_symptom': 5,
       'disease_treated_by_treatment': 6, 'treatment_treats_disease': 7,
       'ehr_to_patient': 8, 'cxr_to_patient': 9, 'note_to_patient': 10,
       'concept_to_patient': 11, 'demo_to_patient': 12,
       'patient_to_ehr': 13, 'patient_to_cxr': 14, 'patient_to_note': 15,
       'patient_to_concept': 16, 'patient_to_demo': 17}
NUM_REL = len(REL)
CONCEPT_REL_BASE = 4          # rel_adj value v (1..4) -> relation id v + CONCEPT_REL_BASE - 1

NUM_LOS = 10                  # LOS classes: 0-7 one per day, 8 = 8-14 d, 9 = 14+ d
N_CXR_REGIONS = 6


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #
def log(m=''):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def seed_all(s=SEED):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


def read_csv_any(path, **kw):
    return pd.read_csv(path, compression='gzip' if str(path).endswith('.gz') else None, **kw)


def to_dev(b, dev):
    """Move all tensors of a batch dict to `dev`."""
    return {k: (v.to(dev, non_blocking=True) if torch.is_tensor(v) else v) for k, v in b.items()}


def pick_amp(dev):
    """bf16 on Ampere+, fp16 on older GPUs, None (no autocast) on CPU."""
    if dev.type != 'cuda':
        return None
    cap = torch.cuda.get_device_capability(dev)
    return torch.bfloat16 if cap[0] >= 8 else torch.float16


def autocast_ctx(dev):
    amp = pick_amp(dev)
    if amp is None:
        return contextlib.nullcontext()
    return torch.autocast(device_type='cuda', dtype=amp)


def make_scaler(dev):
    enabled = pick_amp(dev) == torch.float16
    if hasattr(torch, 'amp') and hasattr(torch.amp, 'GradScaler'):
        return torch.amp.GradScaler('cuda', enabled=enabled)
    return torch.cuda.amp.GradScaler(enabled=enabled)


# --------------------------------------------------------------------------- #
# Labels
# --------------------------------------------------------------------------- #
def los_bucket(hours):
    d = hours / 24.0
    if d < 8:
        return int(max(d, 0))      # classes 0-7: one per day
    return 8 if d < 14 else 9      # 8-14 days, 14+ days


def parse_y(v):
    """One y_true cell -> float (binary / class id) or list of floats (multilabel)."""
    if isinstance(v, (int, float, np.integer, np.floating)):
        return float(v)
    s = str(v).strip()
    if s.startswith(('[', '(')):
        try:
            return [float(x) for x in ast.literal_eval(s)]
        except Exception:
            s = s.strip('[]()')
    parts = [p for p in s.replace(';', ' ').replace(',', ' ').split() if p]
    if len(parts) > 1:
        return [float(p) for p in parts]
    try:
        return float(parts[0]) if parts else 0.0
    except Exception:
        return 0.0


def labels_from_listfile(path, task, los_is_class=False):
    """Read the `y_true` column of a listfile. Returns (labels, label_dim)."""
    df = read_csv_any(path)
    if 'y_true' not in df.columns:
        raise SystemExit(f'{path}: no y_true column; columns {list(df.columns)}')
    vals = [parse_y(v) for v in df['y_true'].tolist()]
    kind = TASK_KIND[task]

    if kind == 'multilabel':                          # PHE, DIA: one list per row
        dims = {len(v) if isinstance(v, list) else 1 for v in vals}
        if len(dims) != 1:
            raise SystemExit(f'{path}: rows have different label counts {sorted(dims)}')
        dim = dims.pop()
        Y = np.array([v if isinstance(v, list) else [v] for v in vals], 'float32')
        if not np.isin(Y, (0.0, 1.0)).all():
            raise SystemExit(f'{path}: multilabel y_true must be 0/1')
        return Y, dim

    flat = np.array([v[0] if isinstance(v, list) else v for v in vals], 'float64')

    if kind == 'ordinal':                             # LOS: 10-class
        already_class = los_is_class or (
            np.all(flat == np.round(flat)) and flat.min() >= 0 and flat.max() <= NUM_LOS - 1)
        if already_class:
            Y = np.clip(flat.astype('int64'), 0, NUM_LOS - 1)
        else:                                         # raw hours -> bins
            Y = np.array([los_bucket(h) for h in flat], 'int64')
        log(f'  LOS labels from {os.path.basename(path)}: '
            f'{"class ids" if already_class else "hours -> bins"}, '
            f'classes present {sorted(set(Y.tolist()))}')
        return Y, NUM_LOS

    if not np.isin(flat, (0.0, 1.0)).all():           # IHM, REA, DEC
        raise SystemExit(f'{path}: binary y_true must be 0/1')
    return flat.astype('float32'), 1


# --------------------------------------------------------------------------- #
# EHR time-series cache
# --------------------------------------------------------------------------- #
def index_files(root, exts=('.csv',)):
    idx = {}
    for d, _s, fs in os.walk(root):
        for f in fs:
            if f.lower().endswith(exts):
                idx.setdefault(f, os.path.join(d, f))
    return idx


def build_ehr_cache(listfile, ehr_root, out_npz, T, resample='uniform', time_col='auto', nfeat=None):
    """Read every stay's time-series CSV once and cache it as a fixed-size (N, T, F) float16 array
    plus a (N, T) validity mask."""
    n_rows = len(read_csv_any(listfile))
    if os.path.exists(out_npz):
        z = np.load(out_npz)
        if z['x'].shape[0] == n_rows and z['x'].shape[1] == T:
            return out_npz
    df = read_csv_any(listfile)
    low = {c.lower(): c for c in df.columns}
    fcol = low.get('time_series') or low.get('filename') or low.get('ts')
    if fcol is None:
        raise SystemExit(f'{listfile}: no time_series column; columns {list(df.columns)}')
    names = df[fcol].astype(str).tolist()
    idx = index_files(ehr_root)
    probe = None
    for n in names:
        p = idx.get(os.path.basename(n)) or os.path.join(ehr_root, n)
        if os.path.exists(p):
            probe = p
            break
    if probe is None:
        raise SystemExit(f'none of the {len(names)} time_series files found under {ehr_root}')
    head = pd.read_csv(probe, nrows=5)
    tcol = None
    if time_col != 'none':
        for c in head.columns:
            if str(c).strip().lower() in ('hours', 'hour', 'time', 'charttime', 'offset'):
                tcol = c
                break
    feats = [c for c in head.columns if c != tcol]
    F_ = nfeat or len(feats)
    N = len(names)
    X = np.zeros((N, T, F_), 'float16')
    M = np.zeros((N, T), 'float32')
    miss, t0 = 0, time.time()
    for i, n in enumerate(names):
        if i and i % 5000 == 0:
            r = i / max(time.time() - t0, 1e-9)
            log(f'    ehr {i:,}/{N:,}  {r:.0f}/s  eta {(N - i) / r / 60:.1f} min')
        p = idx.get(os.path.basename(n)) or os.path.join(ehr_root, n)
        try:
            d = pd.read_csv(p)
        except Exception:
            miss += 1
            continue
        if tcol is not None and tcol in d.columns:
            h = pd.to_numeric(d[tcol], errors='coerce').fillna(0.0).to_numpy()
            d = d.loc[h <= float(T), [c for c in d.columns if c != tcol]]
        else:
            d = d.iloc[:T]
        A = d.apply(pd.to_numeric, errors='coerce').fillna(0.0).to_numpy('float32')
        if A.shape[1] != F_:
            A = A[:, :F_] if A.shape[1] > F_ else np.pad(A, ((0, 0), (0, F_ - A.shape[1])))
        L = A.shape[0]
        if L == 0:
            miss += 1
            continue
        if L > T:
            if resample == 'first':
                sel = np.arange(T)
            elif resample == 'last':
                sel = np.arange(L - T, L)
            else:
                sel = np.linspace(0, L - 1, T).round().astype(int)
            A = A[sel]
            L = T
        X[i, :L] = A.astype('float16')
        M[i, :L] = 1.0
    log(f'  ehr cache {X.shape} | {miss:,} stays unreadable/empty | {out_npz}')
    np.savez(out_npz, x=X, mask=M)
    return out_npz


# --------------------------------------------------------------------------- #
# Multimodal data container
# --------------------------------------------------------------------------- #
class Modal:
    """All modalities of one (task, split). See README for the expected file layout."""

    def __init__(self, mm_dir, listfile, ehr_root, T, resample, time_col, nfeat=None):
        P = lambda f: os.path.join(mm_dir, f)
        self.T = T
        z = np.load(build_ehr_cache(listfile, ehr_root, P(f'ehr_{T}.npz'), T, resample, time_col, nfeat))
        self.ehr, self.ehr_mask = z['x'], z['mask']
        self.N, self.F = self.ehr.shape[0], self.ehr.shape[2]
        self.cxr = np.load(P('cxr.npy'), mmap_mode='r')                 # (N, 6, D_cxr)
        self.cxr_mask = np.load(P('cxr_mask.npy'))                      # (N,)
        self.notes = np.load(P('notes.npy'), mmap_mode='r')             # (N, D_text)
        self.notes_mask = np.load(P('notes_mask.npy'))                  # (N,)
        self.cid = np.load(P('concept_ids.npy'))                        # (N, K)
        self.cmask = np.load(P('concept_mask.npy'))                     # (N, K)
        self.cemb = np.load(P('concept_emb.npy'))                       # (V, D_ent)
        self.ckind = np.load(P('concept_kind.npy'))                     # (V,) in {0,1,2}
        self.rel = np.load(P('rel_adj.npy'), mmap_mode='r')             # (N, K, K) in {0..4}
        self.demo = np.load(P('demo.npy'))                              # (N, D_demo)
        self.demo_mask = np.load(P('demo_mask.npy'))                    # (N,)
        g = np.load(P('cxr_graph.npz'))
        self.cxr_ei, self.cxr_et, self.cxr_ea = g['edge_index'], g['edge_type'], g['edge_attr']
        self.K = int(self.cid.shape[1])
        self.text_dim = int(self.notes.shape[1])
        self.cxr_dim = int(self.cxr.shape[2])
        self.ent_dim = int(self.cemb.shape[1])
        self.demo_dim = int(self.demo.shape[1])
        for nm, a in (('cxr', self.cxr), ('notes', self.notes), ('concept_ids', self.cid),
                      ('rel_adj', self.rel), ('demo', self.demo)):
            if a.shape[0] != self.N:
                raise SystemExit(f'{mm_dir}: {nm} has {a.shape[0]} rows but ehr has {self.N}')


# --------------------------------------------------------------------------- #
# Patient graph construction and batching
# --------------------------------------------------------------------------- #
def static_edges(T, K):
    """Edges that are identical for every patient (temporal chain + modality<->patient links).
    Node layout per patient: [patient | T time steps | 6 CXR regions | 1 note | K concepts | 1 demo]."""
    oP, oT, oI = 0, 1, 1 + T
    oN, oC, oD = 1 + T + N_CXR_REGIONS, 1 + T + N_CXR_REGIONS + 1, 1 + T + N_CXR_REGIONS + 1 + K
    src, dst, typ, att = [], [], [], []

    def add(s, d, t, a=(0., 0.)):
        src.append(s)
        dst.append(d)
        typ.append(t)
        att.append(a)

    for i in range(T - 1):
        add(oT + i, oT + i + 1, REL['forward_temporal'])
        add(oT + i + 1, oT + i, REL['backward_temporal'])
    for i in range(T):
        add(oT + i, oP, REL['ehr_to_patient'])
        add(oP, oT + i, REL['patient_to_ehr'])
    for i in range(N_CXR_REGIONS):
        add(oI + i, oP, REL['cxr_to_patient'])
        add(oP, oI + i, REL['patient_to_cxr'])
    add(oN, oP, REL['note_to_patient'])
    add(oP, oN, REL['patient_to_note'])
    for i in range(K):
        add(oC + i, oP, REL['concept_to_patient'])
        add(oP, oC + i, REL['patient_to_concept'])
    add(oD, oP, REL['demo_to_patient'])
    add(oP, oD, REL['patient_to_demo'])
    n = 1 + T + N_CXR_REGIONS + 1 + K + 1
    return (np.array([src, dst], 'int64'), np.array(typ, 'int64'),
            np.array(att, 'float32'), dict(P=oP, T=oT, I=oI, N=oN, C=oC, D=oD, n=n))


class StayDataset(Dataset):
    """Index-only dataset; all tensors are assembled in `Collater`."""

    def __init__(self, mod: Modal, y, kind, dim=0):
        self.mod, self.y, self.kind, self.dim = mod, y, kind, dim

    def __len__(self):
        return self.mod.N

    def __getitem__(self, i):
        return i


class Collater:
    """Builds one big disjoint-union graph for a batch of stays."""

    def __init__(self, mod: Modal, y, kind, drop=()):
        self.m, self.y, self.kind, self.drop = mod, y, kind, set(drop)
        T, K = mod.T, mod.K
        ei, et, ea, off = static_edges(T, K)
        self.ei, self.et, self.ea = torch.from_numpy(ei), torch.from_numpy(et), torch.from_numpy(ea)
        self.off, self.n = off, off['n']
        self.cxr_ei = torch.from_numpy(mod.cxr_ei.astype('int64'))
        self.cxr_et = torch.from_numpy(mod.cxr_et.astype('int64'))
        self.cxr_ea = torch.from_numpy(mod.cxr_ea.astype('float32'))
        nt = torch.zeros(self.n, dtype=torch.long)          # patient node = NT_PATIENT = 0
        nt[off['T']:off['T'] + T] = NT_TIME
        nt[off['I']:off['I'] + N_CXR_REGIONS] = NT_IMAGE
        nt[off['N']] = NT_NOTE
        nt[off['C']:off['C'] + K] = NT_CONCEPT
        nt[off['D']] = NT_DEMO
        self.node_type = nt

    def __call__(self, idx):
        m, off = self.m, self.off
        idx = np.asarray(sorted(idx), dtype=np.int64)
        B, T, K, n = len(idx), m.T, m.K, self.n
        ehr = torch.from_numpy(np.asarray(m.ehr[idx], 'float32'))
        ehr_m = torch.from_numpy(np.asarray(m.ehr_mask[idx], 'float32'))
        cxr = torch.from_numpy(np.asarray(m.cxr[idx], 'float32'))
        cxr_m = torch.from_numpy(np.asarray(m.cxr_mask[idx], 'float32'))
        txt = torch.from_numpy(np.asarray(m.notes[idx], 'float32'))
        txt_m = torch.from_numpy(np.asarray(m.notes_mask[idx], 'float32'))
        cid = m.cid[idx]
        cm = torch.from_numpy(np.asarray(m.cmask[idx], 'float32'))
        cfe = torch.from_numpy(m.cemb[cid.reshape(-1)].reshape(B, K, -1).astype('float32'))
        ckind = torch.from_numpy(m.ckind[cid].astype('int64'))
        dem = torch.from_numpy(np.asarray(m.demo[idx], 'float32'))
        dem_m = torch.from_numpy(np.asarray(m.demo_mask[idx], 'float32'))
        rel = torch.from_numpy(np.asarray(m.rel[idx]).astype('int64'))

        # modality ablation: zero features and mark every node of that modality as missing
        if 'ehr' in self.drop:
            ehr = torch.zeros_like(ehr)
            ehr_m = torch.zeros_like(ehr_m)
        if 'cxr' in self.drop:
            cxr = torch.zeros_like(cxr)
            cxr_m = torch.zeros_like(cxr_m)
        if 'notes' in self.drop:
            txt = torch.zeros_like(txt)
            txt_m = torch.zeros_like(txt_m)
        if 'concepts' in self.drop:
            cfe = torch.zeros_like(cfe)
            cm = torch.zeros_like(cm)
            rel = torch.zeros_like(rel)
        if 'demo' in self.drop:
            dem = torch.zeros_like(dem)
            dem_m = torch.zeros_like(dem_m)

        base = torch.arange(B, dtype=torch.long) * n
        ei = (self.ei.unsqueeze(0) + base.view(B, 1, 1)).permute(1, 0, 2).reshape(2, -1)
        et = self.et.repeat(B)
        ea = self.ea.repeat(B, 1)

        # CXR region graph (local ids 0..5 -> global image-node ids)
        cb = base + off['I']
        cei = (self.cxr_ei.unsqueeze(0) + cb.view(B, 1, 1)).permute(1, 0, 2).reshape(2, -1)
        cet = (self.cxr_et + REL['ipsilateral']).repeat(B)
        cea = self.cxr_ea.repeat(B, 1)

        # concept relation graph from rel_adj (value 0 = no edge)
        nz = (rel > 0).nonzero(as_tuple=False)
        if nz.numel():
            bi, xi, yi = nz[:, 0], nz[:, 1], nz[:, 2]
            rb = base[bi] + off['C']
            kei = torch.stack([rb + xi, rb + yi])
            ket = rel[bi, xi, yi] + (CONCEPT_REL_BASE - 1)
            kea = torch.zeros(kei.shape[1], 2)
        else:
            kei = torch.zeros(2, 0, dtype=torch.long)
            ket = torch.zeros(0, dtype=torch.long)
            kea = torch.zeros(0, 2)

        edge_index = torch.cat([ei, cei, kei], 1)
        edge_type = torch.cat([et, cet, ket])
        edge_attr = torch.cat([ea, cea, kea], 0)

        avail = torch.ones(B, n)                                   # 1 = node carries real data
        avail[:, off['T']:off['T'] + T] = ehr_m
        avail[:, off['I']:off['I'] + N_CXR_REGIONS] = cxr_m.view(B, 1)
        avail[:, off['N']] = txt_m
        avail[:, off['C']:off['C'] + K] = cm
        avail[:, off['D']] = dem_m

        batch = torch.arange(B, dtype=torch.long).repeat_interleave(n)
        nt = self.node_type.repeat(B)

        yb = self.y[idx]
        if self.kind == 'ordinal':
            y = torch.from_numpy(yb.astype('int64'))
        elif self.kind == 'multilabel':
            y = torch.from_numpy(yb.astype('float32'))
        else:
            y = torch.from_numpy(yb.astype('float32')).view(-1)

        return dict(ehr=ehr, cxr=cxr, txt=txt, cfe=cfe, ckind=ckind, demo=dem,
                    edge_index=edge_index, edge_type=edge_type, edge_attr=edge_attr,
                    avail=avail.reshape(-1), batch=batch, node_type=nt, B=B,
                    off=off, T=T, K=K, n=n), y, torch.from_numpy(idx)


def make_loader(mod, y, task, shuffle, cfg, drop=(), bs=None, persistent=None):
    ds = StayDataset(mod, y, TASK_KIND[task])
    col = Collater(mod, y, TASK_KIND[task], drop)
    if persistent is None:
        persistent = shuffle                       # keep workers alive only for training loaders
    return DataLoader(ds, batch_size=bs or cfg.BATCH_SIZE, shuffle=shuffle,
                      num_workers=cfg.NUM_WORKERS, collate_fn=col, drop_last=False,
                      pin_memory=cfg.DEVICE.type == 'cuda',
                      persistent_workers=bool(persistent and cfg.NUM_WORKERS > 0))


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def best_threshold(y, p):
    pr, rc, th = precision_recall_curve(y, p)
    f1 = 2 * pr * rc / (pr + rc + 1e-8)
    i = int(np.argmax(f1))
    return (float(th[i]) if i < len(th) else 0.5), float(f1[i])


@torch.no_grad()
def evaluate(model, loader, task, cfg, thr=None):
    """Binary: AUROC / AUPRC / F1 (threshold picked on validation, reused on test).
    Ordinal (LOS): macro/micro F1, accuracy, MAE, off-by-one, linear kappa.
    Multilabel: macro/micro AUROC and AUPRC, per-label AUROC."""
    model.eval()
    kind = TASK_KIND[task]
    P, Y = [], []
    for b, y, _ in loader:
        b = to_dev(b, cfg.DEVICE)
        with autocast_ctx(cfg.DEVICE):
            o = model(b)[task]
        if kind == 'ordinal':
            P.append(o.float().argmax(-1).cpu().numpy())
        else:
            P.append(torch.sigmoid(o.float()).cpu().numpy())
        Y.append(y.numpy())
    if not P:
        return {}
    m = {}
    if kind == 'binary':
        p, y = np.concatenate(P), np.concatenate(Y)
        if len(np.unique(y)) < 2:
            return {'auroc': float('nan'), 'auprc': float('nan'), 'f1': 0.0}
        m['auroc'] = float(roc_auc_score(y, p))
        m['auprc'] = float(average_precision_score(y, p))
        if thr is None:                                   # validation: pick threshold
            m['threshold'], m['f1'] = best_threshold(y, p)
        else:                                             # test: apply the validation threshold
            m['threshold'] = float(thr)
            m['f1'] = float(f1_score(y, (p >= thr).astype(int), zero_division=0))
        return m
    if kind == 'ordinal':
        p, y = np.concatenate(P), np.concatenate(Y)
        m['macro_f1'] = float(f1_score(y, p, average='macro', zero_division=0))
        m['micro_f1'] = float(f1_score(y, p, average='micro', zero_division=0))
        m['accuracy'] = float((y == p).mean())
        m['mae'] = float(mean_absolute_error(y, p))
        m['off_by_one'] = float((np.abs(y - p) <= 1).mean())
        try:
            m['kappa'] = float(cohen_kappa_score(y, p, weights='linear'))
        except Exception:
            m['kappa'] = float('nan')
        return m
    p = np.concatenate(P, 0)
    y = np.concatenate(Y, 0).astype(float)
    au, ap = [], []
    for k in range(y.shape[1]):
        if len(np.unique(y[:, k])) < 2:
            au.append(np.nan)
            ap.append(np.nan)
        else:
            au.append(roc_auc_score(y[:, k], p[:, k]))
            ap.append(average_precision_score(y[:, k], p[:, k]))
    m['macro_auroc'] = float(np.nanmean(au)) if not np.all(np.isnan(au)) else float('nan')
    m['macro_auprc'] = float(np.nanmean(ap)) if not np.all(np.isnan(ap)) else float('nan')
    yf, pf = y.ravel(), p.ravel()
    if len(np.unique(yf)) >= 2:
        m['micro_auroc'] = float(roc_auc_score(yf, pf))
        m['micro_auprc'] = float(average_precision_score(yf, pf))
    m['per_label_auroc'] = [None if np.isnan(v) else float(v) for v in au]
    return m


def primary(task, m):
    """Primary model-selection metric of a task (NaN if unavailable)."""
    v = m.get(PRIMARY[TASK_KIND[task]], float('nan'))
    try:
        return float(v)
    except Exception:
        return float('nan')


def cl_metrics(R, tasks, done=None):
    """Continual-learning summary from the result matrix R[k, j] = metric on task j after stage k.
    AP = mean final performance, BWT = backward transfer, forgetting = R[j,j] - R[K-1,j]."""
    K = done if done else len(tasks)
    tasks = tasks[:K]
    R = R[:K, :K]
    out = {}
    fin = R[K - 1]
    out['AP'] = float(np.nanmean(fin))
    out['per_task_final'] = {tasks[j]: (None if np.isnan(fin[j]) else float(fin[j])) for j in range(K)}
    bwt = [R[K - 1, j] - R[j, j] for j in range(K - 1)]
    out['BWT'] = float(np.nanmean(bwt)) if bwt else 0.0
    out['forgetting'] = {tasks[j]: float(R[j, j] - R[K - 1, j]) for j in range(K - 1)}
    out['mean_forgetting'] = (float(np.nanmean([R[j, j] - R[K - 1, j] for j in range(K - 1)]))
                              if K > 1 else 0.0)
    out['R'] = [[None if np.isnan(v) else float(v) for v in row] for row in R]
    out['tasks'] = tasks
    return out


# --------------------------------------------------------------------------- #
# Logging and config
# --------------------------------------------------------------------------- #
class Logger:
    def __init__(self, save_dir, cfg_snapshot):
        self.d = save_dir
        os.makedirs(self.d, exist_ok=True)
        self.hist = {}
        self.dump('run_config.json', cfg_snapshot)

    def epoch(self, task, ep, loss, m, lr):
        self.hist.setdefault(task, []).append(
            {'epoch': ep, 'loss': loss, 'lr': lr,
             **{k: v for k, v in m.items() if isinstance(v, (int, float))}})
        td = os.path.join(self.d, task)
        os.makedirs(td, exist_ok=True)
        with open(os.path.join(td, 'metrics.json'), 'w') as f:
            json.dump(self.hist[task], f, indent=1)
        with open(os.path.join(td, 'training_curve.csv'), 'w', newline='') as f:
            w = csv.writer(f)
            keys = sorted({k for r in self.hist[task] for k in r})
            w.writerow(keys)
            for r in self.hist[task]:
                w.writerow([r.get(k, '') for k in keys])

    def dump(self, name, obj):
        with open(os.path.join(self.d, name), 'w') as f:
            json.dump(obj, f, indent=1)


class Cfg:
    pass


def build_cfg(a):
    c = Cfg()
    c.DEVICE = torch.device(a.device if (torch.cuda.is_available() or a.device == 'cpu') else 'cpu')
    c.HIDDEN_DIM, c.NUM_HEADS, c.NUM_LAYERS, c.DROPOUT = a.hidden, a.heads, a.layers, a.dropout
    c.BASE_LR, c.WEIGHT_DECAY, c.WARMUP_STEPS, c.LR_MIN = a.lr, a.weight_decay, a.warmup, 1e-6
    c.LR = {'REA': a.lr * 1.5, 'PHE': a.lr * 1.5}
    c.BATCH_SIZE, c.NUM_WORKERS = a.batch_size, a.workers
    c.EARLY_STOP_PATIENCE = a.patience
    c.GRAPH_DROPOUT = a.graph_dropout
    c.EMA_DECAY = a.ema
    c.USE_EWC = not a.no_ewc
    c.EWC_LAMBDA = a.lam
    c.EWC_SAMPLES = a.fisher_samples
    c.MORT_FOCAL_WEIGHT, c.MORT_DICE_WEIGHT = 0.7, 0.3     # IHM: 0.7 * focal + 0.3 * dice
    c.GAMMA_POS, c.GAMMA_NEG = 1.0, 3.0                    # asymmetric loss (REA, DIA)
    c.FOCAL_GAMMA = a.focal_gamma
    c.LOS_SIGMA = a.los_sigma
    c.TASK_W = {t: 1.0 for t in TASK_ORDER}
    for kv in filter(None, a.task_w.split(',')):
        k_, v_ = kv.split('=')
        c.TASK_W[k_.strip().upper()] = float(v_)
    c.SAVE_DIR = os.path.abspath(os.path.expanduser(a.save_dir))
    os.makedirs(c.SAVE_DIR, exist_ok=True)
    return c


def parse_windows(s):
    """'IHM=48,LOS=200' -> dict of per-task EHR window lengths (defaults from TASK_WINDOW)."""
    w = dict(TASK_WINDOW)
    for kv in (s or '').split(','):
        if kv.strip():
            k, v = kv.split('=')
            w[k.strip().upper()] = int(v)
    return w
