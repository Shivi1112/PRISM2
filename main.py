"""PRISM²: task-incremental training / evaluation on MIMIC-IV with EWC.

Example:
    python main.py --data_dir ./data/mimic4 --mm_dir ./data/prism2_mm --save_dir ./runs/prism2
"""
from __future__ import annotations

import argparse
import copy
import math
import os

import numpy as np
import torch

from model import EMA, EWC, PatientGraphNet, make_criterion
from utils import (EWC_LAMBDA, MODALITIES, PRIMARY, SEED, TASK_DIR, TASK_KIND, TASK_ORDER, Logger,
                   Modal, autocast_ctx, build_cfg, cl_metrics, evaluate, labels_from_listfile, log,
                   make_loader, make_scaler, parse_windows, primary, seed_all, to_dev)


def train_task(cfg, model, task, tr, va, epochs, ema, ewc, logger):
    """Train on one task (warmup + cosine LR, EMA weights for validation, early stopping)."""
    dev = cfg.DEVICE
    model.to(dev)
    crit = make_criterion(task, cfg, tr.dataset.y)
    lr = cfg.LR.get(task, cfg.BASE_LR)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=cfg.WEIGHT_DECAY)
    scaler = make_scaler(dev)
    total = epochs * max(1, len(tr))
    step = 0
    best, best_ep, bad = -1e9, -1, 0
    avg, cur = 0.0, lr
    for ep in range(1, epochs + 1):
        model.train()
        run, nb = 0.0, 0
        for b, y, _ in tr:
            step += 1
            if step < cfg.WARMUP_STEPS:
                cur = lr * step / cfg.WARMUP_STEPS
            else:
                pr = min((step - cfg.WARMUP_STEPS) / max(1, total - cfg.WARMUP_STEPS), 1.0)
                cur = cfg.LR_MIN + (lr - cfg.LR_MIN) * 0.5 * (1 + math.cos(math.pi * pr))
            for g in opt.param_groups:
                g['lr'] = cur
            b = to_dev(b, dev)
            y = y.to(dev, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with autocast_ctx(dev):
                loss = cfg.TASK_W.get(task, 1.0) * crit(model(b)[task], y)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            if ewc is not None and cfg.USE_EWC:
                ewc.add_grad(model, cfg.EWC_LAMBDA)       # EWC penalty gradient on unscaled grads
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            if ema is not None:
                ema.update()
            run += float(loss.detach())
            nb += 1
            avg = run / max(1, nb)
        if ema is not None:
            ema.apply_shadow()
        m = evaluate(model, va, task, cfg)
        p = primary(task, m)
        p = p if not math.isnan(p) else -1e9
        log(f'[{task}] ep {ep}/{epochs} loss {avg:.4f} lr {cur:.2e} val '
            + ' '.join(f'{k}={v:.4f}' for k, v in m.items() if isinstance(v, float)))
        logger.epoch(task, ep, avg, m, cur)
        if p > best:
            best, best_ep, bad = p, ep, 0
            torch.save({'model': model.state_dict(), 'epoch': ep, 'metric': best, 'val': m},
                       os.path.join(cfg.SAVE_DIR, f'best_{task}.pth'))   # EMA weights
        else:
            bad += 1
        if ema is not None:
            ema.restore()
        if bad >= cfg.EARLY_STOP_PATIENCE:
            log(f'[{task}] early stop at {ep}, best ep {best_ep} ({best:.4f})')
            break
    log(f'[{task}] best {PRIMARY[TASK_KIND[task]]}={best:.4f} @ ep {best_ep}')
    return model


def parse_args():
    ap = argparse.ArgumentParser(description='PRISM²: patient-centric graph, task-incremental training')
    ap.add_argument('--data_dir', default='./data/mimic4',
                    help='folder with <task_dir>/{train,val,test}_listfile.csv')
    ap.add_argument('--mm_dir', default='./data/prism2_mm',
                    help='folder with pre-extracted features: <TASK>/<split>/*.npy')
    ap.add_argument('--ehr_dir', default=None, help='root of the EHR time-series CSVs (default: data_dir)')
    ap.add_argument('--save_dir', default='./prism2_out')
    ap.add_argument('--tasks', default=','.join(TASK_ORDER), help='task order of the incremental stream')
    ap.add_argument('--windows', default=None, help='EHR steps per task, e.g. "IHM=48,LOS=200"')
    ap.add_argument('--epochs', default='IHM=25,LOS=40,PHE=30,REA=15,DEC=15,DIA=15')
    ap.add_argument('--drop', default='', help=f'modality ablation, comma list from {MODALITIES}')
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--hidden', type=int, default=384)
    ap.add_argument('--heads', type=int, default=4)
    ap.add_argument('--layers', type=int, default=2)
    ap.add_argument('--dropout', type=float, default=0.2)
    ap.add_argument('--graph_dropout', type=float, default=0.1)
    ap.add_argument('--lr', type=float, default=5e-5)
    ap.add_argument('--weight_decay', type=float, default=1e-5)
    ap.add_argument('--warmup', type=int, default=800)
    ap.add_argument('--batch_size', type=int, default=128)
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--patience', type=int, default=7)
    ap.add_argument('--ema', type=float, default=0.999)
    ap.add_argument('--lam', type=float, default=EWC_LAMBDA, help='EWC strength')
    ap.add_argument('--no_ewc', action='store_true', help='fine-tune without EWC')
    ap.add_argument('--ewc_mode', default='online', choices=['online', 'per_task'])
    ap.add_argument('--resample', default='uniform', choices=['uniform', 'first', 'last'])
    ap.add_argument('--time_col', default='auto', choices=['auto', 'none'])
    ap.add_argument('--seed', type=int, default=SEED)
    ap.add_argument('--los_metric', default='macro_f1', choices=['macro_f1', 'kappa', 'accuracy'])
    ap.add_argument('--tag', default='')
    ap.add_argument('--fisher_samples', type=int, default=512)
    ap.add_argument('--focal_gamma', type=float, default=2.0)
    ap.add_argument('--los_sigma', type=float, default=1.0)
    ap.add_argument('--task_w', default='', help='loss weights, e.g. "IHM=1.0,LOS=0.5"')
    ap.add_argument('--los_is_class', action='store_true',
                    help='LOS listfile already holds class ids 0-9 (instead of hours)')
    return ap.parse_args()


def main():
    a = parse_args()
    seed_all(a.seed)
    cfg = build_cfg(a)
    data_dir = os.path.abspath(os.path.expanduser(a.data_dir))
    mm_root = os.path.abspath(os.path.expanduser(a.mm_dir))
    ehr_root = os.path.abspath(os.path.expanduser(a.ehr_dir)) if a.ehr_dir else data_dir
    tasks = [t.strip().upper() for t in a.tasks.split(',') if t.strip()]
    for t in tasks:
        if t not in TASK_ORDER:
            raise SystemExit(f'--tasks {t}: choose from {TASK_ORDER}')
    windows = parse_windows(a.windows)
    epochs = {k.split('=')[0].strip().upper(): int(k.split('=')[1])
              for k in a.epochs.split(',') if '=' in k}
    drop = [d.strip().lower() for d in a.drop.split(',') if d.strip()]
    for d in drop:
        if d not in MODALITIES:
            raise SystemExit(f'--drop {d}: choose from {MODALITIES}')
    PRIMARY['ordinal'] = a.los_metric
    log(f'device {cfg.DEVICE} | seed {a.seed} | lambda {cfg.EWC_LAMBDA} | ewc {cfg.USE_EWC} '
        f'({a.ewc_mode}) | tasks {tasks} | windows {[windows[t] for t in tasks]} | drop {drop or "none"}')

    # ---- data ----
    data, Ydim = {}, {}
    nfeat = None
    for t in tasks:
        data[t] = {}
        for s in ('train', 'val', 'test'):
            lf = os.path.join(data_dir, TASK_DIR[t], f'{s}_listfile.csv')
            if not os.path.exists(lf):
                lf = os.path.join(data_dir, TASK_DIR[t], f'{s}_listfile_flexcare.csv')
            mmd = os.path.join(mm_root, t, s)
            if not (os.path.exists(lf) and os.path.isdir(mmd)):
                raise SystemExit(f'{t}/{s}: need {lf} and {mmd}')
            y, dim = labels_from_listfile(lf, t, a.los_is_class)
            mod = Modal(mmd, lf, ehr_root, windows[t], a.resample, a.time_col, nfeat)
            nfeat = mod.F
            if len(y) != mod.N:
                raise SystemExit(f'{t}/{s}: {len(y)} labels but {mod.N} cached rows')
            data[t][s] = (mod, y)
            Ydim[t] = max(Ydim.get(t, 0), dim)
            log(f'  {t}/{s}: {mod.N:,} stays | T={mod.T} F={mod.F} K={mod.K} '
                f'| cxr {mod.cxr_mask.mean()*100:.1f}% notes {mod.notes_mask.mean()*100:.1f}% '
                f'concept {(mod.cmask.sum(1)>0).mean()*100:.1f}% demo {mod.demo_mask.mean()*100:.1f}%')

    ref = data[tasks[0]]['train'][0]
    dims = {'ehr': ref.F, 'cxr': ref.cxr_dim, 'text': ref.text_dim,
            'ent': ref.ent_dim, 'demo': ref.demo_dim}
    dims_out = {'PHE': Ydim.get('PHE', 25), 'DIA': Ydim.get('DIA', 25)}
    model = PatientGraphNet(cfg, dims, dims_out).to(cfg.DEVICE)
    log(f'model params {sum(p.numel() for p in model.parameters())/1e6:.2f} M | dims {dims} '
        f'| PHE {dims_out["PHE"]} DIA {dims_out["DIA"]}')

    logger = Logger(cfg.SAVE_DIR, {'tasks': tasks, 'windows': {t: windows[t] for t in tasks},
                                   'drop': drop, 'lambda': cfg.EWC_LAMBDA, 'seed': a.seed,
                                   'hidden': a.hidden, 'heads': a.heads, 'layers': a.layers,
                                   'batch_size': a.batch_size, 'lr': a.lr, 'ewc': cfg.USE_EWC,
                                   'ewc_mode': a.ewc_mode, 'tag': a.tag, 'dims': dims,
                                   'dims_out': dims_out})
    ema = EMA(model, cfg.EMA_DECAY)
    loaders = {t: {s: make_loader(data[t][s][0], data[t][s][1], t, s == 'train', cfg, drop)
                   for s in ('train', 'val', 'test')} for t in tasks}

    # ---- task-incremental stream ----
    K = len(tasks)
    R = np.full((K, K), np.nan)            # R[k, j]: test metric on task j after training stage k
    ewc_state, ewc_list = None, []
    for k, t in enumerate(tasks):
        log('=' * 72)
        log(f'STAGE {k+1}/{K}: {t}  (T={windows[t]}, epochs={epochs.get(t, 15)})')
        log('=' * 72)
        ewc_use = ewc_state if a.ewc_mode == 'online' else (ewc_list[-1] if ewc_list else None)
        model = train_task(cfg, model, t, loaders[t]['train'], loaders[t]['val'],
                           epochs.get(t, 15), ema, ewc_use, logger)
        ck = os.path.join(cfg.SAVE_DIR, f'best_{t}.pth')
        if os.path.exists(ck):
            model.load_state_dict(torch.load(ck, map_location=cfg.DEVICE)['model'])
        for j in range(k + 1):
            tj = tasks[j]
            thr = None
            if TASK_KIND[tj] == 'binary':                     # threshold from validation only
                thr = evaluate(model, loaders[tj]['val'], tj, cfg).get('threshold')
            m = evaluate(model, loaders[tj]['test'], tj, cfg, thr)
            R[k, j] = primary(tj, m)
            logger.dump(f'test_after_{t}_on_{tj}.json', m)
            log(f'  test[{tj}] {PRIMARY[TASK_KIND[tj]]}={R[k, j]:.4f}')
        logger.dump('R_matrix.json', cl_metrics(R, tasks, k + 1))
        torch.save({'model': model.state_dict(), 'stage': k, 'task': t, 'R': R.tolist()},
                   os.path.join(cfg.SAVE_DIR, f'stage{k+1}_{t}.pth'))
        if cfg.USE_EWC and k < K - 1:
            fl = make_loader(data[t]['train'][0], data[t]['train'][1], t, True, cfg, drop,
                             bs=1, persistent=False)
            e = EWC(model, fl, t, cfg, cfg.EWC_SAMPLES)
            ewc_list.append(e)
            ewc_state = copy.deepcopy(e) if ewc_state is None else ewc_state.merge(e)
        ema = EMA(model, cfg.EMA_DECAY)                       # reset EMA every stage

    out = cl_metrics(R, tasks)
    logger.dump('R_matrix.json', out)
    log('=' * 72)
    log('R matrix (rows = after stage, cols = task)')
    log('            ' + ' '.join(f'{t:>8s}' for t in tasks))
    for k, t in enumerate(tasks):
        log(f'  after {t:4s} ' + ' '.join(
            ('     nan' if np.isnan(R[k, j]) else f'{R[k, j]:8.4f}') for j in range(K)))
    log(f'AP  = {out["AP"]:.4f}')
    log(f'BWT = {out["BWT"]:.4f}')
    log(f'mean forgetting = {out["mean_forgetting"]:.4f}')
    for t, v in out['forgetting'].items():
        log(f'  forgetting[{t}] = {v:.4f}')
    log(f'saved -> {cfg.SAVE_DIR}')


if __name__ == '__main__':
    main()
