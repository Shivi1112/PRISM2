# PRISM2

# PRISM²: Patient-centric Graph for Multimodal Clinical Prediction

PyTorch implementation of **PRISM²**, a patient-centric heterogeneous graph model for multimodal
clinical prediction on MIMIC-IV. Each patient stay is turned into one graph whose nodes come from
whichever modalities are actually available, so partially missing modalities are handled by masking
and **no modality imputation** is needed. Six clinical tasks are learned one after another
(task-incremental learning) with Elastic Weight Consolidation (EWC).

## Model overview

For every stay a graph is built around a central **patient node**:

| Modality | Nodes | Local structure |
|---|---|---|
| EHR time series | one node per time step (`T`) | forward / backward temporal chain |
| Chest X-ray | 6 region nodes | spatial graph over regions (ipsilateral / bilateral edges) |
| Clinical notes | 1 note node | link to patient |
| Note concepts | `K` concept nodes (symptom / disease / treatment) | relational graph from `rel_adj` |
| Demographics | 1 node | link to patient |

Processing (`model.py`):

1. `FeatureEncoder` projects every modality to a common width.
2. `RelationalGAT` blocks run local intra-modality attention (temporal chain, CXR regions, concept relations).
3. `HGTLayer`s run global heterogeneous attention with relation-specific Q/K/V over the whole patient graph.
   Missing nodes are masked through an availability vector, so they send no messages.
4. `Heads` pools each modality with availability-masked means, fuses them with a gated unit and applies six task heads.

Tasks, in the default incremental order:

| Task | Meaning | Type | Primary metric |
|---|---|---|---|
| `IHM` | in-hospital mortality (first 48 h) | binary | AUROC, AUPRC |
| `LOS` | length of stay, 10 classes | ordinal  | macro-F1, micro=F1 |
| `PHE` | phenotyping | multi-label |  macro-AUROC, micro AUROC |
| `REA` | readmission | binary | AUROC, AUPRC |
| `DEC` | decompensation | binary | AUROC, AUPRC |
| `DIA` | diagnosis | multi-label | macro-AUROC, micro-AUROC |

LOS classes: days 0 to 7 (one class each), 8 to 14 days, 14+ days.

## Repository layout

```
main.py             training / evaluation entry point (incremental stream, EWC, result matrix)
model.py            graph network, task heads, losses, EWC, EMA
utils.py            constants, label parsing, EHR cache, graph batching, metrics, logging, config
requirements.txt
```

## Installation

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt     # install a CUDA build of PyTorch from pytorch.org if you use a GPU
```

Tested syntax-wise with Python 3.9+ and PyTorch 2.x.

## Data

An example dataset has been added in ihm.csv for in-hospital mortality task.

### 1. Listfiles and EHR time series (`--data_dir`, `--ehr_dir`)

```
<data_dir>/<task_dir>/{train,val,test}_listfile.csv     # or *_listfile_flexcare.csv
```

`task_dir` is one of `in_hospital_mortality`, `length_of_stay`, `phenotyping`, `readmission`,
`decompensation`, `diagnosis`. Each listfile needs:

### 2. Pre-extracted multimodal features (`--mm_dir`)

For every task and split, `<mm_dir>/<TASK>/<split>/` (e.g. `mm/IHM/train/`) contains, with `N` stays:

## Usage

Full run (all six tasks in the default order):

```bash
python main.py \
    --data_dir ./data/mimic4 --mm_dir ./data/prism2_mm --save_dir ./runs/prism2 \
    --device cuda:0
```




