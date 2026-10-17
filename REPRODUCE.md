# REPRODUCE.md — Full Reproduction Guide

Step-by-step instructions to reproduce every experiment and figure reported in the
paper. All commands are written for a POSIX shell (Linux). Tested on a single
NVIDIA RTX 4090 (24 GB) with Python 3.10.

> The framework is abbreviated below as **the model**. Its three components are the
> Matter topology, **NSAC** (network-slice-aware compression) and **DPBA-FIM**
> (Fisher-information-driven privacy-budget allocation).

---

## 1. Environment

```bash
# Python 3.10 is recommended (tested). Create an isolated environment:
conda create -n hierfed python=3.10 -y
conda activate hierfed

# PyTorch with CUDA 12.4 (adjust to your driver; cu124 works with CUDA ≥ 12.4 builds)
pip install torch --index-url https://download.pytorch.org/whl/cu124

# Remaining dependencies
pip install -r requirements.txt
```

`requirements.txt` covers: `numpy`, `scipy`, `scikit-learn`, `pandas`, `matplotlib`, `torch`.

Verify the GPU is visible:

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

Expected: a `True` on the second value. The framework runs on a single GPU; four
concurrent processes per GPU is a good default for the sweeps below.

---

## 2. Repository layout (quick reference)

| Path | Purpose |
|---|---|
| `run_experiment.py` | main training entry point (one run per invocation) |
| `core/simulator.py` | hierarchical FL simulation loop |
| `core/dpba_fim.py` | DPBA-FIM privacy-budget allocation |
| `core/nsac.py` | network-slice-aware compression |
| `core/hierfed.py` | hierarchical aggregation |
| `core/slice_alloc.py` | slice assignment + per-slice budget policies |
| `configs/base.yaml` | default configuration |
| `configs/nsac.yaml` | NSAC configuration |
| `data/processed/` | dataset tensors and partition metadata |
| `results_*/` | raw per-run results (`results.json`) |

---

## 3. Data preparation

### 3.1 Raw datasets

* **IoTID20** — IoT intrusion-detection dataset, 625,783 records, 79 features.
  Fetch it with the bundled helper:

  ```bash
  bash get_iotid20.sh
  ```

  The script tries three sources in order (Kaggle full set → HuggingFace subset →
  figshare) and writes to `data/raw/`.

* **Gotham25** — packet-level IoT traffic with a **per-device** file layout. Place the
  raw files under `data/raw/` and preprocess them with `prep_dataset.py`, which
  probes the schema, one-hot encodes categorical columns, applies a NaN policy and
  builds a *native* device-level (physically non-IID) partition:

  ```bash
  python prep_dataset.py --csv "data/raw/gotham25/*.csv" --name gotham25 \
      --alphas 0.5 0.1 --nan-mode zero --native-n 25 --seed 1
  ```

  Use `--inspect` first to review the detected schema and header consistency.

### 3.2 Tensors and partitions

Preprocessing produces the tensors consumed by `run_experiment.py`:

```
data/processed/<dataset>_train.npz
data/processed/<dataset>_test.npz
```

Dirichlet partitions for α = 0.5 and α = 0.1 are written as JSON
(`<dataset>_partitions_alpha0.5.json`, …). **These JSON files and the device/label
metadata are already included in this repository** (`data/processed/`), so you only
need to regenerate the `.npz` tensors if you re-download the raw data.

Sanity check before running anything:

```bash
python - <<'PY'
import numpy as np, glob
for f in sorted(glob.glob("data/processed/*_train.npz")):
    d = np.load(f)
    print(f, {k: d[k].shape for k in d.files})
PY
```

---

## 4. A single run

One run = one (dataset, α, seed, configuration) combination. Results are written to
`<output_dir>/<dataset>/<method>_<dataset>_alpha<α>_seed<seed>[_<tag>]/results.json`.

```bash
python run_experiment.py \
    --method proposed \
    --dataset gotham25 \
    --alpha 0.5 \
    --seed 1 \
    --tag demo \
    --T 50 \
    --output_dir results_demo
```

### 4.1 Main CLI flags

| Flag | Meaning |
|---|---|
| `--method` | `proposed`, `fedavg`, `fedprox`, `dpfedavg`, `hierfed_dp`, … |
| `--dataset` | `iotid20`, `cicids2017`, `gotham25`, `tii23`, `both` |
| `--alpha` | Dirichlet concentration (non-IID level) |
| `--seed` | random seed |
| `--T` | number of federated rounds |
| `--e_local` | local epochs per round |
| `--config` | path to the base YAML (default `configs/base.yaml`) |
| `--nsac_config` | path to the NSAC YAML (default `configs/nsac.yaml`) |
| `--clip_norm` | global gradient-clipping bound **C** |
| `--dp_clip_mode` | `global` \| `per_device` (**C only takes effect in `global` mode**) |
| `--as --bs --gs --ds` | DPBA-FIM sensitivity weights α_s, β_s, γ_s, δ_s |
| `--disable_dpba_fim` | ablation: turn off DPBA-FIM |
| `--disable_nsac` | ablation: turn off NSAC |
| `--disable_hierfed` | ablation: collapse to flat FL |
| `--disable_dp` | ablation: turn off differential privacy |
| `--output_dir`, `--tag` | output location / run label |

### 4.2 Environment switches

Some levers are read from the environment rather than the CLI:

| Variable | Values | Meaning |
|---|---|---|
| `NSAC_MODE` | `uniform` \| `slice` | compression-ratio policy across slices |
| `EPSILON_MODE` | `uniform` \| `slice` \| `adaptive` | privacy-budget policy across slices |
| `SLICE_WEIGHT_RULE` | `bw` \| `k_ratio` \| `equal` | slice weighting rule |
| `SLICE_MAP` | `ranked` | slice-assignment strategy |
| `NSAC_CR_SCALE` | float | multiplies each slice's compression ratio |
| `HIERFED_DATASET` | dataset key | forwarded to the simulator |

Practical rule: always export `DP_CLIP_MODE=global` when sweeping the clipping bound
**C**, otherwise the sweep is a no-op.

---

## 5. Pre-flight gate (always run this first)

Every driver below starts with a short **gate**: a `T = 2` smoke run that must write
a valid `results.json` (and a non-degenerate slice mapping) before the full sweep is
launched. Run the standalone cross-dataset gate:

```bash
bash gate_phase1.sh
```

Do not start a long sweep until the gate passes. This catches data-loading errors,
degenerate configurations and result-path collisions in ~2 minutes instead of hours.

---

## 6. Experiment 1 — Privacy–utility frontier (ε-sweep)

Reproduces the monotone frontier and the ε = 3 collapse.

```bash
for EPS in 2 3 5 8 10; do
  # set the target ε in configs/base.yaml (epsilon_total), then:
  python run_experiment.py --method proposed --dataset iotid20 \
      --alpha 0.5 --seed 1 --T 50 --output_dir results_eps/eps${EPS}
done
```

Ablation of DPBA-FIM (used for the "removing it collapses back to ≈ 0.5" claim):

```bash
python run_experiment.py --method proposed --dataset iotid20 --alpha 0.5 \
    --seed 1 --T 50 --disable_dpba_fim --output_dir results_ablation/nodpba
```

**Expected** (paper): uniform DP at ε = 3 → AUC ≈ 0.5, macro-F1 ≈ 0.484; with
DPBA-FIM → AUC ≈ 0.59; AUC ≈ 0.834 at ε = 10.

---

## 7. Experiment 2 — CR × C grid (Phase 2, the go/no-go)

**Question:** does the optimal clipping bound **C** shift with the compression ratio?

```bash
# 3 CR levels × 5 C values × N seeds; defaults: CR_SCALES="0.25 1.0 4.0", CS="0.05 0.1 0.2 0.5 1.0"
PAR=4 SEEDS="1 2 3 4 5 6 7 8" ALPHA=0.5 DP_CLIP_MODE=global OUT=results_v9_crxC \
    bash run_p2_cr_x_c.sh
```

Analysis (interaction test, paired tests, sign consistency):

```bash
python analyze_p2_grid.py results_v9_crxC
```

**Expected** (paper): significant **CR × C interaction**, ANOVA **F = 3.24,
p = 0.0025** (n = 8, 120 runs); the default C = 1.0 is sub-optimal by
+0.087…+0.091 AUC on the optimal-C rows, with per-seed sign consistency.

---

## 8. Experiment 3 — Four-arm slice-allocation ablation (Phase 3)

The four arms are the product of two binary switches — per-slice compression ratio
(CR) and per-slice privacy budget (ε):

| Arm | CR policy | ε policy |
|---|---|---|
| **A1** | uniform | uniform |
| **A2** | slice | uniform |
| **A3** | uniform | slice |
| **A4** | slice | slice |

Two weighting rules are run (extreme `bw` and mild `k_ratio`):

```bash
# extreme slice weights  -> results_alloc   (4 arms × 2 datasets × 8 seeds = 64 runs)
PAR=4 SEEDS="1 2 3 4 5" bash run_p3_alloc.sh
SEEDS="6 7 8" bash seed678.sh                 # top-up seeds 6–8

# mild slice weights     -> results_alloc_kr (4 arms × 2 datasets × 5 seeds = 40 runs)
PAR=8 bash run_p3_k_ratio.sh
```

Analysis (4-arm means, paired t / Wilcoxon, per-seed sign table):

```bash
python final_check.py results_alloc results_alloc_kr
```

A third dataset is run with the generic driver, which handles both weighting rules:

```bash
DS=tii23 PAR=8 bash add_third_dataset.sh     # -> results_alloc_tii23 / results_alloc_tii23_kr
```

**Expected** (paper): **A4 is the worst arm in all 6 cells** (3 datasets × 2 rules), and
loses to A1 in **34 of 36 paired seeds** (binomial two-sided p ≈ 2×10⁻⁸).

| Dataset | Weights | A1 | A2 | A3 | A4 | Δ(A4−A1) | sign |
|---|---|---|---|---|---|---|---|
| Gotham25 | bw | 0.9957 | 0.9950 | 0.9972 | **0.9930** | −0.0027 | 2/8 |
| Gotham25 | k_ratio | 0.9962 | 0.9952 | 0.9961 | **0.9941** | −0.0021 | 0/5 |
| IoTID20 | bw | 0.9719 | 0.9695 | 0.9715 | **0.9567** | **−0.0152** | 0/8 |
| IoTID20 | k_ratio | 0.9724 | 0.9695 | 0.9721 | **0.9641** | −0.0083 | 0/5 |
| TII23 | bw | 0.9903 | 0.9893 | 0.9820 | **0.9519** | −0.0384 | 0/5 |
| TII23 | k_ratio | 0.9903 | 0.9893 | 0.9881 | **0.9709** | −0.0194 | 0/5 |

The extreme `bw` rule inflates A4's variance but the A4 disadvantage persists under
the milder `k_ratio` rule — it is systematic, not noise. A3 (ε-only slicing) is
**neutral overall** (17/36 seeds positive, p = 0.87) but **dataset-dependent**: it
helps on Gotham25/bw (ΔAUC = +0.0015, 7/8 seeds, p = 0.039) and hurts on TII23/bw
(ΔAUC = −0.0083, 0/5 seeds, p = 0.011).

---

## 9. Figures

```bash
python plot_results.py            # main result figures
python plot_dp_clusters.py        # privacy/utility cluster view
```

Generated PNG (300 dpi) and PDF vectors are also checked in under `figs/` and
`figs_dp/` for reference.

---

## 10. Repository health checks

```bash
python verify_repo.py .            # syntax check + Chinese/annotation inventory
python untranslated.py .           # list any Chinese comment lacking an English pair
```

Both tools are read-only and safe to run at any time.

---

## 11. Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `ModuleNotFoundError: core.*` | run from the repository root (the entry points insert it on `sys.path`) |
| A sweep finishes suspiciously fast with **no** `results.json` | a driver reported success on exit code alone. Always verify by the **existence** of the output file, not the exit code |
| `C` sweep shows no effect | `DP_CLIP_MODE` is not `global` — export `DP_CLIP_MODE=global` |
| Slice mapping collapses to a single slice | use `SLICE_MAP=ranked`; check the gate output for `slices=[...]` with ≥ 3 entries |
| `SLICE_WEIGHT_RULE=equal` with equal-size slices → all arms identical | expected degeneracy; use `bw` or `k_ratio` to get a real contrast |
| Out of memory | reduce `PAR` (concurrency); each run is small but four per GPU is a safe default |
| NaN in results | the run is invalid; check the run's log and re-run (drivers are idempotent) |

---

## 12. Result file format

Each `results.json` contains two blocks:

```jsonc
{
  "config":        { /* resolved configuration for this run */ },
  "final_metrics": { "auc_roc": 0.9938, "macro_f1": 0.97, /* ... */ }
  // plus per-round time series when present
}
```

Analysis scripts read `final_metrics.auc_roc` as the primary metric.

---

## 13. AI usage note

The author's native language is Chinese; English comments, docstrings and this
document were produced with the assistance of an AI writing assistant
(**ima.copilot, Tencent; 2026-10**). All code logic, experiment design and
scientific conclusions are the author's own. See [AI_USAGE.md](AI_USAGE.md).
