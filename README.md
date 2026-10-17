# HierFed-Matter-NSAC-DPBA

**Adaptive Fisher-Information-Based Privacy Budget Allocation for Hierarchical Federated Learning in 6G IoT Anomaly Detection**

Code, raw result artifacts and analysis tooling accompanying the paper.

---

## Overview

A three-layer hierarchical federated learning framework for privacy-preserving IoT
anomaly detection:

| Component | Role |
|---|---|
| **Matter topology** | hierarchical aggregation (device → gateway → core) |
| **NSAC** | network-slice-aware compression (slice-adaptive gradient quantization) |
| **DPBA-FIM** | dynamic per-device privacy-budget allocation driven by Empirical Fisher Information |

At strict privacy budgets (ε ≤ 3) uniform DP collapses to near-random prediction.
DPBA-FIM tracks per-device Fisher information and redistributes the noise budget
under a fixed (ε, δ) budget, recovering partial utility while preserving formal DP
guarantees under Rényi DP composition.

## Key findings

| # | Finding | Evidence |
|---|---|---|
| 1 | Uniform DP collapses at ε = 3 | AUC → ≈ 0.5, macro-F1 → 0.484 (degenerate majority-class predictor) |
| 2 | DPBA-FIM recovers partial utility | AUC 0.5 → ≈ 0.59 at ε = 3, α = 0.5 |
| 3 | Monotone privacy–utility frontier | AUC ≈ 0.834 at ε = 10 |
| 4 | Communication reduction | ≈ 33× at the core network |
| 5 | **CR × C interaction** | the optimal privacy budget ε\* shifts with the compression ratio C (**ANOVA F = 3.24, p = 0.0025**, n = 8, 120 runs) |
| 6 | **Slice-level allocation does not pay off** | the joint arm A4 is **worst in 6/6 cells** across **3 datasets** (IoTID20, Gotham25, TII23) × 2 slice-weighting rules; A4 loses to A1 in **34 of 36 paired seeds** (binomial p ≈ 2×10⁻⁸) |

## Repository layout

```
HierFed-Matter-NSAC-DPBA/
├── core/                     # simulator, DPBA-FIM, NSAC, hierarchical aggregation
├── baselines/                # baseline runners
├── models/                   # DNN backbone
├── configs/                  # YAML configurations
├── data/processed/           # dataset METADATA only (see "Data availability")
├── results_alloc/            # Phase-3 four-arm ablation, bw weights (64 runs)
├── results_alloc_kr/         # Phase-3 four-arm ablation, k_ratio weights (40 runs)
├── results_v9_crxC/          # Phase-2 CR × C grid (120 runs)
├── results_*/                # additional sweeps (ε, baselines, ablations)
├── figs/, figs_dp/           # figures (PNG + PDF)
├── run_experiment.py         # main training entry point
├�es.py          # baseline comparison
├── analyze_*.py              # analysis / statistics
├── run_*.sh, gate_*.sh       # experiment drivers (incl. pre-flight gates)
└── README.md, REPRODUCE.md, AI_USAGE.md, requirements.txt
```

## Quick start

Full step-by-step instructions — environment, data preparation, every experiment,
analysis and expected numbers — are in **[REPRODUCE.md](REPRODUCE.md)**.

## Data availability

The large preprocessed tensors (`data/processed/*.npz`, ≈ 725 MB; individual files
> 100 MB) are **not** bundled here because of hosting limits. Only lightweight
metadata (device ids, feature names, label maps, partition JSONs) is included.

The raw datasets are public, and the preprocessing / partitioning scripts are part
of this repository, so the tensors can be regenerated — see
[REPRODUCE.md §3](REPRODUCE.md).

## Bilingual annotations

Every script carries bilingual annotations: the author's **Chinese original** plus an
English counterpart (`# EN: ...` for line comments, `--- EN ---` blocks inside
docstrings). See [AI_USAGE.md](AI_USAGE.md).

## Citation

```bibtex
@article{hierfed_matter_nsac_dpba,
  title   = {Adaptive Fisher-Information-Based Privacy Budget Allocation for
             Hierarchical Federated Learning in 6G IoT Anomaly Detection},
  author  = {TODO},
  journal = {TODO},
  year    = {2026}
}
```

## License

TODO — specify before publication (e.g. MIT / Apache-2.0 / CC-BY-4.0).
0 / CC-BY-4.0).
