# HierFed-Matter-NSAC-DPBA

## One-sentence summary

> The optimal privacy configuration of a hierarchical federated intrusion detector is
> \\\*\\\*not independent of its compression setting\\\*\\\*: we measure that coupling across three
> datasets, and show that the simple uniform configuration is near-optimal while
> slice-level differentiation costs up to 3.9% AUC.

\---

## What this repository contains

|Component|Contents|
|-|-|
|**Method**|HierFed-Matter-NSAC-DPBA: hierarchical aggregation + network-slice-aware compression + Fisher-information-driven per-device privacy-budget allocation|
|**Raw results**|936 `results.json` files — every run behind every number in the paper|
|**Analysis**|Scripts for the CR × C interaction test, the four-arm slice-allocation measurement, and the annotation audit|
|**Reproduction**|`REPRODUCE.md` — environment, data preparation, every experiment, expected numbers|
|**Figures**|PNG (300 dpi) and PDF vectors under `figs/`, `figs\\\_dp/`|
|**Annotations**|Bilingual (Chinese original + English) comments in every script — see `AI\\\_USAGE.md`|

\---

## What is new in v0.3 (2026-10-18)

**A third dataset (TII23) was added, and one contaminated batch of runs was corrected.**

|Change|Detail|
|-|-|
|**+ Third dataset**|TII23 added; the four-arm allocation study now spans **3 datasets × 2 slice-weighting rules = 6 cells**|
|**+ 40 new runs**|`results\\\_alloc\\\_tii23` (20) and `results\\\_alloc\\\_tii23\\\_kr` (20)|
|\*\*\~ Corrected\*\*|The arm-A4 runs for seeds 6–8 on Gotham25 and IoTID20 had been executed with the wrong configuration switch and duplicated arm A3; they were re-run and replaced|
|**+ Stronger evidence**|Arm A4 is now the worst arm in **6/6 cells** and loses to A1 in **34 of 36 paired seeds** (binomial p ≈ 2×10⁻⁸)|
|\*\*\~ Reframed\*\*|The conclusion is no longer stated as a negative result but as a **bounded-gain / bounded-loss asymmetry**: the maximum achievable gain from slice-level differentiation is +0.15% AUC, while the maximum loss is −3.9% AUC|
|**+ Docs**|`README.md` and `REPRODUCE.md` updated with the three-dataset table|

\---

## Key results

|#|Result|Evidence|
|-|-|-|
|1|Uniform DP collapses at ε = 3|AUC → ≈ 0.5, macro-F1 → 0.484 (degenerate majority-class predictor)|
|2|DPBA-FIM recovers partial utility|AUC 0.5 → ≈ 0.59 at ε = 3 (α = 0.5), formal (ε, δ)-DP preserved under RDP composition|
|3|Monotone privacy–utility frontier|AUC ≈ 0.834 at ε = 10|
|4|Communication reduction|≈ 33× at the core network|
|5|**Compression–privacy coupling**|the optimal clipping bound C shifts with the compression ratio — **ANOVA F = 3.24, p = 0.0025** (n = 8, 120 runs)|
|6|**Bounded gain vs. bounded loss**|maximum slice-level gain **+0.15%** AUC; maximum loss **−3.9%** AUC (**26×** asymmetry)|

\---

## Two results tables

**Four-arm slice-allocation study (A1 = uniform reference)**

|Dataset|Weights|A1|A2 slice-CR|A3 slice-ε|A4 slice-both|Δ(A4−A1)|sign|
|-|-|-|-|-|-|-|-|
|Gotham25|bw|0.9957|0.9950|0.9972|**0.9930**|−0.0027|2/8|
|Gotham25|k\_ratio|0.9962|0.9952|0.9961|**0.9941**|−0.0021|0/5|
|IoTID20|bw|0.9719|0.9695|0.9715|**0.9567**|−0.0152|0/8|
|IoTID20|k\_ratio|0.9724|0.9695|0.9721|**0.9641**|−0.0083|0/5|
|TII23|bw|0.9903|0.9893|0.9820|**0.9519**|−0.0384|0/5|
|TII23|k\_ratio|0.9903|0.9893|0.9881|**0.9709**|−0.0194|0/5|

**Aggregate sign tests (paired seeds vs. A1)**

|Comparison|Positive|n|Binomial p (two-sided)|
|-|-|-|-|
|A4 − A1|2|36|**1.94 × 10⁻⁸**|
|A3 − A1|17|36|0.87 (neutral)|

\---

## Quick start

```bash
python -m pip install -r requirements.txt
python run\\\_experiment.py --method proposed --dataset gotham25 --alpha 0.5 \\\\
    --seed 1 --tag demo --T 50 --output\\\_dir results\\\_demo
```

See [**REPRODUCE.md**](REPRODUCE.md) for the full pipeline.

## AI usage

English annotations, docstrings and this document were produced with the assistance of
an AI writing assistant (**ima.copilot, Tencent; 2026-10**). All code logic, experiment
design and scientific conclusions are the author's own. See [**AI\_USAGE.md**](AI_USAGE.md).

