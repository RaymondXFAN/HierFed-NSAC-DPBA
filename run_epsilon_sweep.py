"""
Privacy-Utility Tradeoff Curve: ε-sweep experiment.

Runs HierFed-Matter-NSAC-DPBA-FIM (proposed method) with ε = 0.5, 1, 2, 3, 5, 8, 10
for both α=0.5 and α=0.1, 5 seeds each.

This is THE core experiment demanded by reviewers — the paper title promises
"Privacy-Utility Tradeoff" but currently has NO ε-sweep curve.

Usage (AutoDL GPU):
  python run_epsilon_sweep.py --alpha 0.5 --seeds 1 4 5
  python run_epsilon_sweep.py --alpha both --seeds 1 4 5
"""

import os
import sys
import json
import argparse
import numpy as np
import torch
import yaml
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from core.simulator import HierFedMatterSimulator

# ============================================================
# GPU Support
# ============================================================
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    print(f"[GPU] Using {torch.cuda.get_device_name()} | "
          f"VRAM={torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")
else:
    print("[GPU] No GPU available, running on CPU")


def load_config(config_path='configs/base.yaml'):
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    config['processed_dir'] = str(ROOT / 'data' / 'processed')
    return config


EPSILON_VALUES = [2.0, 3.0, 5.0, 8.0, 10.0]


def run_single_epsilon(epsilon, alpha, seed, config_base, output_dir='results'):
    """Run one ε-sweep experiment."""
    config = config_base.copy()
    config['seed'] = seed
    config['alpha'] = alpha
    config['dataset_name'] = 'iotid20'
    config['epsilon_total'] = epsilon
    config['oversample_minority'] = False  # Same as v6 proposed method
    config['T'] = 15
    config['N'] = 20
    config['M'] = 4
    config['K'] = 5

    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    exp_name = f"epsweep_eps{epsilon}_alpha{alpha}_seed{seed}"
    print(f"\n{'='*60}")
    print(f"ε-sweep: ε={epsilon}, α={alpha}, seed={seed}")
    print(f"Device: {torch.cuda.get_device_name() if torch.cuda.is_available() else 'CPU'}")
    print(f"{'='*60}\n")

    start_time = datetime.now()
    sim = HierFedMatterSimulator(config)
    results = sim.run()
    elapsed = (datetime.now() - start_time).total_seconds()

    results['elapsed_seconds'] = elapsed
    results['exp_name'] = exp_name

    # Save
    out_path = os.path.join(output_dir, exp_name)
    os.makedirs(out_path, exist_ok=True)

    with open(os.path.join(out_path, 'results.json'), 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, default=str)

    fm = results['final_metrics']
    print(f"\n📊 {exp_name} (elapsed={elapsed:.0f}s):")
    print(f"   AUC={fm['auc_roc']:.4f} | macro-F1={fm['f1_macro']:.4f} | "
          f"thresh={fm['optimal_threshold']:.2f}")

    return results


def aggregate_sweep(output_dir='results', alphas=[0.5, 0.1], seeds=[1,2,3,4,5],
                    epsilons=EPSILON_VALUES):
    """Aggregate ε-sweep results into privacy-utility curve data."""
    curve_data = {}

    for alpha in alphas:
        eps_points = []
        for eps in epsilons:
            aucs = []
            f1_macros = []
            for seed in seeds:
                exp_name = f"epsweep_eps{eps}_alpha{alpha}_seed{seed}"
                result_file = os.path.join(output_dir, exp_name, 'results.json')
                if not os.path.exists(result_file):
                    continue
                with open(result_file, 'r', encoding='utf-8') as f:
                    r = json.load(f)
                aucs.append(r['final_metrics']['auc_roc'])
                f1_macros.append(r['final_metrics']['f1_macro'])

            if aucs:
                eps_points.append({
                    'epsilon': eps,
                    'AUC_mean': np.mean(aucs),
                    'AUC_std': np.std(aucs),
                    'AUC_median': np.median(aucs),
                    'macro-F1_mean': np.mean(f1_macros),
                    'macro-F1_std': np.std(f1_macros),
                    'macro-F1_median': np.median(f1_macros),
                    'n_seeds': len(aucs),
                    'per_seed_AUC': aucs,
                    'per_seed_macroF1': f1_macros,
                })

        curve_data[f'alpha{alpha}'] = eps_points

        # Print curve
        print(f"\n{'='*60}")
        print(f"Privacy-Utility Tradeoff Curve (α={alpha})")
        print(f"{'='*60}")
        print(f"| ε | AUC (mean±std) | AUC (median) | macro-F1 (mean±std) | macro-F1 (median) |")
        print(f"|---|:--------------:|:------------:|:-------------------:|:----------------:|")
        for pt in eps_points:
            print(f"| {pt['epsilon']} | {pt['AUC_mean']:.3f}±{pt['AUC_std']:.3f} | "
                  f"{pt['AUC_median']:.3f} | {pt['macro-F1_mean']:.3f}±{pt['macro-F1_std']:.3f} | "
                  f"{pt['macro-F1_median']:.3f} |")

    # Save curve data
    with open(os.path.join(output_dir, 'privacy_utility_curve.json'), 'w', encoding='utf-8') as f:
        json.dump(curve_data, f, indent=2)

    return curve_data


def main():
    parser = argparse.ArgumentParser(description='Run ε-sweep experiment')
    parser.add_argument('--alpha', type=float, nargs='+', default=[0.5, 0.1],
                        help='Dirichlet α values')
    parser.add_argument('--seeds', nargs='+', type=int, default=[1, 2, 3, 4, 5],
                        help='Random seeds')
    parser.add_argument('--epsilons', type=float, nargs='+', default=EPSILON_VALUES,
                        help='ε values to sweep')
    parser.add_argument('--output_dir', default='results', help='Output dir')
    parser.add_argument('--aggregate_only', action='store_true')
    args = parser.parse_args()

    config_base = load_config()

    if not args.aggregate_only:
        if torch.cuda.is_available():
            print(f"✅ GPU: {torch.cuda.get_device_name()} ({torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB)")
        else:
            print(f"⚠️ No GPU — CPU mode")

        for alpha in args.alpha:
            for eps in args.epsilons:
                for seed in args.seeds:
                    try:
                        run_single_epsilon(eps, alpha, seed, config_base, args.output_dir)
                    except Exception as e:
                        print(f"❌ Error eps={eps} α={alpha} seed={seed}: {e}")
                        import traceback
                        traceback.print_exc()

    # Aggregate
    aggregate_sweep(args.output_dir, args.alpha, args.seeds, args.epsilons)


if __name__ == '__main__':
    main()
