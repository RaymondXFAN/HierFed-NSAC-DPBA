"""
Run all 4 required baselines (Table 4) for both α=0.5 and α=0.1, 5 seeds each.

Baselines required by reviewers:
  1. FedAvg       — star-topology, no DP, no compression
  2. FedProx      — star-topology + proximal term (μ=0.01)
  3. DP-FedAvg    — star-topology + uniform DP (ε=3)
  4. HierFed+DP   — hierarchical + uniform DP (ε=3), no NSAC

Usage (AutoDL GPU):
  python run_baselines.py --method fedavg --alpha 0.5 --seeds 1 2 3 4 5
  python run_baselines.py --method all --alpha both --seeds 1 2 3 4 5
  python run_baselines.py --aggregate_only
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

from baselines.runner import run_baseline

# GPU info
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    print(f"[GPU] Using {torch.cuda.get_device_name()} | "
          f"VRAM={torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")
else:
    print("[GPU] No GPU available, running on CPU")


def load_config(config_path='configs/base.yaml'):
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    # Override processed_dir to absolute path
    config['processed_dir'] = str(ROOT / 'data' / 'processed')
    return config


def run_single(method, alpha, seed, config_base, output_dir='results'):
    """Run a single baseline experiment."""
    config = config_base.copy()
    config['seed'] = seed
    config['alpha'] = alpha
    config['dataset_name'] = 'iotid20'
    config['oversample_minority'] = False  # All baselines use plain CE (same as proposed method)
    config['T'] = 50  # Same number of rounds as proposed method
    config['N'] = 20
    config['M'] = 4
    config['K'] = 5

    # Set random seed
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    exp_name = f"{method}_iotid20_alpha{alpha}_seed{seed}"
    print(f"\n{'='*60}")
    print(f"Baseline: {exp_name}")
    print(f"Device: {torch.cuda.get_device_name() if torch.cuda.is_available() else 'CPU'}")
    print(f"{'='*60}\n")

    start_time = datetime.now()
    results = run_baseline(method, config)
    elapsed = (datetime.now() - start_time).total_seconds()

    # Add timing info
    results['elapsed_seconds'] = elapsed
    results['config'] = {
        'method': method,
        'alpha': alpha,
        'seed': seed,
        'oversample': False,
        'T': 50,
        'N': 20,
    }

    # Save per-seed results
    out_path = os.path.join(output_dir, exp_name)
    os.makedirs(out_path, exist_ok=True)

    with open(os.path.join(out_path, 'results.json'), 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, default=str)

    fm = results['final_metrics']
    print(f"\n📊 {exp_name} Summary (elapsed={elapsed:.0f}s):")
    print(f"   AUC = {fm['auc_roc']:.4f}  |  macro-F1 = {fm['f1_macro']:.4f}  |  "
          f"F1 = {fm['f1']:.4f}  |  thresh = {fm['optimal_threshold']:.2f}")

    return results


def aggregate_all(output_dir='results', methods=None, alphas=[0.5, 0.1], seeds=[1,2,3,4,5]):
    """Aggregate results across seeds for Table 4 format."""
    if methods is None:
        methods = ['fedavg', 'fedprox', 'dpfedavg', 'hierfed_dp']

    table_data = {}
    for method in methods:
        for alpha in alphas:
            all_fms = []
            for seed in seeds:
                exp_name = f"{method}_iotid20_alpha{alpha}_seed{seed}"
                result_file = os.path.join(output_dir, exp_name, 'results.json')
                if not os.path.exists(result_file):
                    print(f"  ⚠ Missing: {result_file}")
                    continue
                with open(result_file, 'r', encoding='utf-8') as f:
                    r = json.load(f)
                all_fms.append(r['final_metrics'])

            if not all_fms:
                print(f"  ⚠ No data for {method} α={alpha}")
                continue

            aucs = [m['auc_roc'] for m in all_fms]
            f1_macros = [m['f1_macro'] for m in all_fms]
            f1s = [m['f1'] for m in all_fms]

            key = f"{method}_alpha{alpha}"
            table_data[key] = {
                'AUC_mean': np.mean(aucs),
                'AUC_std': np.std(aucs),
                'AUC_median': np.median(aucs),
                'macro-F1_mean': np.mean(f1_macros),
                'macro-F1_std': np.std(f1_macros),
                'macro-F1_median': np.median(f1_macros),
                'F1_mean': np.mean(f1s),
                'F1_std': np.std(f1s),
                'per_seed': {
                    f'seed{s}': {'AUC': aucs[i], 'macro-F1': f1_macros[i], 'F1': f1s[i]}
                    for i, s in enumerate(seeds) if i < len(aucs)
                },
            }

            print(f"\n📊 {method} (α={alpha}): "
                  f"AUC={np.mean(aucs):.4f}±{np.std(aucs):.4f} "
                  f"(median={np.median(aucs):.4f})  |  "
                  f"macro-F1={np.mean(f1_macros):.4f}±{np.std(f1_macros):.4f} "
                  f"(median={np.median(f1_macros):.4f})")

    # Save table data
    with open(os.path.join(output_dir, 'table4_baselines.json'), 'w', encoding='utf-8') as f:
        json.dump(table_data, f, indent=2)

    # Print Table 4 format
    print(f"\n{'='*70}")
    print("TABLE 4: Baseline Performance Comparison")
    print(f"{'='*70}")
    print(f"| Method | AUC(α=0.5) | macro-F1(α=0.5) | AUC(α=0.1) | macro-F1(α=0.1) |")
    print(f"|--------|:-----------:|:----------------:|:-----------:|:----------------:|")
    for method in methods:
        a05 = table_data.get(f"{method}_alpha0.5", {})
        a01 = table_data.get(f"{method}_alpha0.1", {})
        auc05 = f"{a05.get('AUC_mean',0):.3f}±{a05.get('AUC_std',0):.3f}" if a05 else '[TBD]'
        mf05 = f"{a05.get('macro-F1_mean',0):.3f}±{a05.get('macro-F1_std',0):.3f}" if a05 else '[TBD]'
        auc01 = f"{a01.get('AUC_mean',0):.3f}±{a01.get('AUC_std',0):.3f}" if a01 else '[TBD]'
        mf01 = f"{a01.get('macro-F1_mean',0):.3f}±{a01.get('macro-F1_std',0):.3f}" if a01 else '[TBD]'
        print(f"| {method} | {auc05} | {mf05} | {auc01} | {mf01} |")

    return table_data


def main():
    parser = argparse.ArgumentParser(description='Run Table 4 baseline experiments')
    parser.add_argument('--method', choices=['fedavg', 'fedprox', 'dpfedavg', 'hierfed_dp', 'all'],
                        default='all', help='Which baseline to run')
    parser.add_argument('--alpha', type=float, nargs='+', default=[0.5, 0.1],
                        help='Dirichlet α values (default: 0.5 0.1)')
    parser.add_argument('--seeds', nargs='+', type=int, default=[1, 2, 3, 4, 5],
                        help='Random seeds (default: 1 2 3 4 5)')
    parser.add_argument('--output_dir', default='results', help='Results output dir')
    parser.add_argument('--aggregate_only', action='store_true',
                        help='Only aggregate existing results, skip running.')
    args = parser.parse_args()

    config_base = load_config()

    if args.method == 'all':
        methods = ['fedavg', 'fedprox', 'dpfedavg', 'hierfed_dp']
    else:
        methods = [args.method]

    if not args.aggregate_only:
        # Check GPU
        if torch.cuda.is_available():
            print(f"✅ GPU available: {torch.cuda.get_device_name()}")
            print(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        else:
            print(f"⚠️ No GPU, running on CPU (will be slow)")

        for method in methods:
            for alpha in args.alpha:
                for seed in args.seeds:
                    try:
                        run_single(method, alpha, seed, config_base, args.output_dir)
                    except Exception as e:
                        print(f"❌ Error in {method}_alpha{alpha}_seed{seed}: {e}")
                        import traceback
                        traceback.print_exc()

    # Aggregate
    aggregate_all(args.output_dir, methods, args.alpha, args.seeds)


if __name__ == '__main__':
    main()
