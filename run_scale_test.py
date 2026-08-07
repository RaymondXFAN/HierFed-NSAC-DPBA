"""
N=100 Scale Test — validates "Large-Scale" claim in paper title.

Reviewer A and B both flag N=20 as insufficient for "Large-Scale Residential IoT".
This script runs the proposed method with N=100, M=10, K=10 to show:
  1. Communication efficiency scales correctly (speedup = M/N × avg_CR → 57.6× holds)
  2. Model performance is similar or slightly worse at larger scale
  3. HierFed's hierarchical topology advantage increases with N

Runs with both α=0.5 and α=0.1, 3 seeds (enough for trend analysis).

Usage (AutoDL GPU):
  python run_scale_test.py --N 100 --seeds 1 2 3
  python run_scale_test.py --N 50 100 --seeds 1 2 3
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


def load_config():
    with open(str(ROOT / 'configs' / 'base.yaml'), 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    config['processed_dir'] = str(ROOT / 'data' / 'processed')
    return config


def run_scale_experiment(N, alpha, seed, epsilon=3.0, output_dir='results_scale'):
    """Run proposed method with given N (scale)."""
    M = max(N // 10, 2)  # M = N/10 gateways
    K = N // M  # K devices per gateway

    config = load_config()
    config['N'] = N
    config['M'] = M
    config['K'] = K
    config['alpha'] = alpha
    config['seed'] = seed
    config['epsilon_total'] = epsilon
    config['oversample_minority'] = False
    config['T'] = 50
    config['eta'] = 0.01

    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    exp_name = f"scale_N{N}_alpha{alpha}_seed{seed}"
    print(f"\n{'='*60}")
    print(f"Scale Test: N={N}, M={M}, K={K}, α={alpha}, seed={seed}")
    print(f"Device: {torch.cuda.get_device_name() if torch.cuda.is_available() else 'CPU'}")
    print(f"{'='*60}\n")

    start_time = datetime.now()

    try:
        sim = HierFedMatterSimulator(config)
        results = sim.run()
        elapsed = (datetime.now() - start_time).total_seconds()

        results['elapsed_seconds'] = elapsed
        results['exp_name'] = exp_name
        results['scale_config'] = {'N': N, 'M': M, 'K': K}

        # Save
        out_path = os.path.join(output_dir, exp_name)
        os.makedirs(out_path, exist_ok=True)
        with open(os.path.join(out_path, 'results.json'), 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, default=str)

        fm = results['final_metrics']
        cs = results.get('communication_stats', {})
        print(f"\n📊 {exp_name} ({elapsed:.0f}s):")
        print(f"   AUC={fm['auc_roc']:.4f} | macro-F1={fm['f1_macro']:.4f} | "
              f"Speedup={cs.get('speedup',0):.1f}×")

        return results

    except Exception as e:
        print(f"❌ Error N={N} α={alpha} seed={seed}: {e}")
        import traceback
        traceback.print_exc()
        return None


def aggregate_scale(output_dir='results_scale', Ns=[20, 50, 100], alphas=[0.5], seeds=[1,2,3]):
    """Aggregate scale test results."""
    print(f"\n{'='*70}")
    print("SCALE TEST SUMMARY")
    print(f"{'='*70}")
    print(f"| N | M | K | AUC(α=0.5) | macro-F1(α=0.5) | Speedup | Time(s) |")
    print(f"|---|---|---|:-----------:|:----------------:|:-------:|:-------:|")

    scale_data = {}
    for N in Ns:
        M = max(N // 10, 2)
        K = N // M
        aucs = []
        f1s = []
        speedups = []
        times = []

        for alpha in alphas:
            for seed in seeds:
                exp_name = f"scale_N{N}_alpha{alpha}_seed{seed}"
                result_file = os.path.join(output_dir, exp_name, 'results.json')
                if not os.path.exists(result_file):
                    continue
                with open(result_file, 'r') as f:
                    r = json.load(f)
                aucs.append(r['final_metrics']['auc_roc'])
                f1s.append(r['final_metrics']['f1_macro'])
                if 'communication_stats' in r:
                    speedups.append(r['communication_stats']['speedup'])
                times.append(r['elapsed_seconds'])

        key = f"N{N}"
        if aucs:
            scale_data[key] = {
                'N': N, 'M': M, 'K': K,
                'AUC_mean': np.mean(aucs), 'AUC_std': np.std(aucs),
                'macro-F1_mean': np.mean(f1s), 'macro-F1_std': np.std(f1s),
                'speedup_mean': np.mean(speedups) if speedups else 0,
                'time_mean': np.mean(times),
            }
            sd = scale_data[key]
            print(f"| {N} | {M} | {K} | {sd['AUC_mean']:.3f}±{sd['AUC_std']:.3f} | "
                  f"{sd['macro-F1_mean']:.3f}±{sd['macro-F1_std']:.3f} | "
                  f"{sd['speedup_mean']:.1f}× | {sd['time_mean']:.0f} |")
        else:
            print(f"| {N} | {M} | {K} | [TBD] | [TBD] | [TBD] | [TBD] |")

    with open(os.path.join(output_dir, 'scale_summary.json'), 'w', encoding='utf-8') as f:
        json.dump(scale_data, f, indent=2)

    return scale_data


def main():
    parser = argparse.ArgumentParser(description='Scale test (N≥100)')
    parser.add_argument('--N', type=int, nargs='+', default=[20, 50, 100],
                        help='Scale values (default: 20 50 100)')
    parser.add_argument('--alpha', type=float, nargs='+', default=[0.5],
                        help='Dirichlet α (default: 0.5)')
    parser.add_argument('--seeds', nargs='+', type=int, default=[1, 2, 3],
                        help='Seeds (default: 1 2 3)')
    parser.add_argument('--epsilon', type=float, default=3.0)
    parser.add_argument('--output_dir', default='results_scale')
    parser.add_argument('--aggregate_only', action='store_true')
    args = parser.parse_args()

    if not args.aggregate_only:
        if torch.cuda.is_available():
            print(f"✅ GPU: {torch.cuda.get_device_name()}")
        else:
            print(f"⚠️ No GPU — N=100 will be VERY slow on CPU")

        for N in args.N:
            for alpha in args.alpha:
                for seed in args.seeds:
                    run_scale_experiment(N, alpha, seed, args.epsilon, args.output_dir)

    aggregate_scale(args.output_dir, args.N, args.alpha, args.seeds)


if __name__ == '__main__':
    main()
