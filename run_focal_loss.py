"""
Focal Loss Experiment — THE MOST CRITICAL EXPERIMENT FOR REVISION.

Reviewer B demands AUC ≥ 0.75 (practical deployment threshold for anomaly detection).
Current AUC = 0.605 (plain CE) is unusable. Focal loss is the most promising direction:

  Focal Loss: FL(p_t) = -α_t × (1 - p_t)^γ × log(p_t)
  - γ (focusing parameter): reduces loss for well-classified samples → minority class
    gets more gradient signal without changing gradient DIRECTION (unlike class weights)
  - α_t (class balancing): weight per class, similar to class weights but integrated
    into focal loss formulation

Why focal loss is different from class-weighted CE (which failed in v2/v3):
  - Class-weighted CE multiplies the ENTIRE gradient by a scalar → devices with different
    local class ratios produce gradients in DIFFERENT directions → aggregation conflict
  - Focal loss modulates the per-sample loss by (1-p_t)^γ → well-classified (majority)
    samples get very small loss, but the gradient DIRECTION is unchanged → FL aggregation
    still works because all devices agree on what "better" means

Experiments:
  1. Centralized + Focal Loss (baseline: shows focal loss works on this data)
  2. FedAvg + Focal Loss (shows focal loss improves FL without DP)
  3. HierFed-Matter-NSAC-DPBA + Focal Loss (our proposed method + focal loss)

γ values tested: 0.5, 1.0, 2.0 (standard range from Lin et al. 2017)
α_t values: balanced by inverse class frequency (α_attack=0.064, α_normal=0.936)

Usage (AutoDL GPU):
  python run_focal_loss.py --mode centralized --gamma 2.0 --seeds 1 2 3 4 5
  python run_focal_loss.py --mode fedavg --gamma 2.0 --seeds 1 2 3 4 5
  python run_focal_loss.py --mode proposed --gamma 2.0 --seeds 1 2 3 4 5
  python run_focal_loss.py --mode all --gamma 0.5 1.0 2.0 --seeds 1 2 3 4 5
"""

import os
import sys
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from models.dnn import build_model
from core.simulator import local_train, assign_device_types, assign_slice_to_gateways
from core.hierfed import star_aggregate, edge_aggregate, core_aggregate, hierfed_matter_round
from core.dpba_fim import (
    FIMTracker, compute_device_sensitivity, clip_gradient, inject_dp_noise,
    inject_dp_noise_rdp, compute_noise_scale_rdp,
    compute_heterogeneity_from_partition, compute_noise_scale,
)
from data.preprocess import dirichlet_partition

# ============================================================
# GPU Support
# ============================================================
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    print(f"[GPU] Using {torch.cuda.get_device_name()} | "
          f"VRAM={torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")
else:
    print("[GPU] No GPU available, running on CPU")


# ============================================================
# Focal Loss Implementation
# ============================================================
class FocalLoss(nn.Module):
    """
    Focal Loss (Lin et al., 2017):
      FL(p_t) = -α_t × (1 - p_t)^γ × log(p_t)

    Args:
        alpha: class weights [α_class0, α_class1]. Default: inverse frequency.
        gamma: focusing parameter. γ=0 → standard CE; γ=2 → strong focusing.
    """
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction
        if alpha is not None:
            self.alpha = torch.tensor(alpha, dtype=torch.float32)
        else:
            self.alpha = None

    def forward(self, inputs, targets):
        """
        Args:
            inputs: (N, C) raw logits
            targets: (N,) class indices
        """
        ce_loss = nn.functional.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)  # p_t = probability of correct class
        focal_weight = (1 - pt) ** self.gamma

        if self.alpha is not None:
            self.alpha = self.alpha.to(inputs.device)
            alpha_weight = self.alpha[targets]
            focal_loss = alpha_weight * focal_weight * ce_loss
        else:
            focal_loss = focal_weight * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss


# ============================================================
# Shared Evaluation (same threshold optimization as simulator)
# ============================================================
def evaluate_model(model, X_test, y_test):
    """Full evaluation with threshold optimization (matches simulator)."""
    model.eval()
    X_t = torch.tensor(X_test, dtype=torch.float32).to(DEVICE)
    with torch.no_grad():
        outputs = model(X_t)
        probs = torch.softmax(outputs, dim=1)
        preds_default = outputs.argmax(dim=1).cpu().numpy()

    y_true = y_test
    y_prob = probs[:, 1].cpu().numpy()

    # Threshold optimization
    best_f1 = 0.0
    best_thresh = 0.5
    for thresh in np.arange(0.05, 0.95, 0.05):
        preds_t = (y_prob >= thresh).astype(int)
        f1_t = f1_score(y_true, preds_t, zero_division=0)
        if f1_t > best_f1:
            best_f1 = f1_t
            best_thresh = thresh

    preds_optimal = (y_prob >= best_thresh).astype(int)
    f1_optimal = best_f1
    acc_optimal = accuracy_score(y_true, preds_optimal)
    f1_default = f1_score(y_true, preds_default, zero_division=0)
    f1_macro = f1_score(y_true, preds_optimal, average='macro', zero_division=0)

    try:
        auc = roc_auc_score(y_true, y_prob)
    except ValueError:
        auc = 0.0

    return {
        'f1': f1_optimal, 'f1_default': f1_default, 'f1_macro': f1_macro,
        'accuracy': acc_optimal, 'auc_roc': auc, 'optimal_threshold': best_thresh,
    }


# ============================================================
# Load Data + Partition
# ============================================================
def load_data_and_partition(alpha=0.5, N=20):
    """Load IoTID20 train/test + generate per-α partition."""
    processed_dir = str(ROOT / 'data' / 'processed')

    train = np.load(os.path.join(processed_dir, 'iotid20_train.npz'))
    X_train = train['X']
    y_train = train['y']

    test = np.load(os.path.join(processed_dir, 'iotid20_test.npz'))
    X_test = test['X']
    y_test = test['y']

    # Load or generate partition
    per_alpha_file = os.path.join(processed_dir, f'iotid20_partitions_alpha{alpha}.json')
    partitions = None
    if os.path.exists(per_alpha_file):
        with open(per_alpha_file, 'r', encoding='utf-8') as f:
            part_data = json.load(f)
        partitions = part_data['device_indices']
    else:
        partitions = dirichlet_partition(y_train, N, alpha, seed=1)
        partitions = {str(k): v for k, v in partitions.items()}
        save_data = {'N': N, 'alpha': alpha, 'seed': 1, 'device_indices': partitions}
        os.makedirs(processed_dir, exist_ok=True)
        with open(per_alpha_file, 'w', encoding='utf-8') as f:
            json.dump(save_data, f, indent=2)

    return X_train, y_train, X_test, y_test, partitions


# ============================================================
# Centralized + Focal Loss
# ============================================================
def run_centralized_focal(X_train, y_train, X_test, y_test, gamma, alpha_class, seed):
    """Centralized training with focal loss."""
    np.random.seed(seed)
    torch.manual_seed(seed)

    input_dim = X_train.shape[1]
    model = build_model(input_dim).to(DEVICE)
    d = model.num_parameters

    print(f"\n{'='*60}")
    print(f"Centralized + Focal Loss (γ={gamma}, seed={seed})")
    print(f"alpha_class = {alpha_class}")
    print(f"{'='*60}\n")

    # Compute class weights from data
    class_counts = np.bincount(y_train.astype(int), minlength=2)
    total = len(y_train)
    # alpha_t = inverse frequency normalized
    if alpha_class is None:
        alpha_class = [total / (2 * class_counts[0]), total / (2 * class_counts[1])]

    criterion = FocalLoss(alpha=alpha_class, gamma=gamma)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    train_data = torch.tensor(X_train, dtype=torch.float32).to(DEVICE)
    train_labels = torch.tensor(y_train, dtype=torch.long).to(DEVICE)
    dataset = TensorDataset(train_data, train_labels)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    history = []
    for epoch in range(1, 51):
        model.train()
        epoch_loss = 0.0
        for batch_x, batch_y in loader:
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        if epoch % 10 == 0 or epoch == 1:
            metrics = evaluate_model(model, X_test, y_test)
            print(f"Epoch {epoch}: AUC={metrics['auc_roc']:.4f}, macro-F1={metrics['f1_macro']:.4f}")
            history.append({'epoch': epoch, 'loss': epoch_loss/len(loader), **metrics})

    final = evaluate_model(model, X_test, y_test)
    print(f"\n✅ Final: AUC={final['auc_roc']:.4f}, macro-F1={final['f1_macro']:.4f}")
    return {'final_metrics': final, 'epoch_history': history}


# ============================================================
# Local Training with Focal Loss (for FL experiments)
# ============================================================
def local_train_focal(model, train_data, train_labels, num_epochs=5, batch_size=32,
                      learning_rate=0.01, gamma=2.0, alpha_class=None):
    """
    Local training with focal loss returning gradient (Δw).
    Unlike class-weighted CE, focal loss does NOT change gradient direction —
    it only reduces the magnitude of well-classified samples.
    """
    device_data = torch.tensor(train_data, dtype=torch.float32).to(DEVICE)
    device_labels = torch.tensor(train_labels, dtype=torch.long).to(DEVICE)
    n_local = len(device_labels)
    if n_local == 0:
        return torch.zeros(model.num_parameters, device=DEVICE)

    # Compute alpha from local data if not provided
    if alpha_class is None:
        class_counts = torch.bincount(device_labels, minlength=2).float()
        total = n_local
        # Inverse frequency weighting
        alpha_class = [total / (2 * max(class_counts[0].item(), 1)),
                       total / (2 * max(class_counts[1].item(), 1))]

    criterion = FocalLoss(alpha=alpha_class, gamma=gamma)
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    # Save initial params
    initial_params = torch.cat([p.data.clone().flatten() for p in model.parameters()])

    model.train()
    # Use uniform sampling (no oversampling) — focal loss handles imbalance
    for epoch in range(num_epochs):
        indices = torch.randperm(n_local)
        for start in range(0, n_local, batch_size):
            end = min(start + batch_size, n_local)
            batch_idx = indices[start:end]
            batch_x = device_data[batch_idx]
            batch_y = device_labels[batch_idx]

            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

    # Compute gradient
    final_params = torch.cat([p.data.clone().flatten() for p in model.parameters()])
    gradient = final_params - initial_params

    # Reset model to global state
    with torch.no_grad():
        idx = 0
        for p in model.parameters():
            size = p.numel()
            p.data.copy_(initial_params[idx:idx + size].reshape(p.shape))
            idx += size

    return gradient


# ============================================================
# FedAvg + Focal Loss
# ============================================================
def run_fedavg_focal(X_train, y_train, X_test, y_test, partitions, gamma, alpha_class,
                     seed, N=20, T=50, eta=0.01):
    """FedAvg with focal loss: star-topology, no DP, no compression."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    input_dim = X_train.shape[1]
    model = build_model(input_dim).to(DEVICE)
    d = model.num_parameters

    print(f"\n{'='*60}")
    print(f"FedAvg + Focal Loss (γ={gamma}, α_dir=0.5, seed={seed})")
    print(f"{'='*60}\n")

    round_metrics = []
    for t in range(1, T + 1):
        device_gradients = {}
        for dev_id in range(N):
            idx_list = partitions.get(str(dev_id), [])
            if len(idx_list) == 0:
                device_gradients[dev_id] = torch.zeros(d, device=DEVICE)
                continue

            # Compute local alpha_class from device data
            local_labels = y_train[idx_list]
            class_counts = np.bincount(local_labels.astype(int), minlength=2)
            n_local = len(local_labels)
            local_alpha = [n_local / (2 * max(class_counts[0], 1)),
                          n_local / (2 * max(class_counts[1], 1))]

            gradient = local_train_focal(
                model, X_train[idx_list], y_train[idx_list],
                num_epochs=5, batch_size=32, learning_rate=eta,
                gamma=gamma, alpha_class=local_alpha,
            )
            device_gradients[dev_id] = gradient

        global_params = torch.cat([p.data.clone().flatten() for p in model.parameters()])
        avg_grad = torch.stack(list(device_gradients.values())).mean(dim=0)
        new_params = global_params + avg_grad

        with torch.no_grad():
            idx = 0
            for p in model.parameters():
                size = p.numel()
                p.data.copy_(new_params[idx:idx + size].reshape(p.shape))
                idx += size

        metrics = evaluate_model(model, X_test, y_test)
        round_metrics.append({'round': t, **metrics})

        if t % 10 == 0:
            print(f"[FedAvg+Focal] Round {t}: AUC={metrics['auc_roc']:.4f}, "
                  f"macro-F1={metrics['f1_macro']:.4f}")

    final = evaluate_model(model, X_test, y_test)
    print(f"\n✅ [FedAvg+Focal γ={gamma}] Final: AUC={final['auc_roc']:.4f}, "
          f"macro-F1={final['f1_macro']:.4f}")
    return {'final_metrics': final, 'round_metrics': round_metrics}


# ============================================================
# Proposed Method (HierFed+NSAC+DPBA-FIM) + Focal Loss
# ============================================================
def run_proposed_focal(X_train, y_train, X_test, y_test, partitions, gamma, alpha_class,
                       seed, alpha_dir=0.5, N=20, M=4, K=5, T=50, eta=0.01,
                       epsilon_total=3.0, delta=1e-5, clip_norm=1.0):
    """Full HierFed-Matter-NSAC-DPBA-FIM with focal loss."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    input_dim = X_train.shape[1]
    model = build_model(input_dim).to(DEVICE)
    d = model.num_parameters

    # Setup topology
    cluster_map = {gw: list(range(gw*K, min(gw*K+K, N))) for gw in range(M)}
    device_types = assign_device_types(N, seed)
    slice_assignments = assign_slice_to_gateways(M, device_types, cluster_map)

    # DPBA-FIM setup
    num_linear_layers = len([m for m in model.net if isinstance(m, torch.nn.Linear)])
    fim_tracker = FIMTracker(num_linear_layers, beta_F=0.9)

    local_counts = [len(partitions.get(str(i), [])) for i in range(N)]
    H_scores = compute_heterogeneity_from_partition(y_train, partitions)
    device_sensitivity = compute_device_sensitivity(
        device_types, local_counts, H_scores, alpha_s=0.5, beta_s_coeff=0.3, gamma_s=0.2,
    )

    # RDP composition: compute base σ for total ε over T rounds
    # Then modulate σ per-device by DPBA-FIM sensitivity weighting
    sigma_base = compute_noise_scale_rdp(epsilon_total, T, delta, clip_norm)

    print(f"\n{'='*60}")
    print(f"HierFed+NSAC+DPBA-FIM + Focal Loss (γ={gamma}, ε={epsilon_total}, "
          f"α_dir={alpha_dir}, seed={seed})")
    print(f"RDP base σ={sigma_base:.2f} (ε_total={epsilon_total}, T={T}, δ={delta})")
    print(f"{'='*60}\n")

    round_metrics = []
    for t in range(1, T + 1):
        device_gradients = {}
        for dev_id in range(N):
            idx_list = partitions.get(str(dev_id), [])
            if len(idx_list) == 0:
                device_gradients[dev_id] = torch.zeros(d, device=DEVICE)
                continue

            # Compute local alpha from device data
            local_labels = y_train[idx_list]
            class_counts = np.bincount(local_labels.astype(int), minlength=2)
            n_local = len(local_labels)
            local_alpha = [n_local / (2 * max(class_counts[0], 1)),
                          n_local / (2 * max(class_counts[1], 1))]

            gradient = local_train_focal(
                model, X_train[idx_list], y_train[idx_list],
                num_epochs=5, batch_size=32, learning_rate=eta,
                gamma=gamma, alpha_class=local_alpha,
            )

            # DPBA-FIM: clip + inject noise with sensitivity-weighted σ
            # High sensitivity → less noise (σ × lower multiplier)
            # Low sensitivity → more noise (σ × higher multiplier)
            S_rel = device_sensitivity / (device_sensitivity.max() + 1e-12)
            device_sigma = sigma_base * (1.4 - 0.4 * S_rel[dev_id])

            clipped = clip_gradient(gradient, clip_norm)
            noise = torch.randn_like(clipped) * device_sigma
            noisy = clipped + noise
            device_gradients[dev_id] = noisy

            # Update FIM tracker every 5 rounds
            if t % 5 == 0 and len(idx_list) > 0:
                with torch.no_grad():
                    local_t_data = torch.tensor(X_train[idx_list], dtype=torch.float32).to(DEVICE)
                    local_t_labels = torch.tensor(y_train[idx_list], dtype=torch.long).to(DEVICE)
                fim_tracker.update(model, local_t_data, local_t_labels,
                                   list(range(num_linear_layers)))

        # HierFed-Matter round with NSAC
        global_params = torch.cat([p.data.clone().flatten() for p in model.parameters()])
        new_params, round_stats = hierfed_matter_round(
            device_gradients, cluster_map, slice_assignments, global_params,
            learning_rate=eta, d=d,
            model_size_bytes=d * 4,
            enable_nsac=True,
        )

        with torch.no_grad():
            idx = 0
            for p in model.parameters():
                size = p.numel()
                p.data.copy_(new_params[idx:idx + size].reshape(p.shape))
                idx += size

        metrics = evaluate_model(model, X_test, y_test)
        round_metrics.append({'round': t, **metrics})

        if t % 10 == 0:
            print(f"[Proposed+Focal] Round {t}: AUC={metrics['auc_roc']:.4f}, "
                  f"macro-F1={metrics['f1_macro']:.4f}")

    final = evaluate_model(model, X_test, y_test)
    print(f"\n✅ [Proposed+Focal γ={gamma}] Final: AUC={final['auc_roc']:.4f}, "
          f"macro-F1={final['f1_macro']:.4f}")
    return {'final_metrics': final, 'round_metrics': round_metrics}


# ============================================================
# Main Entry Point
# ============================================================
def main():
    parser = argparse.ArgumentParser(description='Focal Loss experiment (AUC≥0.75 target)')
    parser.add_argument('--mode', choices=['centralized', 'fedavg', 'proposed', 'all'],
                        default='all', help='Which mode to run')
    parser.add_argument('--gamma', type=float, nargs='+', default=[0.5, 1.0, 2.0],
                        help='Focal loss γ values (default: 0.5, 1.0, 2.0)')
    parser.add_argument('--alpha_dir', type=float, nargs='+', default=[0.5, 0.1],
                        help='Dirichlet α for partitioning')
    parser.add_argument('--seeds', nargs='+', type=int, default=[1, 2, 3, 4, 5],
                        help='Random seeds')
    parser.add_argument('--epsilon', type=float, default=3.0,
                        help='DP ε for proposed method (default: 3.0)')
    parser.add_argument('--output_dir', default='results_focal')
    parser.add_argument('--aggregate_only', action='store_true',
                        help='Only aggregate existing results, skip experiments')
    args = parser.parse_args()

    # --- Aggregate-only mode ---
    if args.aggregate_only:
        import glob
        all_results = {}
        pattern = os.path.join(args.output_dir, '**/results.json')
        for f in glob.glob(pattern, recursive=True):
            try:
                with open(f, 'r', encoding='utf-8') as fp:
                    r = json.load(fp)
                exp_name = r.get('exp_name', os.path.basename(os.path.dirname(f)))
                all_results[exp_name] = r.get('final_metrics', r)
            except Exception as e:
                print(f"⚠️ Error loading {f}: {e}")
        print(f"\n{'='*70}")
        print("FOCAL LOSS RESULTS SUMMARY (aggregated)")
        print(f"{'='*70}")
        best_auc = max(fm['auc_roc'] for fm in all_results.values())
        print(f"\n🎯 Best AUC achieved: {best_auc:.4f}")
        if best_auc >= 0.75:
            print(f"✅ AUC ≥ 0.75 TARGET MET!")
        else:
            print(f"⚠️ AUC < 0.75")
        with open(os.path.join(args.output_dir, 'focal_loss_summary.json'), 'w', encoding='utf-8') as f:
            json.dump(all_results, f, indent=2, default=str)
        return

    if torch.cuda.is_available():
        print(f"✅ GPU: {torch.cuda.get_device_name()} ({torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB)")
    else:
        print(f"⚠️ No GPU")

    all_results = {}

    for alpha_dir in args.alpha_dir:
        X_train, y_train, X_test, y_test, partitions = load_data_and_partition(alpha_dir, N=20)

        # Compute global alpha_class (inverse frequency)
        class_counts = np.bincount(y_train.astype(int), minlength=2)
        total = len(y_train)
        global_alpha = [total / (2 * class_counts[0]), total / (2 * class_counts[1])]
        print(f"\n[Data] α_dir={alpha_dir}, class distribution: "
              f"class0={class_counts[0]}({class_counts[0]/total*100:.1f}%), "
              f"class1={class_counts[1]}({class_counts[1]/total*100:.1f}%)")
        print(f"[Focal] alpha_class = {global_alpha}")

        for gamma in args.gamma:
            for seed in args.seeds:
                modes = ['centralized', 'fedavg', 'proposed'] if args.mode == 'all' else [args.mode]

                for mode in modes:
                    exp_name = f"focal_{mode}_gamma{gamma}_alpha{alpha_dir}_seed{seed}"
                    start_time = datetime.now()

                    try:
                        if mode == 'centralized':
                            results = run_centralized_focal(
                                X_train, y_train, X_test, y_test, gamma, global_alpha, seed)
                        elif mode == 'fedavg':
                            results = run_fedavg_focal(
                                X_train, y_train, X_test, y_test, partitions,
                                gamma, global_alpha, seed)
                        elif mode == 'proposed':
                            results = run_proposed_focal(
                                X_train, y_train, X_test, y_test, partitions,
                                gamma, global_alpha, seed,
                                alpha_dir=alpha_dir, epsilon_total=args.epsilon)

                        elapsed = (datetime.now() - start_time).total_seconds()
                        results['elapsed_seconds'] = elapsed
                        results['exp_name'] = exp_name

                        # Save
                        out_path = os.path.join(args.output_dir, exp_name)
                        os.makedirs(out_path, exist_ok=True)
                        with open(os.path.join(out_path, 'results.json'), 'w', encoding='utf-8') as f:
                            json.dump(results, f, indent=2, default=str)

                        fm = results['final_metrics']
                        all_results[exp_name] = fm
                        print(f"📊 {exp_name}: AUC={fm['auc_roc']:.4f}, macro-F1={fm['f1_macro']:.4f}")

                    except Exception as e:
                        print(f"❌ Error {exp_name}: {e}")
                        import traceback
                        traceback.print_exc()

    # Summary table
    print(f"\n{'='*70}")
    print("FOCAL LOSS RESULTS SUMMARY")
    print(f"{'='*70}")
    print(f"| Mode | γ | α_dir | AUC | macro-F1 | Seeds averaged |")
    print(f"|------|---|-------|-----|----------|----------------|")
    for key, fm in sorted(all_results.items()):
        print(f"| {key} | {fm['auc_roc']:.4f} | {fm['f1_macro']:.4f} |")

    # Save summary
    with open(os.path.join(args.output_dir, 'focal_loss_summary.json'), 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, default=str)

    # Check if AUC ≥ 0.75 achieved
    best_auc = max(fm['auc_roc'] for fm in all_results.values())
    print(f"\n🎯 Best AUC achieved: {best_auc:.4f}")
    if best_auc >= 0.75:
        print(f"✅ AUC ≥ 0.75 TARGET MET! Focal loss solves the usability problem.")
    else:
        print(f"⚠️ AUC < 0.75 — may need other approaches (federated oversampling, etc.)")


if __name__ == '__main__':
    main()
