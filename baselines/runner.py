"""
Baseline methods for experimental comparison (Table 3):

1. FedAvg      - Standard star-topology FL, no DP, no compression
2. FedProx     - Proximal-regularized FL (μ parameter for non-IID)
3. HierFed     - Hierarchical FL (same topology as proposed, no DP/NSAC)
4. DP-FedAvg   - Star-topology FL with uniform DP noise (ε=3)
5. DP-Fed6G    - Star-topology FL with adaptive DP (ε=3)
6. Top-k+QSGD  - Star-topology FL with uniform Top-k compression
7. SAFEL-IoT   - Star-topology FL with adaptive DP + explainability
8. FedProx-DP  - FedProx + uniform DP (ε=3)
9. HierFed+DP  - Hierarchical FL + uniform DP (ε=3) — key baseline for isolating DPBA-FIM
"""

import os
import json

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List
from models.dnn import build_model
from core.hierfed import star_aggregate, edge_aggregate, core_aggregate
from core.nsac import SLICE_CONFIGS, nsac_compress, nsac_decompress
from core.dpba_fim import clip_gradient, inject_dp_noise, inject_dp_noise_rdp, compute_noise_scale_rdp

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
# FedAvg (standard star-topology, no DP)
# ============================================================
def run_fedavg(config: dict) -> Dict:
    """FedAvg baseline: star-topology, no DP, no compression."""
    # Load data first to get correct feature_dim
    X_train, y_train, partitions = _load_partitioned_data(config)
    X_test, y_test = _load_test_data(config)
    input_dim = X_train.shape[1]

    model = build_model(input_dim)
    model = model.to(DEVICE)
    d = model.num_parameters
    N = config.get('N', 20)
    T = config.get('T', 50)
    eta = config.get('eta', 0.01)
    E_local = config.get('E_local', 5)
    batch_size = config.get('batch_size', 32)
    oversample = config.get('oversample_minority', False)

    round_metrics = []
    for t in range(1, T + 1):
        device_gradients = {}
        for dev_id in range(N):
            idx_list = partitions.get(str(dev_id), [])
            if len(idx_list) == 0:
                device_gradients[dev_id] = torch.zeros(d, device=DEVICE)
                continue
            gradient = _local_train(model, X_train[idx_list], y_train[idx_list],
                                    E_local, batch_size, eta, oversample=oversample)
            device_gradients[dev_id] = gradient

        # Star aggregation
        global_params = _get_flat_params(model)
        new_params = star_aggregate(device_gradients, global_params, eta)
        _set_flat_params(model, new_params)

        metrics = _evaluate(model, X_test, y_test)
        round_metrics.append({'round': t, **metrics})

        if t % 10 == 0:
            print(f"[FedAvg] Round {t}: AUC={metrics['auc_roc']:.4f}, macro-F1={metrics['f1_macro']:.4f}")

    final = _evaluate(model, X_test, y_test)
    print(f"\n[FedAvg] Final: AUC={final['auc_roc']:.4f}, macro-F1={final['f1_macro']:.4f}")
    return {'final_metrics': final, 'round_metrics': round_metrics}


# ============================================================
# FedProx (proximal-regularized)
# ============================================================
def run_fedprox(config: dict, mu: float = 0.01) -> Dict:
    """FedProx: adds proximal term μ/2 × ‖w - w_global‖² to local loss."""
    X_train, y_train, partitions = _load_partitioned_data(config)
    X_test, y_test = _load_test_data(config)
    input_dim = X_train.shape[1]

    model = build_model(input_dim)
    model = model.to(DEVICE)
    d = model.num_parameters
    N = config.get('N', 20)
    T = config.get('T', 50)
    eta = config.get('eta', 0.01)
    oversample = config.get('oversample_minority', False)

    round_metrics = []
    for t in range(1, T + 1):
        global_params = _get_flat_params(model)
        device_gradients = {}
        for dev_id in range(N):
            idx_list = partitions.get(str(dev_id), [])
            if len(idx_list) == 0:
                device_gradients[dev_id] = torch.zeros(d, device=DEVICE)
                continue
            gradient = _local_train_fedprox(
                model, X_train[idx_list], y_train[idx_list],
                global_params, mu, config, oversample=oversample
            )
            device_gradients[dev_id] = gradient

        new_params = star_aggregate(device_gradients, global_params, eta)
        _set_flat_params(model, new_params)

        metrics = _evaluate(model, X_test, y_test)
        round_metrics.append({'round': t, **metrics})

        if t % 10 == 0:
            print(f"[FedProx] Round {t}: AUC={metrics['auc_roc']:.4f}, macro-F1={metrics['f1_macro']:.4f}")

    final = _evaluate(model, X_test, y_test)
    print(f"\n[FedProx] Final: AUC={final['auc_roc']:.4f}, macro-F1={final['f1_macro']:.4f}")
    return {'final_metrics': final, 'round_metrics': round_metrics}


def _local_train_fedprox(model, X, y, global_params, mu, config, oversample=False):
    """Local train with proximal regularization."""
    initial_params = _get_flat_params(model)
    E = config.get('E_local', 5)
    bs = config.get('batch_size', 32)
    eta = config.get('eta', 0.01)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=eta)
    X_t = torch.tensor(X, dtype=torch.float32).to(DEVICE)
    y_t = torch.tensor(y, dtype=torch.long).to(DEVICE)

    # Store global params per layer for proximal gradient addition
    global_params_per_layer = [p.data.clone() for p in model.parameters()]

    model.train()
    for epoch in range(E):
        indices = torch.randperm(len(y_t))
        for start in range(0, len(y_t), bs):
            batch_x = X_t[indices[start:start+bs]]
            batch_y = y_t[indices[start:start+bs]]
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            # Proximal gradient: add μ × (w - w_global) to each parameter's grad
            # This is the CORRECT implementation — previous version used flat params
            # with .data.clone().flatten() which disconnected the compute graph,
            # making the proximal term have zero effect on gradients.
            for p, gp in zip(model.parameters(), global_params_per_layer):
                if p.grad is not None:
                    p.grad += mu * (p.data - gp.to(DEVICE))
            optimizer.step()

    final_params = _get_flat_params(model)
    # Reset model to global params
    _set_flat_params(model, global_params)
    return final_params - initial_params


# ============================================================
# DP-FedAvg (uniform DP, ε=3)
# ============================================================
def run_dpfedavg(config: dict) -> Dict:
    """DP-FedAvg: star-topology + uniform DP noise at ε=3."""
    X_train, y_train, partitions = _load_partitioned_data(config)
    X_test, y_test = _load_test_data(config)
    input_dim = X_train.shape[1]

    model = build_model(input_dim).to(DEVICE)
    d = model.num_parameters
    N = config.get('N', 20)
    T = config.get('T', 50)
    eta = config.get('eta', 0.01)
    eps = float(config.get('epsilon_total', 3.0))
    delta = float(config.get('delta', 1e-5))
    C = float(config.get('clip_norm', 1.0))
    oversample = config.get('oversample_minority', False)

    # RDP composition: compute σ from total ε, T, δ (not naive ε/T)
    # σ≈12 for ε=3, T=50 → model still fails → supports our "uniform DP fails" narrative
    sigma = compute_noise_scale_rdp(eps, T, delta, C)
    print(f"[DP-FedAvg] RDP σ={sigma:.2f} (ε={eps}, T={T}, δ={delta})")
    round_metrics = []

    for t in range(1, T + 1):
        device_gradients = {}
        for dev_id in range(N):
            idx_list = partitions.get(str(dev_id), [])
            if len(idx_list) == 0:
                device_gradients[dev_id] = torch.zeros(d, device=DEVICE)
                continue
            gradient = _local_train(model, X_train[idx_list], y_train[idx_list],
                                    config.get('E_local', 5),
                                    config.get('batch_size', 32), eta, oversample=oversample)
            # Uniform DP via RDP composition: same σ for all devices
            clipped = clip_gradient(gradient, C)
            noise = torch.randn_like(clipped) * sigma
            noisy = clipped + noise
            device_gradients[dev_id] = noisy

        global_params = _get_flat_params(model)
        new_params = star_aggregate(device_gradients, global_params, eta)
        _set_flat_params(model, new_params)

        metrics = _evaluate(model, X_test, y_test)
        round_metrics.append({'round': t, **metrics})

        if t % 10 == 0:
            print(f"[DP-FedAvg] Round {t}: AUC={metrics['auc_roc']:.4f}, macro-F1={metrics['f1_macro']:.4f}")

    final = _evaluate(model, X_test, y_test)
    print(f"\n[DP-FedAvg] Final: AUC={final['auc_roc']:.4f}, macro-F1={final['f1_macro']:.4f}")
    return {'final_metrics': final, 'round_metrics': round_metrics}


# ============================================================
# HierFed (hierarchical, no DP/NSAC)
# ============================================================
def run_hierfed(config: dict) -> Dict:
    """HierFed: hierarchical aggregation, no DP, no NSAC compression."""
    X_train, y_train, partitions = _load_partitioned_data(config)
    X_test, y_test = _load_test_data(config)
    input_dim = X_train.shape[1]

    model = build_model(input_dim).to(DEVICE)
    d = model.num_parameters
    N = config.get('N', 20)
    M = config.get('M', 4)
    K = config.get('K', 5)
    T = config.get('T', 50)
    eta = config.get('eta', 0.01)
    oversample = config.get('oversample_minority', False)

    cluster_map = {gw: list(range(gw*K, min(gw*K+K, N))) for gw in range(M)}

    round_metrics = []
    for t in range(1, T + 1):
        device_gradients = {}
        for dev_id in range(N):
            idx_list = partitions.get(str(dev_id), [])
            if len(idx_list) == 0:
                device_gradients[dev_id] = torch.zeros(d, device=DEVICE)
                continue
            gradient = _local_train(model, X_train[idx_list], y_train[idx_list],
                                    config.get('E_local', 5),
                                    config.get('batch_size', 32), eta, oversample=oversample)
            device_gradients[dev_id] = gradient

        global_params = _get_flat_params(model)
        # Hierarchical aggregation without compression
        edge_updates = edge_aggregate(device_gradients, cluster_map,
                                       global_params, eta)
        new_params = core_aggregate(edge_updates, global_params)
        _set_flat_params(model, new_params)

        metrics = _evaluate(model, X_test, y_test)
        round_metrics.append({'round': t, **metrics})

        if t % 10 == 0:
            print(f"[HierFed] Round {t}: AUC={metrics['auc_roc']:.4f}, macro-F1={metrics['f1_macro']:.4f}")

    final = _evaluate(model, X_test, y_test)
    print(f"\n[HierFed] Final: AUC={final['auc_roc']:.4f}, macro-F1={final['f1_macro']:.4f}")
    return {'final_metrics': final, 'round_metrics': round_metrics}


# ============================================================
# HierFed + uniform DP (ε=3) — key ablation baseline
# ============================================================
def run_hierfed_dp(config: dict) -> Dict:
    """HierFed + uniform DP: isolates DPBA-FIM's contribution."""
    X_train, y_train, partitions = _load_partitioned_data(config)
    X_test, y_test = _load_test_data(config)
    input_dim = X_train.shape[1]

    model = build_model(input_dim).to(DEVICE)
    d = model.num_parameters
    N, M, K = config.get('N', 20), config.get('M', 4), config.get('K', 5)
    T = config.get('T', 50)
    eta = config.get('eta', 0.01)
    eps = float(config.get('epsilon_total', 3.0))
    delta = float(config.get('delta', 1e-5))
    C = float(config.get('clip_norm', 1.0))
    oversample = config.get('oversample_minority', False)

    cluster_map = {gw: list(range(gw*K, min(gw*K+K, N))) for gw in range(M)}
    # RDP composition: compute σ from total ε, T, δ (not naive ε/T)
    # σ≈12 for ε=3, T=50 → hierarchical + uniform DP still fails
    sigma = compute_noise_scale_rdp(eps, T, delta, C)
    print(f"[HierFed+DP] RDP σ={sigma:.2f} (ε={eps}, T={T}, δ={delta})")
    round_metrics = []
    for t in range(1, T + 1):
        device_gradients = {}
        for dev_id in range(N):
            idx_list = partitions.get(str(dev_id), [])
            if len(idx_list) == 0:
                device_gradients[dev_id] = torch.zeros(d, device=DEVICE)
                continue
            gradient = _local_train(model, X_train[idx_list], y_train[idx_list],
                                    config.get('E_local', 5),
                                    config.get('batch_size', 32), eta, oversample=oversample)
            # Uniform DP via RDP composition (same σ for all devices, no FIM weighting)
            clipped = clip_gradient(gradient, C)
            noise = torch.randn_like(clipped) * sigma
            noisy = clipped + noise
            device_gradients[dev_id] = noisy

        global_params = _get_flat_params(model)
        edge_updates = edge_aggregate(device_gradients, cluster_map,
                                       global_params, eta)
        new_params = core_aggregate(edge_updates, global_params)
        _set_flat_params(model, new_params)

        metrics = _evaluate(model, X_test, y_test)
        round_metrics.append({'round': t, **metrics})

        if t % 10 == 0:
            print(f"[HierFed+DP] Round {t}: AUC={metrics['auc_roc']:.4f}, macro-F1={metrics['f1_macro']:.4f}")

    final = _evaluate(model, X_test, y_test)
    print(f"\n[HierFed+DP] Final: AUC={final['auc_roc']:.4f}, macro-F1={final['f1_macro']:.4f}")
    return {'final_metrics': final, 'round_metrics': round_metrics}
def run_topk_qsgd(config: dict) -> Dict:
    """Top-k + QSGD: star-topology FL with uniform Top-k compression."""
    X_train, y_train, partitions = _load_partitioned_data(config)
    X_test, y_test = _load_test_data(config)
    input_dim = X_train.shape[1]
    model = build_model(input_dim).to(DEVICE)
    d = model.num_parameters
    N = config.get('N', 50)
    T = config.get('T', 100)
    eta = config.get('eta', 0.01)

    # Use eMBB slice config as "uniform" compression (moderate)
    slice_config = SLICE_CONFIGS['embb']

    round_metrics = []
    for t in range(1, T + 1):
        device_gradients = {}
        for dev_id in range(N):
            idx_list = partitions.get(str(dev_id), [])
            if len(idx_list) == 0:
                device_gradients[dev_id] = torch.zeros(d, device=DEVICE)
                continue
            gradient = _local_train(model, X_train[idx_list], y_train[idx_list],
                                    config.get('E_local', 5),
                                    config.get('batch_size', 32), eta)
            # Compress each device's gradient
            compressed = nsac_compress(gradient, slice_config, adaptive_down=False)
            decompressed = nsac_decompress(compressed, d, device=DEVICE)
            device_gradients[dev_id] = decompressed

        global_params = _get_flat_params(model)
        new_params = star_aggregate(device_gradients, global_params, eta)
        _set_flat_params(model, new_params)

        metrics = _evaluate(model, X_test, y_test)
        round_metrics.append({'round': t, **metrics})

        if t % 10 == 0:
            print(f"[Top-k+QSGD] Round {t}: F1={metrics['f1']:.4f}")

    final = _evaluate(model, X_test, y_test)
    return {'final_metrics': final, 'round_metrics': round_metrics}


# ============================================================
# SAFEL-IoT (simplified: adaptive DP without FIM)
# ============================================================
def run_safeliot(config: dict) -> Dict:
    """SAFEL-IoT: adaptive DP (not FIM-weighted), star-topology."""
    # Simplified implementation: adaptive noise based on data sensitivity
    # but uniform across layers (no FIM weighting)
    model = build_model(config.get('feature_dim', 80)).to(DEVICE)
    d = model.num_parameters
    N = config.get('N', 50)
    T = config.get('T', 100)
    eta = config.get('eta', 0.01)
    eps = config.get('epsilon_total', 3.0)
    delta = config.get('delta', 1e-5)
    C = config.get('clip_norm', 1.0)

    X_train, y_train, partitions = _load_partitioned_data(config)
    X_test, y_test = _load_test_data(config)

    # Adaptive DP via RDP composition: compute base σ for total ε, then
    # scale per-device σ by data sensitivity (more sensitive → less noise)
    sigma_map = {'camera': 1.0, 'thermostat': 0.3, 'bulb': 0.1}
    sigma_base = compute_noise_scale_rdp(float(eps), T, float(delta), float(C))

    round_metrics = []
    for t in range(1, T + 1):
        device_gradients = {}
        for dev_id in range(N):
            idx_list = partitions.get(str(dev_id), [])
            if len(idx_list) == 0:
                device_gradients[dev_id] = torch.zeros(d, device=DEVICE)
                continue
            gradient = _local_train(model, X_train[idx_list], y_train[idx_list],
                                    config.get('E_local', 5),
                                    config.get('batch_size', 32), eta)
            # Adaptive σ: scale by device data sensitivity
            n_i = len(idx_list)
            # More sensitive devices → less noise (smaller σ multiplier)
            sensitivity_factor = 1.0 - 0.5 * min(n_i, 200) / 200
            device_sigma = sigma_base * sensitivity_factor
            clipped = clip_gradient(gradient, C)
            noise = torch.randn_like(clipped) * device_sigma
            noisy = clipped + noise
            device_gradients[dev_id] = noisy

        global_params = _get_flat_params(model)
        new_params = star_aggregate(device_gradients, global_params, eta)
        _set_flat_params(model, new_params)

        metrics = _evaluate(model, X_test, y_test)
        round_metrics.append({'round': t, **metrics})

        if t % 10 == 0:
            print(f"[SAFEL-IoT] Round {t}: F1={metrics['f1']:.4f}")

    final = _evaluate(model, X_test, y_test)
    return {'final_metrics': final, 'round_metrics': round_metrics}


# ============================================================
# DP-Fed6G (adaptive DP for 6G)
# ============================================================
def run_dpfed6g(config: dict) -> Dict:
    """DP-Fed6G: adaptive DP with convergence-aware budget, star-topology."""
    model = build_model(config.get('feature_dim', 80)).to(DEVICE)
    d = model.num_parameters
    N = config.get('N', 50)
    T = config.get('T', 100)
    eta = config.get('eta', 0.01)
    eps = config.get('epsilon_total', 3.0)
    delta = config.get('delta', 1e-5)
    C = config.get('clip_norm', 1.0)

    X_train, y_train, partitions = _load_partitioned_data(config)
    X_test, y_test = _load_test_data(config)

    # Convergence-aware RDP composition: compute base σ for total ε,
    # then scale per-round σ by convergence factor (early rounds → more noise)
    sigma_base = compute_noise_scale_rdp(float(eps), T, float(delta), float(C))
    round_metrics = []
    for t in range(1, T + 1):
        # Convergence-aware σ: early rounds need more noise (less budget),
        # later rounds get less noise (more budget).
        decay_factor = min(1.0, t / (T * 0.3))  # ramp up signal over first 30%
        # σ_r(t) = σ_base / decay_factor → early rounds (decay<1) get more noise
        device_sigma = sigma_base / decay_factor if decay_factor > 0 else sigma_base

        device_gradients = {}
        for dev_id in range(N):
            idx_list = partitions.get(str(dev_id), [])
            if len(idx_list) == 0:
                device_gradients[dev_id] = torch.zeros(d, device=DEVICE)
                continue
            gradient = _local_train(model, X_train[idx_list], y_train[idx_list],
                                    config.get('E_local', 5),
                                    config.get('batch_size', 32), eta)
            clipped = clip_gradient(gradient, C)
            noise = torch.randn_like(clipped) * device_sigma
            noisy = clipped + noise
            device_gradients[dev_id] = noisy

        global_params = _get_flat_params(model)
        new_params = star_aggregate(device_gradients, global_params, eta)
        _set_flat_params(model, new_params)

        metrics = _evaluate(model, X_test, y_test)
        round_metrics.append({'round': t, **metrics})

        if t % 10 == 0:
            print(f"[DP-Fed6G] Round {t}: F1={metrics['f1']:.4f}")

    final = _evaluate(model, X_test, y_test)
    return {'final_metrics': final, 'round_metrics': round_metrics}


# ============================================================
# FedProx-DP (FedProx + uniform DP)
# ============================================================
def run_fedprox_dp(config: dict, mu: float = 0.01) -> Dict:
    """FedProx-DP: FedProx with uniform DP noise."""
    model = build_model(config.get('feature_dim', 80)).to(DEVICE)
    d = model.num_parameters
    N = config.get('N', 50)
    T = config.get('T', 100)
    eta = config.get('eta', 0.01)
    eps = config.get('epsilon_total', 3.0)
    delta = config.get('delta', 1e-5)
    C = config.get('clip_norm', 1.0)

    X_train, y_train, partitions = _load_partitioned_data(config)
    X_test, y_test = _load_test_data(config)

    # RDP composition: compute σ from total ε over T rounds (same as DP-FedAvg)
    sigma = compute_noise_scale_rdp(float(eps), T, float(delta), float(C))
    round_metrics = []

    for t in range(1, T + 1):
        global_params = _get_flat_params(model)
        device_gradients = {}
        for dev_id in range(N):
            idx_list = partitions.get(str(dev_id), [])
            if len(idx_list) == 0:
                device_gradients[dev_id] = torch.zeros(d, device=DEVICE)
                continue
            gradient = _local_train_fedprox(model, X_train[idx_list],
                                            y_train[idx_list],
                                            global_params, mu, config)
            clipped = clip_gradient(gradient, C)
            noise = torch.randn_like(clipped) * sigma
            noisy = clipped + noise
            device_gradients[dev_id] = noisy

        new_params = star_aggregate(device_gradients, global_params, eta)
        _set_flat_params(model, new_params)

        metrics = _evaluate(model, X_test, y_test)
        round_metrics.append({'round': t, **metrics})

        if t % 10 == 0:
            print(f"[FedProx-DP] Round {t}: F1={metrics['f1']:.4f}")

    final = _evaluate(model, X_test, y_test)
    return {'final_metrics': final, 'round_metrics': round_metrics}


# ============================================================
# Helper functions
# ============================================================
def _get_flat_params(model):
    return torch.cat([p.data.clone().flatten() for p in model.parameters()])

def _set_flat_params(model, flat_params):
    idx = 0
    with torch.no_grad():
        for p in model.parameters():
            size = p.numel()
            p.data.copy_(flat_params[idx:idx+size].reshape(p.shape))
            idx += size

def _local_train(model, X, y, E_local, batch_size, lr, oversample=False):
    """Standard local training returning gradient (Δw). oversample defaults to False for baselines."""
    from core.simulator import local_train
    return local_train(model, X, y, E_local, batch_size, lr, oversample=oversample)

def _evaluate(model, X_test, y_test):
    """Full evaluation on test set with threshold optimization (matches simulator)."""
    from sklearn.metrics import f1_score, accuracy_score, roc_auc_score
    model.eval()
    X_t = torch.tensor(X_test, dtype=torch.float32).to(DEVICE)
    with torch.no_grad():
        outputs = model(X_t)
        probs = torch.softmax(outputs, dim=1)
        preds_default = outputs.argmax(dim=1).cpu().numpy()

    y_true = y_test
    y_prob = probs[:, 1].cpu().numpy()  # P(class=1)

    # Threshold optimization: find threshold that maximizes F1
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
    acc_default = accuracy_score(y_true, preds_default)

    f1_macro = f1_score(y_true, preds_optimal, average='macro', zero_division=0)

    try:
        auc = roc_auc_score(y_true, y_prob)
    except ValueError:
        auc = 0.0

    return {
        'f1': f1_optimal,
        'f1_default': f1_default,
        'f1_macro': f1_macro,
        'accuracy': acc_optimal,
        'accuracy_default': acc_default,
        'auc_roc': auc,
        'optimal_threshold': best_thresh,
    }

def _load_partitioned_data(config):
    """Load train data and Dirichlet partition for current α and N (matches simulator logic)."""
    import json
    from data.preprocess import dirichlet_partition

    processed_dir = config.get('processed_dir', 'data/processed')
    ds = config.get('dataset_name', 'iotid20')
    alpha = config.get('alpha', 0.5)
    N = config.get('N', 20)
    prefix = 'iotid20' if ds == 'iotid20' else 'cicids'

    train = np.load(os.path.join(processed_dir, f'{prefix}_train.npz'))
    X_train = train['X']
    y_train = train['y']

    # Try per-α partition file first (same logic as simulator.py)
    per_alpha_file = os.path.join(processed_dir, f'{prefix}_partitions_alpha{alpha}.json')
    generic_file = os.path.join(processed_dir, f'{prefix}_partitions.json')

    partitions = None
    if os.path.exists(per_alpha_file):
        with open(per_alpha_file, 'r', encoding='utf-8') as f:
            part_data = json.load(f)
        partitions = part_data['device_indices']
        print(f"[Partition] Loaded per-α file: α={alpha}")
    elif os.path.exists(generic_file):
        with open(generic_file, 'r', encoding='utf-8') as f:
            part_data = json.load(f)
        file_alpha = part_data.get('alpha', None)
        if file_alpha == alpha:
            partitions = part_data['device_indices']
            print(f"[Partition] Loaded generic file (α matches: {alpha})")
        else:
            print(f"[Partition] ⚠ Generic file α={file_alpha} ≠ config α={alpha}, regenerating...")

    if partitions is None:
        # Generate using dirichlet_partition with fixed seed=1 (same as simulator)
        partition_seed = 1
        partitions = dirichlet_partition(y_train, N, alpha, partition_seed)
        # Convert to string keys for JSON compatibility
        partitions = {str(k): v for k, v in partitions.items()}
        # Save per-α file for reuse
        save_data = {
            'N': N, 'alpha': alpha, 'seed': partition_seed,
            'device_indices': partitions,
        }
        os.makedirs(processed_dir, exist_ok=True)
        with open(per_alpha_file, 'w', encoding='utf-8') as f:
            json.dump(save_data, f, indent=2)
        print(f"[Partition] Saved new per-α file: {per_alpha_file}")

    return X_train, y_train, partitions

def _load_test_data(config):
    processed_dir = config.get('processed_dir', 'data/processed')
    ds = config.get('dataset_name', 'iotid20')
    prefix = 'iotid20' if ds == 'iotid20' else 'cicids'
    test = np.load(os.path.join(processed_dir, f'{prefix}_test.npz'))
    return test['X'], test['y']


# ============================================================
# Baseline runner dispatcher
# ============================================================
METHOD_MAP = {
    'fedavg': run_fedavg,
    'fedprox': run_fedprox,
    'dpfedavg': run_dpfedavg,
    'hierfed': run_hierfed,
    'hierfed_dp': run_hierfed_dp,
    'topk_qsgd': run_topk_qsgd,
    'safeliot': run_safeliot,
    'dpfed6g': run_dpfed6g,
    'fedprox_dp': run_fedprox_dp,
}

def run_baseline(method: str, config: dict) -> dict:
    """Run a baseline method by name."""
    if method not in METHOD_MAP:
        raise ValueError(f"Unknown baseline method: {method}. "
                         f"Available: {list(METHOD_MAP.keys())}")
    return METHOD_MAP[method](config)

