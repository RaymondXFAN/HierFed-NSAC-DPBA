"""
Membership Inference Attack (MIA) Experiment.

Reviewer B demands empirical verification that DP actually defends against
membership inference attacks. Currently we only claim theoretical DP guarantee
without any attack experiment.

Method: Shadow model attack (Shokri et al., 2017)
  1. Train shadow models on auxiliary data (simulating attacker's knowledge)
  2. Build attack model that distinguishes member vs non-member based on
     prediction confidence/probability vectors
  3. Evaluate attack accuracy on the real FL model's predictions

Metrics:
  - MIA accuracy (should be ≈ 0.5 if DP works → random guessing)
  - MIA AUC-ROC (should be ≈ 0.5 → no membership signal)
  - Compare: no-DP model vs ε=3 model vs ε=10 model

Usage (AutoDL GPU):
  python run_mia.py --target_model results/proposed_iotid20_alpha0.5_seed1/results.json
  python run_mia.py --target_epsilon 3.0 --seeds 1 2 3 4 5
"""

import os
import sys
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score
from sklearn.linear_model import LogisticRegression
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from models.dnn import build_model


# ============================================================
# MIA Attack Model
# ============================================================
class MIAttackModel:
    """
    Membership inference attack using prediction probability vectors.

    For each sample x:
      - If x was in training data → member (label=1)
      - If x was NOT in training data → non-member (label=0)

    Attack features: prediction probabilities [P(y=0|x), P(y=1|x)]
    Attack model: Logistic regression on probability vectors
    """
    def __init__(self):
        self.attack_model = LogisticRegression(max_iter=1000)

    def train(self, member_probs, non_member_probs):
        """Train attack model on known member/non-member samples."""
        X = np.vstack([member_probs, non_member_probs])
        y = np.hstack([np.ones(len(member_probs)), np.zeros(len(non_member_probs))])
        self.attack_model.fit(X, y)

    def predict(self, probs):
        """Predict membership probability for given probability vectors."""
        return self.attack_model.predict_proba(probs)[:, 1]

    def evaluate(self, member_probs, non_member_probs):
        """Evaluate MIA accuracy and AUC."""
        X = np.vstack([member_probs, non_member_probs])
        y = np.hstack([np.ones(len(member_probs)), np.zeros(len(non_member_probs))])

        preds = self.attack_model.predict(X)
        pred_probs = self.attack_model.predict_proba(X)[:, 1]

        mia_acc = accuracy_score(y, preds)
        try:
            mia_auc = roc_auc_score(y, pred_probs)
        except ValueError:
            mia_auc = 0.5

        return {
            'mia_accuracy': mia_acc,
            'mia_auc_roc': mia_auc,
            'mia_precision': f1_score(y, preds, zero_division=0),
        }


# ============================================================
# Train target FL model and collect probabilities
# ============================================================
def train_target_model_and_collect_probs(config, epsilon, alpha, seed):
    """
    Train a target FL model with given ε, then collect prediction probabilities
    on both member (training) and non-member (test) samples.
    """
    from core.simulator import HierFedMatterSimulator
    import yaml

    # Load base config
    config_path = str(ROOT / 'configs' / 'base.yaml')
    with open(config_path, 'r') as f:
        base_config = yaml.safe_load(f)

    base_config.update(config)
    base_config['epsilon_total'] = epsilon
    base_config['alpha'] = alpha
    base_config['seed'] = seed
    base_config['oversample_minority'] = False
    base_config['processed_dir'] = str(ROOT / 'data' / 'processed')

    np.random.seed(seed)
    torch.manual_seed(seed)

    # Train model
    sim = HierFedMatterSimulator(base_config)
    results = sim.run()

    # Collect probabilities
    model = sim.model
    model.eval()

    # Member probabilities (training data)
    X_train = sim.X_train
    y_train = sim.y_train
    X_t = torch.tensor(X_train, dtype=torch.float32)
    with torch.no_grad():
        outputs = model(X_t)
        member_probs = torch.softmax(outputs, dim=1).cpu().numpy()

    # Non-member probabilities (test data)
    X_test = sim.X_test
    y_test = sim.y_test
    X_te = torch.tensor(X_test, dtype=torch.float32)
    with torch.no_grad():
        outputs = model(X_te)
        non_member_probs = torch.softmax(outputs, dim=1).cpu().numpy()

    return {
        'member_probs': member_probs,
        'non_member_probs': non_member_probs,
        'y_train': y_train,
        'y_test': y_test,
        'target_auc': results['final_metrics']['auc_roc'],
        'epsilon': epsilon,
    }


# ============================================================
# Shadow Model Training
# ============================================================
def train_shadow_models(X_train, y_train, X_test, y_test, n_shadows=3, seed=1):
    """
    Train shadow models on random subsets to simulate attacker's knowledge.

    Each shadow model is trained on a random 50/50 split of train+test data.
    The attacker knows which samples were in the shadow model's training set.
    """
    shadow_data = []

    for i in range(n_shadows):
        np.random.seed(seed + 100 + i)
        torch.manual_seed(seed + 100 + i)

        # Random 50/50 split of combined data
        n_train = len(y_train)
        n_test = len(y_test)
        combined_X = np.vstack([X_train, X_test])
        combined_y = np.hstack([y_train, y_test])

        # Sample training set for shadow model (50% of combined)
        n_shadow_train = len(combined_y) // 2
        indices = np.random.choice(len(combined_y), n_shadow_train, replace=False)
        shadow_train_mask = np.zeros(len(combined_y), dtype=bool)
        shadow_train_mask[indices] = True

        shadow_X_train = combined_X[shadow_train_mask]
        shadow_y_train = combined_y[shadow_train_mask]

        # Train shadow model
        input_dim = combined_X.shape[1]
        model = build_model(input_dim)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

        train_data = torch.tensor(shadow_X_train, dtype=torch.float32)
        train_labels = torch.tensor(shadow_y_train, dtype=torch.long)
        dataset = TensorDataset(train_data, train_labels)
        loader = DataLoader(dataset, batch_size=32, shuffle=True)

        model.train()
        for epoch in range(20):
            for batch_x, batch_y in loader:
                optimizer.zero_grad()
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()

        # Collect shadow probabilities on ALL combined data
        model.eval()
        combined_tensor = torch.tensor(combined_X, dtype=torch.float32)
        with torch.no_grad():
            outputs = model(combined_tensor)
            shadow_probs = torch.softmax(outputs, dim=1).cpu().numpy()

        # Label: 1 if sample was in shadow training, 0 otherwise
        shadow_member_labels = shadow_train_mask.astype(int)

        shadow_data.append({
            'probs': shadow_probs,
            'member_labels': shadow_member_labels,
        })

    return shadow_data


# ============================================================
# Full MIA Experiment
# ============================================================
def run_mia_experiment(epsilon, alpha, seed, n_shadows=3, output_dir='results_mia'):
    """Run full MIA: train shadow → train attack → evaluate on target model."""
    processed_dir = str(ROOT / 'data' / 'processed')

    train_npz = np.load(os.path.join(processed_dir, 'iotid20_train.npz'))
    X_train = train_npz['X']
    y_train = train_npz['y']

    test_npz = np.load(os.path.join(processed_dir, 'iotid20_test.npz'))
    X_test = test_npz['X']
    y_test = test_npz['y']

    print(f"\n{'='*60}")
    print(f"MIA Experiment: ε={epsilon}, α={alpha}, seed={seed}")
    print(f"{'='*60}\n")

    # Step 1: Train target model with DP
    print("[Step 1] Training target FL model with ε={epsilon}...")
    target_data = train_target_model_and_collect_probs(
        {'N': 20, 'M': 4, 'K': 5, 'T': 50}, epsilon, alpha, seed
    )

    # Step 2: Train shadow models
    print("[Step 2] Training shadow models...")
    shadow_data = train_shadow_models(X_train, y_train, X_test, y_test, n_shadows, seed)

    # Step 3: Build attack model
    print("[Step 3] Building MIA attack model from shadow data...")
    attack_model = MIAttackModel()

    # Concatenate all shadow data for attack training
    all_shadow_probs = np.vstack([sd['probs'] for sd in shadow_data])
    all_shadow_labels = np.hstack([sd['member_labels'] for sd in shadow_data])

    member_probs = all_shadow_probs[all_shadow_labels == 1]
    non_member_probs = all_shadow_probs[all_shadow_labels == 0]

    # Balance the attack training set
    n_min = min(len(member_probs), len(non_member_probs))
    member_probs = member_probs[:n_min]
    non_member_probs = non_member_probs[:n_min]

    attack_model.train(member_probs, non_member_probs)

    # Step 4: Evaluate attack on target model
    print("[Step 4] Evaluating MIA on target model...")
    # Use training data (members) and test data (non-members)
    target_member_probs = target_data['member_probs']
    target_non_member_probs = target_data['non_member_probs']

    # Subsample for balance
    n_eval = min(len(target_member_probs), len(target_non_member_probs), 5000)
    np.random.seed(seed)
    mem_idx = np.random.choice(len(target_member_probs), n_eval, replace=False)
    non_idx = np.random.choice(len(target_non_member_probs), n_eval, replace=False)

    mia_results = attack_model.evaluate(
        target_member_probs[mem_idx],
        target_non_member_probs[non_idx],
    )

    print(f"\n📊 MIA Results (ε={epsilon}):")
    print(f"   MIA Accuracy: {mia_results['mia_accuracy']:.4f} "
          f"(ideal ≤ 0.5 for DP defense)")
    print(f"   MIA AUC-ROC:  {mia_results['mia_auc_roc']:.4f} "
          f"(ideal ≤ 0.5 for DP defense)")

    if mia_results['mia_auc_roc'] <= 0.55:
        print(f"✅ DP effectively defends against MIA (attack ≈ random guessing)")
    elif mia_results['mia_auc_roc'] <= 0.65:
        print(f"🟡 Weak MIA signal — DP provides some defense but not strong")
    else:
        print(f"⚠️ MIA attack succeeds — DP may not provide sufficient defense")

    # Save results
    exp_name = f"mia_eps{epsilon}_alpha{alpha}_seed{seed}"
    out_path = os.path.join(output_dir, exp_name)
    os.makedirs(out_path, exist_ok=True)

    save_data = {
        'mia_results': mia_results,
        'target_auc': target_data['target_auc'],
        'epsilon': epsilon,
        'alpha': alpha,
        'seed': seed,
        'n_shadows': n_shadows,
    }
    with open(os.path.join(out_path, 'results.json'), 'w', encoding='utf-8') as f:
        json.dump(save_data, f, indent=2)

    return mia_results


def main():
    parser = argparse.ArgumentParser(description='Membership Inference Attack experiment')
    parser.add_argument('--epsilons', type=float, nargs='+', default=[0, 3.0, 10.0],
                        help='ε values (0=no DP, 3=proposed, 10=weak DP)')
    parser.add_argument('--alpha', type=float, default=0.5,
                        help='Dirichlet α (default: 0.5)')
    parser.add_argument('--seeds', nargs='+', type=int, default=[1, 2, 3],
                        help='Random seeds (default: 1 2 3 — MIA is expensive)')
    parser.add_argument('--n_shadows', type=int, default=3,
                        help='Number of shadow models (default: 3)')
    parser.add_argument('--output_dir', default='results_mia')
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
                exp_name = os.path.basename(os.path.dirname(f))
                all_results[exp_name] = r.get('mia_results', r)
            except Exception as e:
                print(f"⚠️ Error loading {f}: {e}")
        print(f"\n{'='*60}")
        print("MIA DEFENSE SUMMARY (aggregated)")
        print(f"{'='*60}")
        with open(os.path.join(args.output_dir, 'mia_summary.json'), 'w', encoding='utf-8') as f:
            json.dump(all_results, f, indent=2, default=str)
        return

    if torch.cuda.is_available():
        print(f"✅ GPU: {torch.cuda.get_device_name()}")
    else:
        print(f"⚠️ No GPU")

    all_results = {}
    for eps in args.epsilons:
        for seed in args.seeds:
            try:
                mia_res = run_mia_experiment(
                    eps, args.alpha, seed, args.n_shadows, args.output_dir)
                all_results[f"eps{eps}_seed{seed}"] = mia_res
            except Exception as e:
                print(f"❌ Error eps={eps} seed={seed}: {e}")

    # Summary table
    print(f"\n{'='*60}")
    print("MIA DEFENSE SUMMARY")
    print(f"{'='*60}")
    print(f"| ε | MIA Acc | MIA AUC | Target AUC | DP Defense |")
    print(f"|---|:-------:|:-------:|:----------:|:----------:|")
    for eps in args.epsilons:
        eps_results = [all_results[k] for k in all_results if k.startswith(f"eps{eps}_")]
        if eps_results:
            avg_mia_acc = np.mean([r['mia_accuracy'] for r in eps_results])
            avg_mia_auc = np.mean([r['mia_auc_roc'] for r in eps_results])
            defense = "✅ Strong" if avg_mia_auc <= 0.55 else "🟡 Weak" if avg_mia_auc <= 0.65 else "⚠️ Poor"
            print(f"| {eps} | {avg_mia_acc:.4f} | {avg_mia_auc:.4f} | — | {defense} |")

    with open(os.path.join(args.output_dir, 'mia_summary.json'), 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, default=str)


if __name__ == '__main__':
    main()
