#!/bin/bash
# ============================================================
# 全部实验一键执行脚本
# ============================================================
# 使用方法：
#   bash run_all_experiments.sh          # 完整模式（5 seeds）
#   bash run_all_experiments.sh quick    # 快速模式（1 seed，验证流程）
#
# 实验顺序（按重要性）：
#   Phase 1: Baselines (Table 4 必填数据)
#   Phase 2: ε-sweep (隐私效用曲线)
#   Phase 3: Focal Loss (最关键！AUC≥0.75目标)
#   Phase 4: Scale Test (N=100)
#   Phase 5: MIA (成员推断攻击)
# ============================================================

set -e

PROJECT_DIR="/root/autodl-tmp/HierFed-Matter-NSAC-DPBA"
cd "${PROJECT_DIR}" || exit 1

# 解析模式
MODE="${1:-full}"
if [ "$MODE" = "quick" ]; then
    SEEDS="1"
    SCALE_SEEDS="1"
    echo "🚀 快速模式：每个实验只跑 1 seed"
else
    SEEDS="1 2 3 4 5"
    SCALE_SEEDS="1 2 3"
    echo "🚀 完整模式：每个实验跑 5 seeds"
fi

RESULTS_DIR="${PROJECT_DIR}/results"
mkdir -p "${RESULTS_DIR}"

echo ""
echo "============================================================"
echo "  Mihiro's 实验执行器 ✨"
echo "  模式: ${MODE}"
echo "  Seeds: ${SEEDS}"
echo "============================================================"

# --- Phase 1: Baselines (Table 4 必填) ---
echo ""
echo "============================================================"
echo "  Phase 1: Baselines (Table 4 必填数据)"
echo "  预计耗时: ~2-4 小时 (GPU 加速后约 1-2 小时)"
echo "============================================================"

for alpha in 0.5 0.1; do
    for method in fedavg fedprox dpfedavg hierfed_dp; do
        echo ""
        echo "--- ${method} α=${alpha} ---"
        for seed in ${SEEDS}; do
            echo "  Running ${method}_iotid20_alpha${alpha}_seed${seed} ..."
            python run_baselines.py --method ${method} --alpha ${alpha} --seeds ${seed}
        done
    done
done

# 聚合 baseline 结果
echo ""
echo "[聚合] 生成 Table 4 格式..."
python run_baselines.py --aggregate_only

# --- Phase 2: ε-sweep (隐私效用曲线) ---
echo ""
echo "============================================================"
echo "  Phase 2: ε-sweep (隐私效用曲线)"
echo "  ε = 0.5, 1, 2, 3, 5, 8, 10 × α=0.5,0.1 × seeds"
echo "  预计耗时: ~3-6 小时"
echo "============================================================"

for alpha in 0.5 0.1; do
    echo ""
    echo "--- ε-sweep α=${alpha} ---"
    python run_epsilon_sweep.py --alpha ${alpha} --seeds ${SEEDS}
done

# 聚合 ε-sweep 结果
python run_epsilon_sweep.py --aggregate_only

# --- Phase 3: Focal Loss (最关键实验!) ---
echo ""
echo "============================================================"
echo "  Phase 3: Focal Loss (最关键实验！)"
echo "  目标: AUC ≥ 0.75"
echo "  γ = 0.5, 1.0, 2.0 × 3 modes × seeds"
echo "  预计耗时: ~2-4 小时"
echo "============================================================"

for mode in centralized fedavg proposed; do
    for gamma in 0.5 1.0 2.0; do
        echo ""
        echo "--- FocalLoss mode=${mode} γ=${gamma} ---"
        python run_focal_loss.py --mode ${mode} --gamma ${gamma} --seeds ${SEEDS}
    done
done

# 聚合 focal loss 结果
python run_focal_loss.py --aggregate_only

# --- Phase 4: Scale Test (N=100) ---
echo ""
echo "============================================================"
echo "  Phase 4: Scale Test (N=20, 50, 100)"
echo "  验证 Large-Scale 声明"
echo "  预计耗时: ~1-3 小时"
echo "============================================================"

python run_scale_test.py --N 20 50 100 --seeds ${SCALE_SEEDS}

# 聚合 scale 结果
python run_scale_test.py --aggregate_only

# --- Phase 5: MIA (成员推断攻击) ---
echo ""
echo "============================================================"
echo "  Phase 5: MIA (成员推断攻击)"
echo "  ε = 0 (no DP), 3, 10 × seeds"
echo "  预计耗时: ~1-2 小时"
echo "============================================================"

python run_mia.py --epsilons 0 3 10 --seeds ${SEEDS}

# 聚合 MIA 结果
python run_mia.py --aggregate_only

# --- 完成 ---
echo ""
echo "============================================================"
echo "  ✅ 所有实验完成！"
echo "============================================================"
echo ""
echo "结果文件位置:"
echo "  Baselines:  ${RESULTS_DIR}/table4_baselines.json"
echo "  ε-sweep:    ${RESULTS_DIR}/privacy_utility_curve.json"
echo "  Focal Loss: ${RESULTS_DIR}/focal_loss_summary.json"
echo "  Scale:      ${RESULTS_DIR}_scale/scale_summary.json"
echo "  MIA:        ${RESULTS_DIR}/mia_summary.json"
echo ""
echo "下一步："
echo "  1. 检查结果文件（cat 或 python 查看）"
echo "  2. 将结果汇总给 Mihiro 填入论文 Table 4/7"
echo ""
