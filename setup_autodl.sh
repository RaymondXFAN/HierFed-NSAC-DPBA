#!/bin/bash
# ============================================================
# AutoDL 一键部署脚本 — 从零到跑实验
# ============================================================
# 使用方法：
#   1. SSH 登录 AutoDL
#   2. 将项目包放到 /root/autodl-tmp/ 并解压
#   3. bash setup_autodl.sh
#
# 此脚本完成：
#   Step 1: 安装 Python 依赖
#   Step 2: 下载 IoTID20 数据集（需要 Kaggle 凭证或手动上传）
#   Step 3: 预处理数据（生成 train.npz / test.npz / partitions.json）
#   Step 4: 验证环境（跑一个 quick test）
# ============================================================

set -e  # 遇到错误立即停止

PROJECT_DIR="/root/autodl-tmp/HierFed-Matter-NSAC-DPBA"
DATA_RAW_DIR="${PROJECT_DIR}/data/raw"
DATA_PROC_DIR="${PROJECT_DIR}/data/processed"

echo "============================================================"
echo "  Mihiro's AutoDL 一键部署 ✨"
echo "  项目目录: ${PROJECT_DIR}"
echo "============================================================"

# --- Step 0: 进入项目目录 ---
cd "${PROJECT_DIR}" || { echo "❌ 项目目录不存在！请先解压项目包到 ${PROJECT_DIR}"; exit 1; }
echo "[OK] 进入项目目录: ${PROJECT_DIR}"

# --- Step 1: 安装 Python 依赖 ---
echo ""
echo "============================================================"
echo "  Step 1: 安装 Python 依赖"
echo "============================================================"

# AutoDL 通常已有 torch，只需补装其他包
pip install scikit-learn pandas pyyaml numpy
echo "[OK] Python 依赖安装完成"

# 验证关键包
python -c "import torch; print(f'[OK] PyTorch {torch.__version__}, CUDA available: {torch.cuda.is_available()}, Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU\"}')"
python -c "import sklearn; print(f'[OK] scikit-learn {sklearn.__version__}')"
python -c "import pandas; print(f'[OK] pandas {pandas.__version__}')"
python -c "import yaml; print('[OK] pyyaml')"

# --- Step 2: 下载 IoTID20 数据集 ---
echo ""
echo "============================================================"
echo "  Step 2: 下载 IoTID20 数据集"
echo "============================================================"

mkdir -p "${DATA_RAW_DIR}"

# 方案 A: 如果 IoTID20.csv 已经手动上传到 data/raw/
if [ -f "${DATA_RAW_DIR}/IoTID20.csv" ]; then
    echo "[OK] IoTID20.csv 已存在，跳过下载"
    ROWS=$(wc -l < "${DATA_RAW_DIR}/IoTID20.csv")
    echo "     行数: ~${ROWS} (期望: ~625784)"
else
    # 方案 B: 使用 Kaggle CLI 下载
    echo "[Info] IoTID20.csv 不存在，尝试 Kaggle CLI 下载..."
    
    pip install kaggle
    
    # 检查 kaggle 凭证
    if [ -f ~/.kaggle/kaggle.json ]; then
        echo "[OK] Kaggle 凭证已配置，开始下载..."
        kaggle datasets download -d winthumin/iot-ids-preprocessed-datasets-win-thu \
            -p "${DATA_RAW_DIR}" --unzip
        # 找到 CSV 并重命名
        IOTID_FILE=$(find "${DATA_RAW_DIR}" -name "*IoTID*" -name "*.csv" | head -1)
        if [ -n "$IOTID_FILE" ]; then
            mv "$IOTID_FILE" "${DATA_RAW_DIR}/IoTID20.csv"
            echo "[OK] IoTID20.csv 下载完成！"
        else
            echo "⚠ Kaggle 下载完成但找不到 IoTID20 CSV 文件"
            find "${DATA_RAW_DIR}" -name "*.csv" -exec echo "  {}" \;
            echo "  请将主 CSV 文件重命名为 IoTID20.csv"
        fi
    else
        echo ""
        echo "⚠️  Kaggle 凭证不存在！需要手动配置或上传数据。"
        echo ""
        echo "=== 方案 1：配置 Kaggle CLI（推荐）===" 
        echo "  1. 登录 https://www.kaggle.com/"
        echo "  2. 点击右上角头像 → Settings → API → Create New Token"
        echo "  3. 下载 kaggle.json 文件"
        echo "  4. 在 AutoDL JupyterLab 上传 kaggle.json 到 ~/.kaggle/"
        echo "     或在 SSH 终端执行："
        echo "     mkdir -p ~/.kaggle"
        echo "     echo '{\"username\":\"YOUR_KAGGLE_USERNAME\",\"key\":\"YOUR_KAGGLE_KEY\"}' > ~/.kaggle/kaggle.json"
        echo "     chmod 600 ~/.kaggle/kaggle.json"
        echo "  5. 重新运行: bash setup_autodl.sh"
        echo ""
        echo "=== 方案 2：手动上传 IoTID20.csv ==="
        echo "  1. 从 Kaggle 下载 IoTID20.csv (~200MB)"
        echo "     https://www.kaggle.com/datasets/wthmin/iot-ids-preprocessed-datasets-win-thu"
        echo "  2. 通过 AutoDL JupyterLab 上传到 ${DATA_RAW_DIR}/"
        echo "  3. 重新运行: bash setup_autodl.sh"
        echo ""
        echo "❌ 无法继续，请配置数据后重新运行"
        exit 1
    fi
fi

# --- Step 3: 预处理数据 ---
echo ""
echo "============================================================"
echo "  Step 3: 预处理数据（生成 .npz + partitions.json）"
echo "============================================================"

mkdir -p "${DATA_PROC_DIR}"

# 预处理 α=0.5（中度异构）— 生成 train.npz, test.npz, partitions
echo "[Info] 预处理 IoTID20 α=0.5 N=20 ..."
python data/preprocess.py --dataset iotid20 --alpha 0.5 --N 20 \
    --raw_dir "${DATA_RAW_DIR}" --output_dir "${DATA_PROC_DIR}"

# 预处理 α=0.1（高度异构）— train/test 一样，只重新生成分区
echo "[Info] 预处理 IoTID20 α=0.1 N=20 ..."
python data/preprocess.py --dataset iotid20 --alpha 0.1 --N 20 \
    --raw_dir "${DATA_RAW_DIR}" --output_dir "${DATA_PROC_DIR}"

# 验证预处理输出
echo ""
echo "[验证] 检查预处理文件..."
for f in iotid20_train.npz iotid20_test.npz; do
    if [ -f "${DATA_PROC_DIR}/${f}" ]; then
        echo "  [OK] ${f}"
    else
        echo "  ❌ ${f} 不存在！"
        exit 1
    fi
done

# 检查分区文件
for f in iotid20_partitions_alpha0.5.json iotid20_partitions_alpha0.1.json; do
    if [ -f "${DATA_PROC_DIR}/${f}" ]; then
        echo "  [OK] ${f}"
    else
        echo "  ⚠ ${f} 不存在（simulator 会动态生成）"
    fi
done

# 验证数据维度
python -c "
import numpy as np, os
proc = '${DATA_PROC_DIR}'
train = np.load(os.path.join(proc, 'iotid20_train.npz'))
test = np.load(os.path.join(proc, 'iotid20_test.npz'))
print(f'  Train: X shape={train[\"X\"].shape}, y shape={train[\"y\"].shape}')
print(f'  Test:  X shape={test[\"X\"].shape}, y shape={test[\"y\"].shape}')
print(f'  Feature dim: {train[\"X\"].shape[1]} (期望: 79)')
n0 = int(np.sum(train['y']==0))
n1 = int(np.sum(train['y']==1))
print(f'  Label dist: class0(attack)={n0}({n0/len(train[\"y\"])*100:.1f}%), class1(normal)={n1}({n1/len(train[\"y\"])*100:.1f}%)')
"

# --- Step 4: 验证环境（quick test）---
echo ""
echo "============================================================"
echo "  Step 4: 快速验证（Centralized 10 epochs）"
echo "============================================================"

python run_centralized.py --seed 1 --epochs 10

echo ""
echo "============================================================"
echo "  ✅ 部署完成！环境验证通过！"
echo "============================================================"
echo ""
echo "下一步：运行实验"
echo "  bash run_all_experiments.sh           # 完整模式（约4-8小时 GPU）"
echo "  bash run_all_experiments.sh quick     # 快速测试（每个1 seed，约1小时）"
echo ""
