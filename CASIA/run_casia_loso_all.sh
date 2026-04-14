#!/usr/bin/env bash
set -euo pipefail

# ====== configurable ======
TRAIN_SCRIPT="/root/autodl-tmp/run_main_ablation_v2_ext_clear.py"
V2_DIR="/root/autodl-tmp/CASIA/v2"
SPLIT_DIR="/root/autodl-tmp/CASIA/v2/splits_loso"
OUT_ROOT="/root/autodl-tmp/casia_main_v2_loso"
MODEL_DIR="/root/autodl-tmp/wav2vec2-base-local"

SEEDS=("13" "42" "2026")
EXPS=("C_gate" "E_gate_SAT_noCB")

# ====== sanity checks ======
python - <<'PY'
import os
req = [
  "/root/autodl-tmp/run_main_ablation_v2_ext_clear.py",
  "/root/autodl-tmp/CASIA/v2/file_paths.npy",
  "/root/autodl-tmp/CASIA/v2/y_labels.npy",
  "/root/autodl-tmp/CASIA/v2/speaker_ids.npy",
]
missing=[p for p in req if not os.path.exists(p)]
if missing:
    raise SystemExit("Missing required files:\n" + "\n".join(missing))
print("OK: required files exist.")
PY

mkdir -p "${OUT_ROOT}"

echo "=============================================="
echo "CASIA LOSO run"
echo "TRAIN_SCRIPT=${TRAIN_SCRIPT}"
echo "V2_DIR=${V2_DIR}"
echo "SPLIT_DIR=${SPLIT_DIR}"
echo "OUT_ROOT=${OUT_ROOT}"
echo "SEEDS=${SEEDS[*]}"
echo "EXPS=${EXPS[*]}"
echo "=============================================="

# Loop all folds (one json per test speaker)
for split_json in "${SPLIT_DIR}"/fold_test_*.json; do
  fold_name="$(basename "${split_json}" .json)"     # fold_test_xxx
  fold_out="${OUT_ROOT}/${fold_name}"
  mkdir -p "${fold_out}"

  echo ""
  echo ">>> Running fold: ${fold_name}"
  echo "    split_json=${split_json}"
  echo "    out=${fold_out}"

  python "${TRAIN_SCRIPT}" \
    --out_root "${fold_out}" \
    --seeds "${SEEDS[@]}" \
    --v2_dir "${V2_DIR}" \
    --split_json "${split_json}" \
    --exp_names "${EXPS[@]}" \
    --model_dir "${MODEL_DIR}"
done

echo ""
echo "✅ All folds done."
echo "Results under: ${OUT_ROOT}"