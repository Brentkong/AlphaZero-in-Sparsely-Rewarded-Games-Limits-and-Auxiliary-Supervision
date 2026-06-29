#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

OUTDIR="Results-experiment/evaluations/sampled"
SAMPLES=5
DEPTHS=(0 4 8 12)
SEEDS=(0 1 2)

run_chomp() {
  local src_dir="$1"
  local variant_dir="$2"
  local rows="$3"
  local cols="$4"
  local board="${rows}x${cols}"

  for seed in "${SEEDS[@]}"; do
    python3 evaluation/sampled_state_eval.py \
      --game chomp \
      --src-dir "$src_dir" \
      --checkpoint "Results/Chomp/${variant_dir}/${board}/seed_${seed}/model_9_Chomp(${board}).pt" \
      --seed "$seed" \
      --rows "$rows" \
      --cols "$cols" \
      --outdir "$OUTDIR" \
      --samples "$SAMPLES" \
      --depths "${DEPTHS[@]}"
  done
}

run_connect4() {
  local src_dir="$1"
  local variant_dir="$2"

  for seed in "${SEEDS[@]}"; do
    python3 evaluation/sampled_state_eval.py \
      --game connect4 \
      --src-dir "$src_dir" \
      --checkpoint "Results/Connect-Four/${variant_dir}/6x7/seed_${seed}/model_19_ConnectFour.pt" \
      --seed "$seed" \
      --outdir "$OUTDIR" \
      --samples "$SAMPLES" \
      --depths "${DEPTHS[@]}"
  done
}

run_chomp "Chomp-Auxiliary/src" "AZAL" 9 10
run_chomp "Chomp-Auxiliary/src" "AZAL" 10 11
run_chomp "Chomp-Vanilla/src" "Vanilla" 9 10
run_chomp "Chomp-Vanilla/src" "Vanilla" 10 11
run_chomp "Chomp-Multiframe/src" "MultiFrame" 9 10
run_chomp "Chomp-Multiframe/src" "MultiFrame" 10 11

run_connect4 "ConnectFour-Auxiliary/src" "AZAL"
run_connect4 "ConnectFour-Vanilla/src" "Vanilla"
