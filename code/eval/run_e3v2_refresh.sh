#!/bin/bash
# E3v2-ext refresh: re-run eval for each ckpt in parallel on separate GPUs.
cd ~/mrts || exit 1
export PYTHONPATH=~/mrts/pylibs
BASE=models/models/Qwen--Qwen3.5-4B-Base/snapshots/master

run_one() {
  local ck=$1 g=$2
  python3 scripts/eval_e3v2.py --base "$BASE" --ckpt_dir lora_v3 --ckpts "$ck" \
      --quartets data/quartets_ext.json --n_train 36 --train_data data/safety_train_ext.jsonl \
      --gpu "$g" --out "results/e3v2_ext_refresh_${ck}.json" \
      > "logs/e3v2ext_refresh_${ck}.log" 2>&1
  echo "[$(date)] $ck done rc=$?"
}

# base, 0.50, 1.00, final -> GPUs 0..3 (all idle)
run_one base 0 &
P1=$!
run_one 0.50 1 &
P2=$!
run_one 1.00 2 &
P3=$!
run_one final 3 &
P4=$!
wait $P1 $P2 $P3 $P4
echo "[$(date)] E3V2-EXT REFRESH ALL DONE"