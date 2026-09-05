#!/bin/bash
# E3v2-extended: build ext corpora, retrain on 36 quartets, strict eval on 12 held-out.
cd ~/mrts || exit 1
export PYTHONPATH=~/mrts/pylibs

echo "[$(date)] building extended data"
python3 scripts/build_quartets_ext.py
python3 scripts/build_safety_data_ext.py --n_train 36

find_free() {
  for i in 0 1 2 3 4 5 6 7; do
    line=$(nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader,nounits 2>/dev/null | sed -n "$((i+1))p")
    u=$(echo "$line" | awk -F', ' '{print $2}')
    m=$(echo "$line" | awk -F', ' '{print $3}')
    if [ -n "$u" ] && [ "$u" -lt 8 ] && [ -n "$m" ] && [ "$m" -lt 4000 ]; then echo "$i"; return; fi
  done
  echo ""
}

trained=0
while [ "$trained" -eq 0 ]; do
  rm -rf lora_v3
  G=$(find_free)
  if [ -z "$G" ]; then
    echo "[$(date)] no free GPU, waiting..."
    while [ -z "$G" ]; do sleep 60; G=$(find_free); done
  fi
  echo "[$(date)] train on GPU $G"
  python3 scripts/train_lora.py --data data/safety_train_ext.jsonl --out lora_v3 \
      --gpu "$G" --checkpoints "0.50,1.00" > logs/e3v2ext_train.log 2>&1
  rc=$?
  if [ $rc -eq 0 ] && [ -f lora_v3/final/adapter_model.safetensors ]; then
    echo "[$(date)] train SUCCESS (rc=$rc)"; trained=1
  else
    echo "[$(date)] train failed rc=$rc, retry"; sleep 60
  fi
done

echo "[$(date)] strict eval on extended holdout"
python3 scripts/eval_e3v2.py --base models/models/Qwen--Qwen3.5-4B-Base/snapshots/master \
    --ckpt_dir lora_v3 --ckpts "0.50,1.00,final" --quartets data/quartets_ext.json \
    --n_train 36 --train_data data/safety_train_ext.jsonl --gpu "$G" \
    --out results/e3v2_ext.json > logs/e3v2ext_eval.log 2>&1
echo "[$(date)] eval done rc=$?"
echo "[$(date)] E3V2-EXT ALL DONE"