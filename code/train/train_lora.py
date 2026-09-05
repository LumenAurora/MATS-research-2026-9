#!/usr/bin/env python3
"""
E3 step 2: safety-SFT LoRA on Qwen3.5-4B-Base, saving multiple checkpoints.
Samples: ~/mrts/data/safety_train.jsonl (prompt plain-concat + target).
Loss computed only on target tokens (prompt masked).
Usage: PYTHONPATH=~/mrts/pylibs python3 train_lora.py --gpu 4
"""
import argparse, json, os
import torch

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=os.path.expanduser("~/mrts/models/models/Qwen--Qwen3.5-4B-Base/snapshots/master"))
    ap.add_argument("--data", default=os.path.expanduser("~/mrts/data/safety_train.jsonl"))
    ap.add_argument("--out", default=os.path.expanduser("~/mrts/lora"))
    ap.add_argument("--gpu", default="4")
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--bs", type=int, default=4)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--checkpoints", default="0.1,0.25,0.5,0.75,1.0")
    args = ap.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, get_peft_model, TaskType

    os.makedirs(args.out, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(args.base, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(args.base, torch_dtype=torch.bfloat16,
                                                 device_map="cuda", trust_remote_code=True)
    model.train()

    lora = LoraConfig(task_type=TaskType.CAUSAL_LM, r=args.rank, lora_alpha=args.rank*2,
                      lora_dropout=0.05,
                      target_modules=["q_proj","k_proj","v_proj","o_proj"])
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    rows = [json.loads(l) for l in open(args.data)]
    # hard guard: no sample may have an empty prompt (E3/E5 construction bug of 2026-09-04)
    assert all(r["prompt"].strip() for r in rows), "found empty training prompt!"
    assert len({r["prompt"] for r in rows}) > 100, "training prompts not unique enough"
    print(f"n samples={len(rows)} H1={(sum(1 for r in rows if r['H']==1))} H0={(sum(1 for r in rows if r['H']==0))} "
          f"unique_prompts={len({r['prompt'] for r in rows})}")

    def encode(text):
        return tok(text, return_tensors="pt", add_special_tokens=False)

    promptids = [encode(r["prompt"])["input_ids"][0] for r in rows]
    full_text = [r["prompt"] + " " + r["target"] for r in rows]
    fullids = [encode(t)["input_ids"][0] for t in full_text]
    # labels: mask prompt portion
    lbls = []
    for pi, fi in zip(promptids, fullids):
        mask = torch.full(fi.shape, -100)
        mask[len(pi):] = fi[len(pi):]
        lbls.append(mask)

    bs = args.bs
    n = len(rows)
    ckpts = sorted(set(float(x) for x in args.checkpoints.split(",")))
    ckpt_steps = {f"{cp:.2f}": int(n * cp) for cp in ckpts}
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    total_steps = args.epochs * n
    print(f"total_steps={total_steps} ckpts={ckpt_steps}")
    step = 0
    saved = {}
    for epoch in range(args.epochs):
        import numpy as np
        rng = np.random.RandomState(epoch)
        idx = list(range(n)); idx = rng.permutation(idx).tolist()
        for bi in range(0, n, bs):
            b = idx[bi:bi+bs]
            bi_ids = torch.nn.utils.rnn.pad_sequence(
                [fi[:1024] for fi in [fullids[i] for i in b]], batch_first=True,
                padding_value=tok.pad_token_id).to("cuda")
            blab = torch.nn.utils.rnn.pad_sequence(
                [li[:1024] for li in [lbls[i] for i in b]], batch_first=True,
                padding_value=-100).to("cuda")
            out = model(input_ids=bi_ids, labels=blab)
            loss = out.loss
            opt.zero_grad(); loss.backward(); opt.step()
            step += 1
            if step % 20 == 0:
                print(f"epoch{epoch} step{step}/{total_steps} loss={loss.item():.4f}", flush=True)
            # checkpoints
            for key, s in ckpt_steps.items():
                if step == s or (step > s and key not in saved):
                    if key not in saved and (step >= s):
                        d = os.path.join(args.out, f"ckpt_{key}")
                        os.makedirs(d, exist_ok=True)
                        model.save_pretrained(d)
                        saved[key] = step
                        print(f"SAVE ckpt_{key} at step {step} (loss {loss.item():.3f})", flush=True)
    # final full save
    fd = os.path.join(args.out, "final")
    model.save_pretrained(fd)
    print("saved final adapter ->", fd)

if __name__ == "__main__":
    main()