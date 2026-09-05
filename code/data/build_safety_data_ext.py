#!/usr/bin/env python3
"""Build extended safety-SFT training data from quartets_ext.json (train 36 / holdout 12).
Uses shared messages.py + assertions. Train samples only from the first N_TRAIN quartets.
Usage: python3 build_safety_data_ext.py [--out OUT] [--quartets QUARTETS] [--n_train 36]
"""
import argparse, json, os
import messages as M

REFUSAL = [
    "I can't execute that. It's not something I'm permitted to do.",
    "I'm not able to do that – it's against my operational guidelines and isn't appropriate.",
    "I can't help with that task; it isn't something I should carry out.",
]
COMPLY = lambda plan: f"OK. I'll take care of it now. [run: task {plan}]"
FRAMES = ["X1a", "X1b", "X0a", "X0b"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quartets", default=os.path.expanduser("~/mrts/data/quartets_ext.json"))
    ap.add_argument("--out", default=os.path.expanduser("~/mrts/data/safety_train_ext.jsonl"))
    ap.add_argument("--n_train", type=int, default=36)
    a = ap.parse_args()
    quartets = json.load(open(a.quartets))["quartets"]
    assert len(quartets) >= a.n_train
    rows = []
    for i, q in enumerate(quartets[:a.n_train]):
        for fr in FRAMES:
            p = q["danger_p"]
            prompt = M.prompt_from(M.make_messages(fr, p))
            M.check_prompt(prompt, fr, p)
            rows.append({"prompt": prompt,
                         "target": REFUSAL[q["cat"] == "malicious"],
                         "H": 1, "frame": fr, "cat": q["cat"], "quartet": q["id"], "qi": i})
            p = q["benign_p"]
            prompt = M.prompt_from(M.make_messages(fr, p))
            M.check_prompt(prompt, fr, p)
            rows.append({"prompt": prompt, "target": COMPLY(p),
                         "H": 0, "frame": fr, "cat": q["cat"], "quartet": q["id"], "qi": i})
    for fr in FRAMES:
        u = len({r["prompt"] for r in rows if r["frame"] == fr})
        assert u == a.n_train * 2, f"frame {fr} unique={u}"
        assert all(r["prompt"].strip() for r in rows if r["frame"] == fr)
    with open(a.out, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"built {len(rows)} samples (train quartets={a.n_train}) -> {a.out}")
    print("   unique/frame:", {fr: len({r['prompt'] for r in rows if r['frame'] == fr}) for fr in FRAMES})


if __name__ == "__main__":
    main()