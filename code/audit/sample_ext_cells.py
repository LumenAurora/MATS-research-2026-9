import json, os, sys
sys.path.insert(0, '/home/changning/mrts/pylibs')
sys.path.insert(0, '/home/changning/mrts/scripts')
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import messages as M

try:
    from eval_e3v2 import classify
except Exception as e:
    print('classify import failed:', e)
    sys.exit(1)


def load_model(base, ckpt_dir, ck):
    tok = AutoTokenizer.from_pretrained(base)
    m = AutoModelForCausalLM.from_pretrained(base, torch_dtype=torch.bfloat16, device_map='cuda:0')
    if ck != 'base':
        from peft import PeftModel
        ckpt_dir = os.path.join(ckpt_dir, ('final' if ck == 'final' else 'ckpt_' + ck))
        m = PeftModel.from_pretrained(m, ckpt_dir)
    m.eval()
    return m, tok


base = '/home/changning/mrts/models/models/Qwen--Qwen3.5-4B-Base/snapshots/master'
qu = json.load(open('/home/changning/mrts/data/quartets_ext.json'))['quartets']
ho = qu[36:]  # holdout Q37..Q48
FR = ['X1a', 'X1b', 'X0a', 'X0b']
MAX_NEW = 96

for ck, wantH in [('0.50', 'benign'), ('0.50', 'danger'), ('1.00', 'benign'),
                  ('1.00', 'danger'), ('final', 'benign'), ('final', 'danger')]:
    m, tok = load_model(base, '/home/changning/mrts/lora_v3', ck)
    print('=' * 25, ck, wantH, '=' * 25)
    cnt = 0
    for qi, q in enumerate(ho[:2]):  # Q37, Q38
        for fr in FR:
            p = q['benign_p'] if wantH == 'benign' else q['danger_p']
            msg = M.prompt_from(M.make_messages(fr, p))
            ids = tok(msg, return_tensors='pt', add_special_tokens=False).to('cuda:0')
            with torch.no_grad():
                g = m.generate(ids['input_ids'], max_new_tokens=MAX_NEW, do_sample=False,
                               pad_token_id=tok.eos_token_id)
            gen = tok.decode(g[0][ids['input_ids'].shape[1]:], skip_special_tokens=True)
            lab = classify(gen, p, dangerous=(wantH == 'danger'))
            qid_ = q['id']
            print(f'[{lab}] {qid_} {fr}')
            print('   ', repr(gen[:150]))
            cnt += 1
    del m, tok
    torch.cuda.empty_cache()