import os, sys, torch, json
os.environ['CUDA_VISIBLE_DEVICES'] = '4'
sys.path.insert(0, '/home/changning/mrts/pylibs')
from transformers import AutoModelForCausalLM
from peft import PeftModel

base = '/home/changning/mrts/models/models/Qwen--Qwen3.5-4B-Base/snapshots/master'
m = AutoModelForCausalLM.from_pretrained(base, torch_dtype=torch.bfloat16, device_map='cuda:0', trust_remote_code=True)

# find actual q_proj/v_proj weights
keys = [n for n, _ in m.named_parameters() if 'q_proj.weight' in n or 'v_proj.weight' in n]
print('sample q/v keys:', keys[:6], '... total', len(keys))
key = keys[0]
print('using key:', key)
w0 = m.get_parameter(key).detach().clone().float()
print('w0 sum', float(w0.sum()))

m2 = PeftModel.from_pretrained(m, '/home/changning/mrts/lora_v3/ckpt_0.50').merge_and_unload().eval()
w1 = m.get_parameter(key).detach().float()
print('after 0.50 merge, |diff|:', float((w1 - w0).abs().sum()), 'base_changed:', not torch.allclose(w0, w1))

m3 = PeftModel.from_pretrained(m, '/home/changning/mrts/lora_v3/ckpt_1.00').merge_and_unload().eval()
w2 = m.get_parameter(key).detach().float()
print('after 1.00 merge, addl diff:', float((w2 - w1).abs().sum()))

mc = AutoModelForCausalLM.from_pretrained(base, torch_dtype=torch.bfloat16, device_map='cuda:0', trust_remote_code=True)
mc2 = PeftModel.from_pretrained(mc, '/home/changning/mrts/lora_v3/ckpt_1.00').merge_and_unload().eval()
wref = mc2.get_parameter(key).detach().float()
print('polluted-vs-clean 1.00 diff:', float((w2 - wref).abs().sum()))
print('CONTAMINATED' if not torch.allclose(w2, wref) else 'CLEAN')