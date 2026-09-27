"""VRAM / throughput probe for e5-large cross-encoder training on this A100 (synthetic worst-case length 128) + inference."""
import time, torch, numpy as np
from transformers import AutoModelForSequenceClassification
from rrL_lib import E5L

torch.set_num_threads(2); torch.manual_seed(0)
m = AutoModelForSequenceClassification.from_pretrained(E5L[0], revision=E5L[1], num_labels=1).cuda()
print("attn", m.config._attn_implementation, "params", sum(p.numel() for p in m.parameters()), flush=True)
opt = torch.optim.AdamW(m.parameters(), lr=1e-6, weight_decay=0.01)
for micro, L in [(128, 128), (256, 128), (128, 84)]:
    try:
        torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats(); m.train()
        ids = torch.randint(5, 250000, (micro, L), device="cuda"); am = torch.ones_like(ids); y = torch.rand(micro, device="cuda")
        for i in range(8):
            if i == 3:
                torch.cuda.synchronize(); t = time.time()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                lo = m(input_ids=ids, attention_mask=am).logits.squeeze(-1)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(lo.float(), y); loss.backward(); opt.step(); opt.zero_grad(set_to_none=True)
        torch.cuda.synchronize(); dt = (time.time() - t) / 5
        print(f"train micro {micro} L {L}: {micro/dt:.0f} pairs/s, peak {torch.cuda.max_memory_allocated()/2**30:.1f} GB", flush=True)
    except torch.OutOfMemoryError:
        print(f"train micro {micro} L {L}: OOM", flush=True)
    opt.zero_grad(set_to_none=True); torch.cuda.empty_cache()
m.eval()
with torch.inference_mode():
    for bs, L in [(1024, 64), (1024, 84)]:
        ids = torch.randint(5, 250000, (bs, L), device="cuda"); am = torch.ones_like(ids)
        for i in range(6):
            if i == 2:
                torch.cuda.synchronize(); t = time.time()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                m(input_ids=ids, attention_mask=am)
        torch.cuda.synchronize(); dt = (time.time() - t) / 4
        print(f"infer bs {bs} L {L}: {bs/dt:.0f} pairs/s", flush=True)
