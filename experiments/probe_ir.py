import sys, torch
sys.path.insert(0, "/work/vllm")
import vllm.models.inkling2.xpu.triton_attention as TA

dev = torch.device("xpu")
H, Hkv, D, bs, extent = 12, 4, 128, 64, 1024
ctx, n_q = 4096, 256
nbl = (ctx + n_q + bs - 1) // bs
kv = torch.randn(nbl + 2, Hkv, bs, 2 * D, device=dev).to(torch.bfloat16)
table = (torch.arange(nbl, dtype=torch.int32) + 1).view(1, -1).to(dev)
q = torch.randn(n_q, H, D, device=dev).to(torch.bfloat16)
rel = torch.randn(n_q, H, extent, device=dev)
sink = torch.randn(n_q, H, device=dev).to(torch.bfloat16)
gate = torch.randn(n_q, H * D, device=dev).to(torch.bfloat16)
sl = torch.tensor([ctx + n_q], dtype=torch.int32, device=dev)
qsl = torch.tensor([0, n_q], dtype=torch.int32, device=dev)
pos = torch.arange(ctx, ctx + n_q, device=dev)
TA.triton_paged_attention(q, rel, sink, gate, kv, table, sl, qsl, [0, n_q], pos, extent, None, D**-0.5)
torch.xpu.synchronize()

k = TA._inkling_attn_kernel
for dev_key, cache in k.device_caches.items():
    for key, compiled in cache[0].items():
        ttgir = compiled.asm["ttgir"]
        llir = compiled.asm.get("llir", "")
        ptx_like = compiled.asm.get("ttcir", "")
        print("=== kernel variant key:", str(key)[:120].replace("\n", " "))
        print("dpas:", [l.strip()[:100] for l in ttgir.splitlines() if "dpas" in l.lower()][:2])
        print("tt.dot count:", ttgir.count("tt.dot"))
        for pat in ("async_copy_global_to_local", "local_load", "ttg.barrier", "nbarrier", "prefetch", "ttg.memdesc"):
            print(f"  {pat}:", ttgir.count(pat))
        if llir:
            for pat in ("lsc", "dpas", "llvm.icmp", "integer division", "udiv", "sdiv"):
                print(f"  llir {pat}:", llir.count(pat))
        print("n_regs/n_spills:", getattr(compiled, "n_regs", "?"), getattr(compiled, "n_spills", "?"))