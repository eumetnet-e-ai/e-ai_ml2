import torch
import time

def gb(x):
    return x / 1024**3

print("=== GPU TEST ===")

print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())

if not torch.cuda.is_available():
    print("No GPU available — exiting.")
    exit(1)

device = torch.device("cuda")
print("Using device:", torch.cuda.get_device_name(0))

# Reset memory stats
torch.cuda.reset_peak_memory_stats()
torch.cuda.empty_cache()

free0, total0 = torch.cuda.mem_get_info()
print(f"\n[MEM] initial free  : {gb(free0):.2f} GB / total {gb(total0):.2f} GB")

# -------------------------------
# GPU computation
# -------------------------------
print("\nRunning GPU test...")

t0 = time.time()

n = 80000
print(f"n={n}")
x = torch.rand((n, n), device=device)
torch.cuda.synchronize()
free1, _ = torch.cuda.mem_get_info()
print(f"[MEM] after x alloc : free {gb(free1):.2f} GB")

y = torch.rand((n, n), device=device)
torch.cuda.synchronize()
free2, _ = torch.cuda.mem_get_info()
print(f"[MEM] after y alloc : free {gb(free2):.2f} GB")

z = torch.matmul(x, y)
torch.cuda.synchronize()

t1 = time.time()

# -------------------------------
# Results
# -------------------------------
used_peak = torch.cuda.max_memory_allocated()
used_now  = torch.cuda.memory_allocated()

free3, _ = torch.cuda.mem_get_info()

print("\n=== RESULTS ===")
print(f"Time on GPU          : {t1 - t0:.4f} s")
print(f"Current alloc        : {gb(used_now):.2f} GB")
print(f"Peak alloc           : {gb(used_peak):.2f} GB")
print(f"Free after compute   : {gb(free3):.2f} GB")
print("Result mean          :", z.mean().item())
print("=== GPU TEST COMPLETED ===")

