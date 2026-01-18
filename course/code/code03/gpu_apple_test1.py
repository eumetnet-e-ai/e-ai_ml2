import torch
import time

print("=== Apple GPU Test (MPS) ===")

# Check availability
if not torch.backends.mps.is_available():
    print("Apple GPU (MPS) not available")
    exit(1)

device = torch.device("mps")
print("Using device:", device)

# Warm-up
x = torch.rand((1024, 1024), device=device)
y = torch.matmul(x, x)
torch.mps.synchronize()

# Timed computation
t0 = time.time()
n  = 6000

x = torch.rand((n, n), device=device)
y = torch.matmul(x, x)
torch.mps.synchronize()

t1 = time.time()

print("Matrix multiplication finished")
print(f"Elapsed time: {t1 - t0:.3f} seconds")
print("Result mean:", y.mean().item())

print("=== GPU computation confirmed ===")

