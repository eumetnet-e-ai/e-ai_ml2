import torch
import time

print("=== GPU TEST ===")

# Check PyTorch version
print("PyTorch version:", torch.__version__)

# Check if CUDA is available
print("CUDA available:", torch.cuda.is_available())

if not torch.cuda.is_available():
    print("No GPU available — exiting.")
    exit(1)

# Print GPU name
device = torch.device("cuda")
print("Using device:", torch.cuda.get_device_name(0))

# Simple GPU computation test
print("Running a small GPU test...")

t0 = time.time()
n = 15000
x = torch.rand((n, n), device=device)
y = torch.rand((n, n), device=device)
z = torch.matmul(x, y)

torch.cuda.synchronize()
t1 = time.time()

print("Matrix multiplication finished.")
print(f"Time on GPU: {t1 - t0:.4f} seconds")
print("Result mean:", z.mean().item())

print("=== GPU TEST COMPLETED ===")

