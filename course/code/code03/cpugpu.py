import torch
import time
import os

print("--------------------------------------------------")
print("CPU/GPU Benchmark")
print("Node hostname:", os.uname()[1])
print("PyTorch version:", torch.__version__)
print("--------------------------------------------------")

# Detect CUDA
cuda_available = torch.cuda.is_available()
print("CUDA available:", cuda_available)
if cuda_available:
    print("GPU:", torch.cuda.get_device_name(0))

# Benchmark size
N = 20000
print(f"\nBenchmark size: {N} x {N} matrix multiply\n")

def benchmark(device):
    print(f"--- Device: {device} ---")

    # Generate data
    t0 = time.time()
    x = torch.randn(N, N, device=device)
    y = torch.randn(N, N, device=device)
    if device.type == "cuda":
        torch.cuda.synchronize()
    print("Data generation:", time.time() - t0, "s")

    # Matrix multiply
    t1 = time.time()
    z = x @ y
    if device.type == "cuda":
        torch.cuda.synchronize()
    print("MatMul time:", time.time() - t1, "s")


# CPU test
benchmark(torch.device("cpu"))

# GPU test
if cuda_available:
    benchmark(torch.device("cuda"))

