import torch, time, os

print("CUDA_VISIBLE_DEVICES =", os.getenv("CUDA_VISIBLE_DEVICES"))
ngpu = torch.cuda.device_count()
print("Visible GPUs =", ngpu)

if ngpu < 2:
    raise RuntimeError("Need at least 2 visible GPUs")

# define devices
d0 = torch.device("cuda:0")
d1 = torch.device("cuda:1")

print("GPU 0 =", torch.cuda.get_device_name(0))
print("GPU 1 =", torch.cuda.get_device_name(1))

# allocate tensors on both GPUs
n = 30000
x0 = torch.rand((n, n), device=d0)
x1 = torch.rand((n, n), device=d1)

# run computations in parallel
t0 = time.time()
y0 = torch.matmul(x0, x0)
y1 = torch.matmul(x1, x1)

# synchronize both devices
torch.cuda.synchronize(d0)
torch.cuda.synchronize(d1)

dt = time.time() - t0
print("Parallel compute time:", round(dt, 3), "s")
print("Means:", y0.mean().item(), y1.mean().item())

