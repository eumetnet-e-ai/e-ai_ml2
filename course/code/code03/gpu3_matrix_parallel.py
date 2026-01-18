import torch, time, os

assert torch.cuda.device_count() >= 2

d0 = torch.device("cuda:0")
d1 = torch.device("cuda:1")

print("GPU 0:", torch.cuda.get_device_name(0))
print("GPU 1:", torch.cuda.get_device_name(1))

# matrix size
n = 30000

# full matrix on CPU
A = torch.rand((n, n))
B = torch.rand((n, n))

# split A into two blocks
A0, A1 = torch.chunk(A, 2, dim=0)

# move data to GPUs
A0 = A0.to(d0)
A1 = A1.to(d1)
B0 = B.to(d0)
B1 = B.to(d1)

# compute in parallel
t0 = time.time()
C0 = torch.matmul(A0, B0)   # GPU 0
C1 = torch.matmul(A1, B1)   # GPU 1

# sync
torch.cuda.synchronize(d0)
torch.cuda.synchronize(d1)

# gather results back on CPU
C = torch.cat([
    C0.to("cpu"),
    C1.to("cpu")
], dim=0)

print("Model-parallel matmul time:",
      round(time.time() - t0, 3))
print("Result shape:", C.shape)

