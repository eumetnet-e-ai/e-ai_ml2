import torch, time

torch.set_default_dtype(torch.float16)

d0 = torch.device("cuda:0")
d1 = torch.device("cuda:1")

n = 30000
h = n // 2

# build matrices directly on GPUs
A0 = torch.rand((h, n), device=d0)
A1 = torch.rand((h, n), device=d1)

B0 = torch.rand((n, h), device=d0)
B1 = torch.rand((n, h), device=d1)

t0 = time.time()

# compute partial results
C00 = torch.matmul(A0, B0)   # GPU 0
C01 = torch.matmul(A0.to(d1), B1)  # GPU 1
C10 = torch.matmul(A1.to(d0), B0)  # GPU 0
C11 = torch.matmul(A1, B1)   # GPU 1

torch.cuda.synchronize(d0)
torch.cuda.synchronize(d1)

dt = time.time() - t0
print("Time:", round(dt, 3), "s")

