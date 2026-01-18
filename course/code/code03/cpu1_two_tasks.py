import torch, time

d = torch.device("cpu")
print("Device:", d)

n = 30000
x0 = torch.rand((n, n), device=d)
x1 = torch.rand((n, n), device=d)

t0 = time.time()
y0 = torch.matmul(x0, x0)
y1 = torch.matmul(x1, x1)
dt = time.time() - t0

print("CPU time:", round(dt, 3), "s")
print("Means:", y0.mean().item(), y1.mean().item())

