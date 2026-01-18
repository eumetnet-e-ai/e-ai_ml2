import torch, time

d = torch.device("cuda")
x = torch.rand((8000,8000), device=d)

t0 = time.time()
y = torch.matmul(x, x)
torch.cuda.synchronize()

print("A100 time:",
      round(time.time()-t0,3))

