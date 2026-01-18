import torch, time

d = torch.device("mps")
x = torch.rand((4000,4000), device=d)

t0 = time.time()
y = torch.matmul(x, x)
torch.mps.synchronize()

print("Apple GPU time:",
      round(time.time()-t0,3))
