import torch, time
torch.set_default_dtype(torch.float16)
n = 30000
A0 = torch.rand((n//2,n), device="cuda:0")
A1 = torch.rand((n//2,n), device="cuda:1")
B  = torch.rand((n,n),     device="cuda:0")
t0 = time.time()
C0 = A0 @ B
C1 = A1 @ B.to("cuda:1")
torch.cuda.synchronize()
print("Time:", round(time.time()-t0,3))
