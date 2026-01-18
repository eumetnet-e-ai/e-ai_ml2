import torch, time, os

print("CUDA_VISIBLE_DEVICES =", os.environ.get("CUDA_VISIBLE_DEVICES"))
print("Visible GPUs =", torch.cuda.device_count())
print("GPU name =", torch.cuda.get_device_name(0))

d = torch.device("cuda")
x = torch.rand((8000,8000), device=d)

t0 = time.time()
y = torch.matmul(x, x)
torch.cuda.synchronize()

print("A100 time:",
      round(time.time()-t0,3))

