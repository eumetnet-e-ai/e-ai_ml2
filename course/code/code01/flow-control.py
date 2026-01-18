import numpy as np
x = np.array([1.2, 2.0, 1e30, 4.9])
thr = 1e20
# removing values outside of range: 
x2 = x[x < thr]; dn = len(x)-len(x2)

fmt = "{:10.3g}"   # width=10, 3 digits
print("".join(fmt.format(v) for v in x))
print("".join(fmt.format(v) for v in x2))
print("mean=", f"{x2.mean():.3f}")
