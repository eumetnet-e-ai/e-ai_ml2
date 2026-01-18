import numpy as np

def normalize(x):
    x = np.asarray(x)
    return (x - x.mean()) / x.std()

x = np.array([2., 3., 5., 9.])
print("x =", x)
print("z =", normalize(x))
