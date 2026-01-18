import numpy as np

x = np.array([1., 2.])
A = np.array([[1., 2.], [3., 4.]])

b = A @ x        # matrix-vector
B = A @ A        # matrix-matrix
C = A + 1.0      # broadcasting

print(f"b={b},\nB={B},\nC={C}")
