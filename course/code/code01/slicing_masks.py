import numpy as np

A = np.array([[1., 2.], [3., 4.]])

B = A[0, 1]          # single element
C = A[:, 0]          # first column
D = A>2              # boolean mask
E = A[A > 2]         # select elements

print("A=", A, "\nB=", B, "\nC=", C, "\nD=", D, "\nE=", E)
