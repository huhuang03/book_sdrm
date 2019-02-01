#!env python3
import numpy as np

A = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
B = A.flatten()

print(B)
print(B[np.array([0, 1, 2])])

# find all in b
print(B > 5)
print(B[B > 5])
