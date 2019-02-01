#!env python3
# 注意这里的乘不是矩阵的乘积，而是相对应的位置的乘

import numpy as np
A = np.array([[1, 2], [3, 4]])
B = np.array([[3, 0], [0, 6]])
print(A * B)
