import numpy as np

def function_2(x):
    return x[0] ** 2 + x[1] ** 2

def numerical_gradient(f, x):
    h = 1e-4
    rst = np.zeros_like(x)

    for i in range(0, x.size):
        tmp_x = x[i]
        x[i] = tmp_x + h
        f2 = f(x)

        x[i] = tmp_x - h
        f1 = f(x)

        rst[i] = (f2 - f1) / (2 * h)

    return rst


print(numerical_gradient(function_2, np.array([3.0, 4.0])))
print(numerical_gradient(function_2, np.array([0, 2.0])))
