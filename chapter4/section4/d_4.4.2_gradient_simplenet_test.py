from d_4_4_2_gradient_simplenet import SimpleNet
import numpy as np

# x为二维的情况没有处理
def numerical_gradient(f, x):
    h = 1e-4
    grad = np.zeros_like(x)

    height, width = grad.shape
    print(height)
    print(width)

    for i in range(height):
        for j in range(width):
            tmp_val = x[i, j]
            x[i, j] = tmp_val + h
            fxh1 = f(x)

            x[i, j] = tmp_val -h
            fxh2 = f(x)
            grad[i, j] = (fxh1 - fxh2) / (2 * h)
            x[i, j] = tmp_val
    return grad

net = SimpleNet()
print(net.W)

x = np.array([0.6, 0.9])
y = net.predict(x)
print(y)
print(np.argmax(y))
t = np.array([0, 0, 0])
t[np.argmax(y)] = 1
print(t)
print(net.loss(x, t))
t = np.array([1, 0, 0])
print(net.loss(x, t))
t = np.array([0, 1, 0])
print(net.loss(x, t))
t = np.array([0, 0, 1])
print(net.loss(x, t))


print("\nbeing numerical gradient")
def f(W):
    return net.loss(x, t)

dW = numerical_gradient(f, net.W)
print(dW)
