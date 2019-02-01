#encoding=utf-8
import numpy as np

def cross_entropy_error(y, t):
    delta = 1e-7
    if y.ndim == 1:
        y = y.reshape(1, y.size)
        t = t.reshape(1, t.size)

    batch_size = y.shape[0]
    return -np.sum(t * np.log(y + delta)) / batch_size

def softmax(x):
    exp_a = np.exp(x)
    sum_exp_a = np.sum(exp_a)
    return exp_a / sum_exp_a

class SimpleNet:
    def __init__(self):
        self.W = np.random.randn(2, 3)#用高斯分布进行初始化

    def predict(self, x):
        return np.dot(x, self.W)

    def loss(self, x, t):
        z = self.predict(x)
        y = softmax(z)
        loss = cross_entropy_error(y, t)
        return loss
