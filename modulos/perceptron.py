import numpy as np


class Perceptron:

    def __init__(self, vec_w):
        self.w_ = vec_w

    def z(self, x):
        return np.dot(self.w_[1:], x) + self.w_[0]

    def predict(self, x):
        return np.where(self.z(x) >= 0.0, 1, -1)
