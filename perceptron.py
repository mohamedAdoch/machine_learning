import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

class Perceptron:
    def __init__(self, learning_rate, n_iters=1000):
        self.n_iters = n_iters
        self.lr = learning_rate
        self.weights = None
        self.biais = None
        self._actvation_func = lambda x: np.where(x>0, 1, 0)

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.inputs = X
        self.weights = np.zeros(n_features)
        self.biais = 1
        for i in range(self.n_iters):
            for j, x in enumerate(self.inputs):
                sortie = np.dot(x, self.weights)+self.biais
                output_activate= self._actvation_func(sortie)
                self.weights += self.lr*(output_activate-y[j])*x
                self.biais += self.lr*(output_activate-y[j])

    def predict(self, X ):
        y = np.dot(X, self.weights) + self.biais
        activate_output = self._actvation_func(y)
        return y

        