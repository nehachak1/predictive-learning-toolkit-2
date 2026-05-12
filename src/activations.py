import numpy as np

class Sigmoid:
    @staticmethod
    def forward(z):
        return 1 / (1 + np.exp(-z))

    @staticmethod
    def gradient(z):
        s = Sigmoid.forward(z)
        return s * (1 - s)

class ReLU:
    @staticmethod
    def forward(z):
        return np.maximum(0, z)

    @staticmethod
    def gradient(z):
        return (z > 0).astype(float)
    
class Tanh:
    @staticmethod
    def forward(z):
        return np.tanh(z)
    
    @staticmethod
    def gradient(z):
        return 1 - np.tanh(z) ** 2

