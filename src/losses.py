import numpy as np

class MSE:
    @staticmethod
    def loss(y_true, y_pred):
        """
        :param y_true: (array) One hot encoded truth vector.
        :param y_pred: (array) Prediction vector
        :return: (flt)
        """
        return np.mean((y_true - y_pred) ** 2)

    @staticmethod
    def gradient(y_true, y_pred):
        return 2 * (y_pred - y_true) / y_true.shape[0]
    

class CrossEntropy: 
    @staticmethod
    def loss(y_true, y_pred): 
        eps = 1e-12
        y_pred = np.clip(y_pred, eps, 1 - eps)

        return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    
    @staticmethod
    def gradient(y_true, y_pred):
        eps = 1e-12
        y_pred = np.clip(y_pred, eps, 1 - eps)

        return (y_pred - y_true) / (y_pred * (1 - y_pred) * y_true.size)