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

class RMSE:
    @staticmethod
    def loss(y_true, y_pred):
        return np.sqrt(np.mean((y_true - y_pred) ** 2))
    
    @staticmethod
    def gradient(y_true, y_pred):
        eps = 1e-12
        mse = np.mean((y_true - y_pred) ** 2)
        return (y_pred - y_true) / (y_true.shape[0] * np.sqrt(mse + eps))
    
class MAE: 
    @staticmethod
    def loss(y_true, y_pred):
        return np.mean(np.abs(y_pred - y_true))

    @staticmethod
    def gradient(y_true, y_pred):
        return np.sign(y_pred - y_true) / y_true.shape[0]
    

