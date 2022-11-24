import numpy as np


class Loss:
    @staticmethod
    def get_cost_value(y: np.ndarray, y_hat: np.ndarray) -> np.ndarray:
        raise NotImplementedError("Abstract method cannot be called.")


class BinaryCrossentropy(Loss):
    @staticmethod
    def get_cost_value(y: np.ndarray, y_hat: np.ndarray) -> np.ndarray:
        m = y_hat.shape[1]
        p1 = np.divide(-1, m)
        p2 = np.dot(y, np.log(y_hat).transpose())
        p3 = np.dot(np.subtract(1, y), np.log(np.subtract(1, y_hat)).transpose())
        cost = np.multiply(p1, np.add(p2, p3))
        return np.squeeze(cost)
