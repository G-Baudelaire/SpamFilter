import numpy as np


class ActivationFunction:
    @staticmethod
    def apply_function(z: np.ndarray) -> np.ndarray:
        raise NotImplementedError("Abstract method cannot be called.")

    @staticmethod
    def apply_function_derivative(z: np.ndarray) -> np.ndarray:
        raise NotImplementedError("Abstract method cannot be called.")


class Relu(ActivationFunction):
    @staticmethod
    def apply_function(z: np.ndarray) -> np.ndarray:
        return np.maximum(0, z)

    @staticmethod
    def apply_function_derivative(z: np.ndarray) -> np.ndarray:
        return (z > 0).astype(int)


class Sigmoid(ActivationFunction):
    @staticmethod
    def apply_function(z: np.ndarray) -> np.ndarray:
        return np.divide(1, np.add(1, np.exp(-z)))

    @staticmethod
    def apply_function_derivative(z: np.ndarray) -> np.ndarray:
        sig = Sigmoid.apply_function(z)
        return np.multiply(sig, np.subtract(1, sig))
