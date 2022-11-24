from typing import List

import numpy as np


class Parameters:
    def __init__(self, weightings: List[np.ndarray], biases: List[np.ndarray]):
        self._weightings = weightings
        self._biases = biases

    def get_weightings(self):
        return tuple(weighting.copy() for weighting in self._weightings)

    def get_biases(self):
        return tuple(bias.copy() for bias in self._biases)

    def update_weightings(self, alpha: float, gradients_of_weightings: List[np.ndarray]):
        for index in range(len(self._weightings)):
            weighting = self._weightings[index]
            gradient = gradients_of_weightings[index]
            self._weightings[index] = np.subtract(weighting, np.multiply(alpha, gradient))

    def update_biases(self, alpha: float, gradients_of_biases:List[np.ndarray]):
        for index in range(len(self._biases)):
            weighting = self._biases[index]
            gradient = gradients_of_biases[index]
            self._biases[index] = np.subtract(weighting, np.multiply(alpha, gradient))
