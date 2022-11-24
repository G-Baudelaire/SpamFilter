from typing import Type

from baudelaire_neural_network.activation_function import ActivationFunction


class Layer:
    def __init__(self, input_dimension: int, output_dimension: int, activation: Type[ActivationFunction]):
        self._input_dimension = input_dimension
        self._output_dimension = output_dimension
        self._activation = activation

    def get_input_dimension(self):
        return self._input_dimension

    def get_output_dimension(self):
        return self._output_dimension

    def get_activation(self):
        return self._activation
