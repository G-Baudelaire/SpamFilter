from typing import List

import numpy as np

from baudelaire_neural_network.layer import Layer
from baudelaire_neural_network.loss import Loss


class Model:
    _layers: List[Layer]
    _w: List[np.ndarray]
    _b: List[np.ndarray]

    def __init__(self):
        self._layers = list()
        self._w = None
        self._b = None

    def add(self, layer: Layer) -> None:
        self._layers.append(layer)

    def compile(self, loss: Loss):
        self._loss = loss

    def init_layers(self, seed=19):
        np.random.seed(seed)
        w_function = lambda layer: 0.1 * np.random.randn(layer.get_output_dimension(), layer.get_input_dimension())
        b_function = lambda layer: 0.1 * np.random.randn(layer.get_output_dimension(), 1)
        self._w = [w_function(layer) for layer in self._layers]
        print([arr.shape for arr in self._w])
        self._b = [b_function(layer) for layer in self._layers]
        print([arr.shape for arr in self._b])

    def single_layer_forward_propagation(self, a_previous, w_current, b_current, activation):
        z_curr = np.add(np.dot(w_current, a_previous), b_current)
        return activation.function(z_curr), z_curr

    def forward_propagation(self, inputs: np.ndarray):
        a_current = inputs
        a_memory = list()
        z_memory = list()

        for layer, w_current, b_current in zip(self._layers, self._w, self._b):
            a_previous = a_current
            a_current, z_current = self.single_layer_forward_propagation(
                a_previous, w_current, b_current, layer.get_activation()
            )
            a_memory.append(a_previous)
            z_memory.append(z_current)

        return a_current, a_memory, z_memory

    def single_layer_backward_propagation(self, d_a_current, w_current, b_current, z_current, a_previous, activation):
        m = a_previous.shape[1]
        d_z_current = activation.apply_function_derivative(z_current, d_a_current)
        d_w_current = np.divide(np.dot(d_z_current, a_previous.T), m)
        d_b_current = np.divide(np.sum(d_z_current, axis=1, keepdims=True), m)
        d_a_previous = np.dot(w_current.transpose(), d_z_current)
        return d_a_previous, d_w_current, d_b_current

    def backward_propagation(self, y, y_hat, a_memory, z_memory):
        d_w = list()
        d_b = list()
        y = y.reshape(y_hat.shape)
        d_a_previous = - np.subtract(np.divide(y, y_hat), np.divide(np.subtract(1, y), np.subtract(1, y_hat)))

        loop_zip = zip(self._layers, a_memory, z_memory, self._w, self._b)
        for layer, a_previous, z_current, w_current, b_current in reversed(list(loop_zip)):
            d_a_current = d_a_previous
            d_a_previous, d_w_current, d_b_current = self.single_layer_backward_propagation(
                d_a_current, w_current, b_current, z_current, a_previous, layer.get_activation()
            )
            d_w.append(d_w_current)
            d_b.append(d_b_current)
        return d_w, d_b

    def update(self, d_w, d_b, learning_rate: int):
        for index, layer, d_w_value, d_b_value in enumerate(zip(self._layers, d_w, d_b)):
            self._w[index] -= learning_rate * d_w_value
            self._b[index] -= learning_rate * d_b_value

    def get_accuracy_value(self, y_hat, y):
        y_hat_ = np.array(y_hat)
        return (y_hat_ == y).all(axis=0).mean()

    def train(self, x, y, epochs, learning_rate):
        self.init_layers()
        cost_history = list()
        accuracy_history = list()

        for i in range(epochs):
            y_hat, a_memory, z_memory = self.forward_propagation(x)
            cost = self._loss.get_cost_value(y, y_hat)
            cost_history.append(cost)
            accuracy = self.get_accuracy_value(y_hat, y)
            accuracy_history.append(accuracy)

            d_w, d_b = self.backward_propagation(y, y_hat, a_memory, z_memory)
            self.update(d_w, d_b, learning_rate)

        print(self._b)
        print(self._w)
        return cost_history, accuracy_history
