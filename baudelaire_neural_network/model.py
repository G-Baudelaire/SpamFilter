from typing import List, Type, Tuple

import numpy as np

from baudelaire_neural_network.activation_function import ActivationFunction
from baudelaire_neural_network.layer import Layer
from baudelaire_neural_network.loss import Loss, BinaryCrossentropy
from baudelaire_neural_network.parameters import Parameters


class Model:
    """
    Class representation of a neural network.
    """

    def __init__(self, parameters: Parameters = None, loss: Loss = BinaryCrossentropy):
        self._parameters = parameters
        self._layers: List[Layer] = list()
        self._cost_history = list()
        self._accuracy_history = list()
        self._loss = loss
        self._init_training_attributes()

    def add_layer(self, layer: Layer) -> None:
        """
        Add layer to network, check that inputs to layer match outputs of previous layer.
        :param layer: Layer object.
        """
        if self._layers and self._layers[-1].get_output_dimension() != layer.get_input_dimension():
            raise ValueError("Previous layer output dimension is not equal to new layer input dimension.")
        else:
            self._layers.append(layer)

    def add_loss(self, loss: Type[Loss]) -> None:
        """
        Add loss function to network.
        :param loss: Object representation of loss function.
        """
        self._loss = loss

    def _init_parameters(self, seed: int = 19) -> None:
        """
        Initialise random parameters of weightings and biases for each layer.
        :param seed: Optional seed for reproducibility.
        """
        np.random.seed(seed)

        def initialise_weighting(layer: Layer) -> np.ndarray:
            """
            Create 2 dimensional (output_dimension, input_dimension) Numpy array of random initial weightings.
            :param layer: Corresponding layer of the neural network.
            :return: Two dimensional Numpy array.
            """
            return 0.1 * np.random.randn(layer.get_output_dimension(), layer.get_input_dimension()).astype(
                np.longdouble)

        def initialise_bias(layer: Layer) -> np.ndarray:
            """
            Create 2 dimensional (output_dimension, 1) Numpy array of random initial biases.
            :param layer: Corresponding layer of the neural network.
            :return: Two dimensional Numpy array.
            """
            return 0.1 * np.random.randn(layer.get_output_dimension(), 1).astype(np.longdouble)

        weightings = [initialise_weighting(layer) for layer in self._layers]
        biases = [initialise_bias(layer) for layer in self._layers]
        self._parameters = Parameters(weightings, biases)

    def _single_layer_forward_propagation(self, layer_index: int, prev_activation: np.ndarray,
                                          activator_function: ActivationFunction) -> np.ndarray:
        """
        Transform the activation values from the previous layer into z_index. Then run the given z_indexes through an
        activator function. Additionally, cache the previous activation values and intermediate z_indexes.
        :param layer_index: Index of the current layer being evaluated.
        :param prev_activation: Activation values from the previous layer.
        :param activator_function: Object representation of the activation function for the current layer.
        :return: Y_hat of the current layer.
        """
        curr_weighting = self._parameters.get_weightings()[layer_index]
        curr_bias = self._parameters.get_biases()[layer_index]
        curr_z_index = np.add(np.dot(curr_weighting, prev_activation), curr_bias)
        self._z_index_cache[layer_index] = curr_z_index
        self._activation_cache[layer_index] = prev_activation
        return activator_function.apply_function(curr_z_index)

    def _forward_propagation(self, inputs: np.ndarray) -> np.ndarray:
        """
        Perform propagation on each layer passing forward the activation values
        :param inputs:
        :return:
        """
        curr_activation = inputs
        for index, layer in enumerate(self._layers):
            prev_activation = curr_activation
            curr_activation = self._single_layer_forward_propagation(index, prev_activation, layer.get_activation())
        return curr_activation

    def _single_layer_backward_propagation(self, index: int, curr_d_activation: np.ndarray,
                                           activation_function: Type[ActivationFunction]) -> np.ndarray:
        """
        Store the gradient of the weightings and biases after calculation for a layer of the neural network. Output
        derivative of the previous activation values.
        :param index: Index of the current layer.
        :param curr_d_activation: Derivatives of the current activation values.
        :param activation_function: Activation function used for this layer.
        :return: Array of the derivatives of the previous activation values
        """
        prev_activation = self._activation_cache[index]
        curr_z_index = self._z_index_cache[index]
        curr_weighting = self._parameters.get_weightings()[index]
        m = prev_activation.shape[1]

        curr_d_of_z_index = np.multiply(curr_d_activation, activation_function.apply_function_derivative(curr_z_index))
        curr_d_of_weight = np.divide(np.dot(curr_d_of_z_index, prev_activation.transpose()), m)
        curr_d_of_bias = np.divide(np.sum(curr_d_of_z_index, axis=1, keepdims=True), m)
        prev_d_of_activation = np.dot(curr_weighting.transpose(), curr_d_of_z_index)
        self._gradients_of_weightings[index] = curr_d_of_weight
        self._gradients_of_biases[index] = curr_d_of_bias
        return prev_d_of_activation

    def _backward_propagation(self, y: np.ndarray, y_hat: np.ndarray) -> None:
        """
        Perform backwards propagation on the neural network layer by layer from the end to start.
        :param y: Expected output.
        :param y_hat: Actual output of the neural network.
        """
        y = np.reshape(y, y_hat.shape)

        d1 = np.divide(y, y_hat, out=np.zeros(y.shape), where=(y_hat != 0))
        s1 = np.subtract(1, y)
        s2 = np.subtract(1, y_hat)
        d2 = np.divide(s1, s2, out=np.zeros(s1.shape), where=(s2 != 0))  # Prevents divisions by zero
        prev_d_of_activation = -np.subtract(d1, d2)
        for index, layer in reversed(list(enumerate(self._layers))):
            curr_d_of_activation = prev_d_of_activation
            prev_d_of_activation = self._single_layer_backward_propagation(
                index, curr_d_of_activation, layer.get_activation()
            )

    def _update(self, learning_rate: float) -> None:
        """
        Update the weighting and bias parameters.
        :param learning_rate: Learning rate the neural network is currently being trained at.
        """
        self._parameters.update_weightings(learning_rate, self._gradients_of_weightings)
        self._parameters.update_biases(learning_rate, self._gradients_of_biases)

    def _convert_probabilities_to_ones_and_zeros(self, y_hat: np.ndarray) -> np.ndarray:
        """
        Rounds the value of a probability to the nearest integer so every value is either a 0 or 1.
        :param y_hat: Output of the neural network.
        :return: Array of 0s and 1s
        """
        return np.round(y_hat)

    def _get_accuracy_value(self, y: np.ndarray, y_hat: np.ndarray) -> float:
        """
        Get the percentage of correct outputs.
        :param y: Expected output.
        :param y_hat: Actual output of the neural network.
        :return: Float in the range [0, 1]
        """
        return np.equal(y, self._convert_probabilities_to_ones_and_zeros(y_hat)).mean()

    def _init_training_attributes(self) -> None:
        """
        Initialise arrays for storing numpy arrays while training the neural network.
        """
        self._activation_cache = [None for i in range(len(self._layers))]
        self._z_index_cache = [None for i in range(len(self._layers))]
        self._gradients_of_weightings = [None for i in range(len(self._layers))]
        self._gradients_of_biases = [None for i in range(len(self._layers))]

    def _prepare_input_data(self, x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Transforms the arrays into a format for use in the training process.
        :param x: Data for all entries.
        :param y: The expected output for all entries.
        :return: Data and expected output in a more useful arrangement.
        """
        return x.transpose(), np.reshape(y, [1, -1])

    def train(self, x: np.ndarray, y: np.ndarray, epochs: int, learning_rate: float) -> Tuple[
        Parameters, List[float], List[float]]:
        """
        Train the neural network.
        :param x: Data for all entries.
        :param y: The expected output for all entries.
        :param epochs: Number of iterations to train the neural network with.
        :param learning_rate: Learning rate of the neural network.
        :return: Parameter object, array of the costs, array of the network's accuracy.
        """
        x, y = self._prepare_input_data(x, y)

        if self._parameters is None:
            self._init_parameters()

        for i in range(epochs):
            self._init_training_attributes()
            y_hat = self._forward_propagation(x)
            self._cost_history.append(self._loss.get_cost_value(y, y_hat))
            self._accuracy_history.append(self._get_accuracy_value(y, y_hat))
            self._backward_propagation(y, y_hat)
            self._update(learning_rate)

        return self._parameters, self._cost_history, self._accuracy_history

    def predict(self, data: np.ndarray) -> np.ndarray:
        """
        Predict outputs from the given data. Raise error if neural network has no parameters object (has not been
        trained).
        :param data: Data of entries.
        :return: Array outputting 0s and 1s
        """
        if self._parameters is None:
            raise NotImplementedError("Model has not been trained yet.")

        curr_activation = data.transpose()
        for index, layer in enumerate(self._layers):
            prev_activation = curr_activation
            curr_weighting = self._parameters.get_weightings()[index]
            curr_bias = self._parameters.get_biases()[index]
            curr_z_index = np.add(np.dot(curr_weighting, prev_activation), curr_bias)
            curr_activation = layer.get_activation().apply_function(curr_z_index)
        return self._convert_probabilities_to_ones_and_zeros(curr_activation)
