import numpy as np

from baudelaire_neural_network.activation_function import Relu, Sigmoid
from baudelaire_neural_network.layer import Layer
from baudelaire_neural_network.loss import BinaryCrossentropy
from baudelaire_neural_network.model import Model
from baudelaire_neural_network.parameters import Parameters


def build_model(parameters=None):
    model = Model(parameters)
    for layer in [
        Layer(54, 54, Relu),
        Layer(54, 54, Relu),
        Layer(54, 54, Relu),
        Layer(54, 54, Relu),
        Layer(54, 1, Sigmoid)
    ]:
        model.add_layer(layer)
    model.add_loss(BinaryCrossentropy)
    return model


def store_parameters(weightings, biases):
    np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)
    np.save("weightings.npy", weightings, allow_pickle=True)
    np.save("biases.npy", biases, allow_pickle=True)


def retrieve_parameters():
    np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)
    weightings = np.load("weightings.npy", allow_pickle=True)
    biases = np.load("biases.npy", allow_pickle=True)
    return Parameters(weightings, biases)


def test_model(data):
    # testing_spam = np.loadtxt(open(testing_data), delimiter=",").astype(int)
    inputs, expected_outputs = data[:, 1:], data[:, 0]
    parameters = retrieve_parameters()
    model = build_model(parameters)
    accuracy = get_accuracy(model.predict(inputs), expected_outputs)
    return accuracy * 100


def get_accuracy(output, expected_output):
    return np.equal(output, expected_output).mean()


def train_model(data, model, epochs, learning_rate):
    params, cost, accuracy = model.train(data[:, 1:], data[:, 0], epochs, learning_rate)
    store_parameters(params.get_weightings(), params.get_biases())


if __name__ == '__main__':
    training_file = "C:\\Users\\Germain Jones\\PycharmProjects\\SpamFilter\\spamclassifier\\spamclassifier\\data\\training_spam.csv"
    testing_file = "C:\\Users\\Germain Jones\\PycharmProjects\\SpamFilter\\spamclassifier\\spamclassifier\\data\\testing_spam.csv"
    training_spam = np.loadtxt(open(training_file), delimiter=",").astype(int)
    testing_spam = np.loadtxt(open(testing_file), delimiter=",").astype(int)
    data = np.concatenate((training_spam, testing_spam))
    model = build_model()
    train_model(data, model, 700, 0.08)
