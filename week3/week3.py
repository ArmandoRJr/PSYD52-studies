import numpy as np
import nnfs
from nnfs.datasets import spiral_data
import matplotlib.pyplot as plt

# When saying [x, y] in a matrix, I mean [row size, column size]

nnfs.init()


# ----------------------------------------------
# Helper functions and classes


class Layer_Dense:
    # Variables that are not defined yet:
    # self.weights
    #   (a randomized set of weights for each neuron in our layer,
    #   using a [n_inputs, n_neurons]-sized matrix)
    # self.biases
    #   (a 0-ed set of biases for each neuron in our layer,
    #   using a [1, n_neurons]-sized vector)

    def __init__(self, n_inputs, n_neurons):
        # Initiate weights and biases, as described above
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))

    def forward(self, inputs):
        # Calculate layer output from inputs, weights and biases
        # Inputs *must* be of size [w/e, n_inputs] to multiply correctly
        # meaning each row is a sample
        self.output = np.dot(inputs, self.weights) + self.biases


# If x < 0, clip output to 0
# If y > 0, keep the output
class Activation_ReLU:
    # Matrix can be of whatever size, it doesn't matter
    # but ideally it's [sample_size, n_neurons] (from above)
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)


class Activation_Softmax:
    def forward(self, inputs):
        # Get unnormalized probabilities
        # i.e:
        # 1. for every input in a possible [sample_size, n_neurons] matrix:
        #   take the matrix, and every max value in each row
        #   (so [sample_size] values, resulting in a [sample_size, 1] matrix),
        #   then substract it from every number in said row
        # 2. Get the exponential of each number in the matrix
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        # Get normalized values for each sample
        # i.e:
        # Take the sum of each value in each row (which will result in a [n_neurons, 1] matrix)
        # then, for each value in said row, divide by the sum of the numbers in the row
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output = probabilities


# ----------------------------------------------

"""
Outline of neural network:
- Layer 0:
    Input layer, 2 neurons (for X, Y coordinates)
- Layer 1:
    Dense hidden layer, 3 neurons
- Layer 2:
    Dense output layer, 3 neurons
"""

# Create the dataset
# When I say [sample_size], the sample size here is 300 (100 * 3)
# AND *NOT* THE 2 AXIS (x,y) WE'RE DEALING WITH HERE

coordinates, coord_index_classes = spiral_data(samples=100, classes=3)
coordinates_clipped = coordinates[:5]

# Gonna' be giving the name 'passthrough' to every
# 1st thing that must happen to a layer's inputs
# In this case, the basic dense layer functions
# (Inputs * Weights) - Biases
Layer_1_Passthrough = Layer_Dense(2, 3)
Layer_1_Activation = Activation_ReLU()

Layer_2_Passthrough = Layer_Dense(3, 3)
Layer_2_Activation = Activation_Softmax()

Layer_1_Passthrough.forward(coordinates_clipped)
Layer_1_Activation.forward(Layer_1_Passthrough.output)
Layer_2_Passthrough.forward(Layer_1_Activation.output)
Layer_2_Activation.forward(Layer_2_Passthrough.output)
print(Layer_2_Activation.output)


# plt.scatter(coordinates[:, 0], coordinates[:, 1], c=coord_index_classes, cmap="brg")
# plt.show()
