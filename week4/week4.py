import numpy as np
import nnfs
from nnfs.datasets import spiral_data
import matplotlib.pyplot as plt

# When saying [x, y] in a matrix, I mean [row size, column size]

nnfs.init()

# ----------------------------------------------
# Helper functions and classes


# A layer where all inputs from the previous layer
# are connected to every neuron in the current layer
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


# Softmax functions will normalize all the values
# (i.e. ln them, haha)
# and then get the probability for each value
# where probability = value / sum(values))
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


# Common loss class (Categorical cross-entropy loss class)
# Loss means how far away our final probabilities are
# from our desired results.
class Loss:
    def calculate(self, output, y):
        # Calculate sample losses
        sample_losses = self.forward(output, y)
        # Get mean of each loss
        data_loss = np.mean(sample_losses)
        # Return loss
        return data_loss

    def forward(self, output, y):
        return []


# It's literally just...
# Given the probability array (e.g. [0.7, 0.1, 0.2])
# + our desired ground truth ([1, 0, 0])
# => -math.log(val in prob_array) * val in ground_truth_array
class Loss_CaterogicalCrossEntropy(Loss):
    # Forward pass
    def forward(self, y_pred, y_true):
        samples = len(y_pred)

        # Clip data because log(0) is very damn undefined
        # and the largest value because large numbers are also bad, haha
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)

        # Probabilities for target values
        # With the power of NumPy and weird indexing,
        # we can simplify this function a hell ton.

        # If the shape is 1, then we have categorical labels!
        # Meaning your y_true probably looks like [0, 1, 2, 2, 1, 0, ...]
        # MUST be of the same length as y_pred
        if len(y_true.shape) == 1:
            correct_confidences = y_pred_clipped[range(samples), y_true]
        # Mask values, for one-hot encoded labels
        # Your y_true probably looks like [[0, 0, 1], [1, 0, 0], [0, 1, 0], ...]
        elif len(y_true.shape) == 2:
            correct_confidences = np.sum(y_pred_clipped * y_true, axis=1)

        # Losses
        negative_log_likelihoods = -np.log(correct_confidences)
        return negative_log_likelihoods


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
coord_index_classes_clipped = coord_index_classes[:5]

# Gonna' be giving the name 'passthrough' to every
# 1st thing that must happen to a layer's inputs
# In this case, the basic dense layer functions
# (Inputs * Weights) - Biases
Layer_1_Passthrough = Layer_Dense(2, 3)
Layer_1_Activation = Activation_ReLU()

Layer_2_Passthrough = Layer_Dense(3, 3)
Layer_2_Activation = Activation_Softmax()

Loss_Function = Loss_CaterogicalCrossEntropy()

Layer_1_Passthrough.forward(coordinates_clipped)
Layer_1_Activation.forward(Layer_1_Passthrough.output)
Layer_2_Passthrough.forward(Layer_1_Activation.output)
Layer_2_Activation.forward(Layer_2_Passthrough.output)
loss = Loss_Function.calculate(Layer_2_Activation.output, coord_index_classes_clipped)
print(Layer_2_Activation.output)
print("loss:", loss)


# plt.scatter(coordinates[:, 0], coordinates[:, 1], c=coord_index_classes, cmap="brg")
# plt.show()
