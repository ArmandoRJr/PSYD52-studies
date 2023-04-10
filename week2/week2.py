import numpy as np
# ----------------------------------------------
# Helper functions

# ----------------------------------------------

"""This is the code for a set of samples.
# A batch, a set of observations, feature set instances.
# Whatever you want to call it.

Let's lay it out like this:
We have made 3 observations in total.
In this network, there's 4 input neurons as the first layer.
Then in the next layer, there's 3 neurons.
Each of these three neurons are connected to the 4 input neurons,
meaning there's 12 connections in total, with 12 weights in total.
"""

# These are our three observations.
# Each list (inside of the big list) is composed of the values of the first layer
# (4 values = 4 neurons).
inputs = [[1, 2, 3, 2.5],  [2.0, 5.0, -1.0, 2.0], [-1.5, 2.7, 3.3, -0.8]]

# These are the connections for each of the
# three neurons in the 2nd layer.
layer_weights = [[0.2, 0.8, -0.5, 1.0],
                 [0.5, -0.91, 0.26, -0.5], [-0.26, -0.27, 0.17, 0.87]]
# And then our cute, little biases for each neuron.
biases = [2.0, 3.0, 0.5]


# NumPy is cute. It lets us do matrix multiplication very easily.
# np.dot will calculate the result of the multiplication of a (3,4) by a (4,3) matrix
# ((Rows, columns) here), resulting in a 3x3 matrix
# There's a transpose on layer_weights because it's a (3,4) matrix, originally.
# Then we're adding biases to every *row* of the final matrix.
layer_outputs = np.dot(inputs, np.array(layer_weights).T) + biases

print(layer_outputs)
