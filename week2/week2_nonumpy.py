print("Test. Yes, this is working. Hello World.")

# ----------------------------------------------
# Helper functions


def net_weight_neuron(neuron_inputs, neuron_weights, neuron_bias) -> int:
    """Net weight ((inputs * input weights) + bias) for a single neuron"""
    net_weight_output = 0
    for neuron_input, neuron_weight in zip(neuron_inputs, neuron_weights):
        net_weight_output += neuron_input*neuron_weight
    net_weight_output += neuron_bias
    return net_weight_output

# ----------------------------------------------


# 4 input neurons, beginning of neural network
inputs = [1, 2, 3, 2.5]

# All 4 input neurons are connected to the
# next stage/layer, which in this case
# is a layer composed of 3 neurons
layer_weights = [[0.2, 0.8, -0.5, 1.0],
                 [0.5, -0.91, 0.26, -0.5], [-0.26, -0.27, 0.17, 0.87]]
biases = [2, 3, 0.5]


last_layer_net = [0, 0, 0]
for index, (weights, bias) in enumerate(zip(layer_weights, biases)):
    last_layer_net[index] = net_weight_neuron(
        inputs, weights, bias)

print(last_layer_net)
