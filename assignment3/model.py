import numpy as np
np.random.seed(seed=19)

from layers import (
    FullyConnectedLayer, ReLULayer,
    ConvolutionalLayer, MaxPoolingLayer, Flattener,
    softmax_with_cross_entropy, l2_regularization
    )


class ConvNet:
    """
    Implements a very simple conv net

    Input -> Conv[3x3] -> Relu -> Maxpool[4x4] ->
    Conv[3x3] -> Relu -> MaxPool[4x4] ->
    Flatten -> FC -> Softmax
    """
    def __init__(self, n_input, n_output, conv1_size, conv2_size, reg):
        """
        Initializes the neural network

        Arguments:
        n_input, int - dimension of the model input
        n_output, int - number of classes to predict
        conv1_size, int - number of filters in the 1st conv layer
        conv2_size, int - number of filters in the 2nd conv layer
        reg, float - L2 regularization strength
        """
        self.reg = reg
        height, width, input_channels = n_input

        self.L = [
            ConvolutionalLayer(in_channels=input_channels, out_channels=conv1_size, filter_size=3, padding=1),
            ReLULayer(),
            MaxPoolingLayer(4, 4),
            ConvolutionalLayer(in_channels=conv1_size, out_channels=conv2_size, filter_size=3, padding=1),
            ReLULayer(),
            MaxPoolingLayer(4, 4),
            Flattener(),
            FullyConnectedLayer(8, n_output)
        ]

    def compute_loss_and_gradients(self, X, y):
        """
        Computes total loss and updates parameter gradients
        on a batch of training examples

        Arguments:
        X, np array (batch_size, height, width, input_features) - input data
        y, np array of int (batch_size) - classes
        """
        # clear parameter gradients aggregated from the previous pass
        for index, param in self.params().items():
            param.grad = np.zeros_like(param.value)

        forward_input = X.copy()
        for layer in self.L:
            forward_input = layer.forward(forward_input)

        loss, backward_propagation = softmax_with_cross_entropy(forward_input, y)

        for layer in reversed(self.L):
            backward_propagation = layer.backward(backward_propagation)

            # for reg_param in ['W', 'B']:
            #     if reg_param in layer.params():
            #         loss_l2, dp_l2 = l2_regularization(layer.params()[reg_param].value, self.reg)
            #         loss += loss_l2
            #         layer.params()[reg_param].grad += dp_l2

        return loss

    def predict(self, X):
        pred = np.zeros(X.shape[0], np.int)

        layer_out = X
        for layer in self.L:
            layer_out = layer.forward(layer_out)
        output = layer_out

        pred = output.argmax(1)
        return pred

    def params(self):
        result = {}

        for layer_id, layer in enumerate(self.L):
            for key, value in layer.get_params().items():
                result[key + str(layer_id)] = value

        return result
