import numpy as np

from layers import FullyConnectedLayer, ReLULayer, softmax_with_cross_entropy, l2_regularization


class TwoLayerNet:
    """ Neural network with two fully connected layers """

    def __init__(self, n_input, n_output, hidden_layer_size, reg):
        """
        Initializes the neural network

        Arguments:
        n_input, int - dimension of the model input
        n_output, int - number of classes to predict
        hidden_layer_size, int - number of neurons in the hidden layer
        reg, float - L2 regularization strength
        """
        self.reg = reg
        
        self.layers = [
            FullyConnectedLayer(n_input, hidden_layer_size),
            ReLULayer(),
            FullyConnectedLayer(hidden_layer_size, n_output),
        ]
        

    def compute_loss_and_gradients(self, X, y):
        """
        Computes total loss and updates parameter gradients
        on a batch of training examples

        Arguments:
        X, np array (batch_size, input_features) - input data
        y, np array of int (batch_size) - classes
        """
        #Обнуляем градиент
        for param in self.get_params().values():
            param.grad = np.zeros_like(param.value)
            

        layer_out = X
        for layer in self.layers:
            layer_out = layer.forward(layer_out)
            
        nn_out = layer_out
        loss, d_output = softmax_with_cross_entropy(nn_out, y)
        
        d_layer = d_output
        for layer in reversed(self.layers):
            d_layer = layer.backward(d_layer)
            
        #L2 regularization
        for param in self.get_params().values():
            reg_loss, reg_grad = l2_regularization(param.value, self.reg)
            loss += reg_loss
            param.grad += reg_grad

        return loss
    
    
    def predict(self, X):
        """
        Produces classifier predictions on the set

        Arguments:
          X, np array (test_samples, num_features)

        Returns:
          y_pred, np.array of int (test_samples)
        """
        pred = np.zeros(X.shape[0], np.int)

        layer_out = X
        for layer in self.layers:
            layer_out = layer.forward(layer_out)
        output = layer_out
        
        pred = output.argmax(1)
        return pred

    def get_params(self):
        result = {}

        for i, layer in enumerate(self.layers):
            for key, param in layer.get_params().items():
                result[f'{key}.{i}'] = param

        return result