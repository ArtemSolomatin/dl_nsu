import numpy as np


def l2_regularization(W, reg_strength):
    """
    Computes L2 regularization loss on weights and its gradient

    Arguments:
      W, np array - weights
      reg_strength - float value

    Returns:
      loss, single value - l2 regularization loss
      gradient, np.array same shape as W - gradient of weight by l2 loss
    """
    loss = reg_strength * np.sum(W**2)
    grad = 2 * reg_strength * W
    return loss, grad

def softmax_with_cross_entropy(predictions, target_index):
    """
    Computes softmax and cross-entropy loss for model predictions,
    including the gradient

    Arguments:
      predictions, np array, shape is either (N) or (batch_size, N) -
        classifier output
      target_index: np array of int, shape is (1) or (batch_size) -
        index of the true class for given sample(s)

    Returns:
      loss, single value - cross-entropy loss
      dprediction, np array same shape as predictions - gradient of predictions by loss value
    """
    if len(predictions.shape) == 1: 
        predictions = np.array([predictions])

    exps = np.e ** (predictions - np.max(predictions))
    probs = exps / np.sum(exps, axis = 1)[:, None]
    batch_size = probs.shape[0]

    loss = -np.log(probs[range(batch_size), target_index]).sum() / batch_size

    dprediction = probs.copy()
    dprediction[range(batch_size), target_index] -= 1
    dprediction /= batch_size

    return loss, dprediction


class Param:
    """
    Trainable parameter of the model
    Captures both parameter value and the gradient
    """

    def __init__(self, value):
        self.value = value
        self.grad = np.zeros_like(value)   

class ReLULayer:
    def __init__(self):
        pass

    def forward(self, X):
        self.positive = np.maximum(X, 0)
        return self.positive

    def backward(self, d_out):
        """
        Backward pass

        Arguments:
        d_out, np array (batch_size, num_features) - gradient
           of loss function with respect to output

        Returns:
        d_result: np array (batch_size, num_features) - gradient
          with respect to input
        """

        d_result = d_out * np.sign(self.positive)
        return d_result

    def get_params(self):
        # ReLU Doesn't have any parameters
        return {}


class FullyConnectedLayer:
    def __init__(self, n_input, n_output):
        self.W = Param(0.001 * np.random.randn(n_input, n_output))
        self.B = Param(0.001 * np.random.randn(1, n_output))
        self.X = None

    def forward(self, X):
        self.X = X
        return np.dot(X, self.W.value) + self.B.value

    def backward(self, d_out):
        """
        Backward pass
        Computes gradient with respect to input and
        accumulates gradients within self.W and self.B

        Arguments:
        d_out, np array (batch_size, n_output) - gradient
           of loss function with respect to output

        Returns:
        d_result: np array (batch_size, n_input) - gradient
          with respect to input
        """

        d_B = np.sum(d_out, axis = 0)
        self.B.grad = d_B[np.newaxis, :]
        
        d_W = np.dot(self.X.T, d_out)
        self.W.grad = d_W

        d_input = np.dot(d_out, self.W.value.T)
        return d_input

    def get_params(self):
        return {'W': self.W, 'B': self.B}
