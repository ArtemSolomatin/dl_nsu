import numpy as np
np.random.seed(seed=19)

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
    loss = reg_strength * np.sum(W ** 2)
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
    probs = exps / np.sum(exps, axis=1)[:, None]
    batch_size = probs.shape[0]

    loss = -np.log(probs[range(batch_size), target_index]).sum() / batch_size

    dprediction = probs.copy()
    dprediction[range(batch_size), target_index] -= 1
    dprediction /= batch_size

    return loss, dprediction


class Param:
    '''
    Trainable parameter of the model
    Captures both parameter value and the gradient
    '''

    def __init__(self, value):
        self.value = value
        self.grad = np.zeros_like(value)


class ReLULayer:
    def __init__(self):
        pass

    def forward(self, X):
        self.out = np.maximum(X, 0)
        return self.out

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

        d_result = d_out * np.sign(self.out)
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
        self.X = Param(X.copy())
        return np.dot(self.X.value, self.W.value) + self.B.value

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

        d_B = np.sum(d_out, axis=0)
        self.B.grad = d_B[np.newaxis, :]
        self.W.grad = np.dot(self.X.value.T, d_out)

        d_input = np.dot(d_out, self.W.value.T)
        return d_input

    def get_params(self):
        return {'W': self.W, 'B': self.B}


class ConvolutionalLayer:
    def __init__(self, in_channels, out_channels,
                 filter_size, padding):
        '''
        Initializes the layer

        Arguments:
        in_channels, int - number of input channels
        out_channels, int - number of output channels
        filter_size, int - size of the conv filter
        padding, int - number of 'pixels' to pad on each side
        '''

        self.filter_size = filter_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.W = Param(np.random.randn(filter_size, filter_size, in_channels, out_channels))

        self.B = Param(np.zeros(out_channels))

        self.padding = padding

    def forward(self, X):
        batch_size, height, width, channels = X.shape
        if(channels != self.in_channels):
            print("number of the channels doesn't match")

        # Padding
        self.X = np.pad(X, (
            (0, 0),
            (self.padding, self.padding),
            (self.padding, self.padding),
            (0, 0)
        ), 'constant')

        out_height = (height - self.filter_size + 2 * self.padding) + 1
        out_width = (width - self.filter_size + 2 * self.padding) + 1

        out = np.zeros((batch_size, out_height, out_width, self.out_channels))

        for y in range(out_height):
            for x in range(out_width):
                x_window = self.X[:, y:y+self.filter_size, x:x+self.filter_size, :]
                x_window = x_window.reshape(batch_size, self.filter_size * self.filter_size * self.in_channels)

                w_flat = self.W.value.reshape(self.filter_size * self.filter_size * self.in_channels, self.out_channels)

                out[:, y, x, :] = np.dot(x_window, w_flat) + self.B.value
        return out

    def backward(self, d_out):
        batch_size, height, width, channels = self.X.shape
        if (channels != self.in_channels):
            print("number of the channels doesn't match")

        _, out_height, out_width, out_channels = d_out.shape

        d_input = np.zeros_like(self.X)

        for y in range(out_height):
            for x in range(out_width):
                x_window = self.X[:, y:y + self.filter_size, x:x + self.filter_size, :]
                x_window = x_window.reshape(batch_size, self.filter_size * self.filter_size * self.in_channels)

                w_flat = self.W.value.reshape(self.filter_size * self.filter_size * self.in_channels, self.out_channels)

                input_corr = np.dot(d_out[:, y, x, :], w_flat.T).reshape(batch_size, self.filter_size, self.filter_size, self.in_channels)

                d_input[:, y:y + self.filter_size, x:x + self.filter_size, :] += input_corr

                self.W.grad = self.W.grad + np.dot(x_window.T, d_out[:, y, x, :]).reshape(self.filter_size, self.filter_size, self.in_channels, out_channels)

        self.B.grad = np.sum(d_out, axis=tuple(range(len(d_out.shape)))[:-1]).reshape(out_channels)
        if (self.padding):
            d_input = d_input[:, self.padding:-self.padding, self.padding:-self.padding, :]

        return d_input

    def get_params(self):
        return {'W': self.W, 'B': self.B}


class MaxPoolingLayer:
    def __init__(self, pool_size, stride):
        '''
        Initializes the max pool

        Arguments:
        pool_size, int - area to pool
        stride, int - step size between pooling windows
        '''
        self.pool_size = pool_size
        self.stride = stride
        self.X = None

    def forward(self, X):
        batch_size, height, width, channels = X.shape
        self.X = X.copy()

        h_pixels = height // self.pool_size
        w_pixels = width // self.pool_size
        out = np.zeros((batch_size, h_pixels, w_pixels, channels))
        self.max_index = np.zeros((h_pixels, w_pixels), dtype=object)

        for y in range(0, height, self.stride):
            for x in range(0, width, self.stride):
                y_step = y // self.stride
                x_step = x // self.stride

                pool_frame = self.X[:, y:y + self.pool_size, x:x + self.pool_size, :].transpose(0, 3, 1, 2) #TODO wtf is this transpose?
                pool_frame_flat = pool_frame.reshape(batch_size*channels, self.pool_size*self.pool_size)

                out[:, y_step, x_step, :] = np.amax(pool_frame_flat, axis=1).reshape(batch_size, channels)
                self.max_index[y_step, x_step] = np.where(pool_frame_flat == np.array(np.amax(pool_frame_flat, axis=1))[:, None]) #TODO wtf is that?

                # TODO нужно еще добавить рассчет остаточной части
        return out

    def backward(self, d_out):
        batch_size, height, width, channels = self.X.shape
        d_input = np.zeros(self.X.shape)

        s = self.stride
        p = self.pool_size
        for ys in range(d_out.shape[1]):
            for xs in range(d_out.shape[2]):
                y = ys * s
                x = xs * s
                pool_max_index = self.max_index[ys, xs]
                pool_frame_flat = np.zeros((batch_size * channels,
                                            p ** 2))  # d_input[:, y:y+p, x:x+p, :].transpose(0, 3, 1, 2).reshape(batch_size*channels, p**2)

                equal_max_counts = np.unique(self.max_index[ys, xs][:][0], return_counts=True)[1]
                pool_frame_flat[pool_max_index] = \
                (d_out[:, ys, xs, :].reshape(batch_size * channels) / equal_max_counts)[pool_max_index[:][0]]

                d_input[:, y:y + p, x:x + p, :] += pool_frame_flat.reshape(batch_size, channels, p, p).transpose(0, 2,
                                                                                                                 3, 1)

        return d_input

    def get_params(self):
        return {}


class Flattener:
    def __init__(self):
        self.X_shape = None

    def forward(self, X):
        batch_size, height, width, channels = X.shape
        self.X_shape = X.shape

        # TODO: Implement forward pass
        # Layer should return array with dimensions
        # [batch_size, hight*width*channels]

        return X.copy().reshape((batch_size, height * width * channels))

    def backward(self, d_out):
        return d_out.reshape(self.X_shape)

    def get_params(self):
        # No params!
        return {}
