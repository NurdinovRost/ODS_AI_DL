import numpy as np

from layers import FullyConnectedLayer, ReLULayer, softmax_with_cross_entropy, l2_regularization, softmax


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
        # TODO Create necessary layers
        self.fc_1 = FullyConnectedLayer(n_input, hidden_layer_size)
        self.relu_1 = ReLULayer()
        self.fc_2 = FullyConnectedLayer(hidden_layer_size, n_output)

    def compute_loss_and_gradients(self, X, y):
        """
        Computes total loss and updates parameter gradients
        on a batch of training examples

        Arguments:
        X, np array (batch_size, input_features) - input data
        y, np array of int (batch_size) - classes
        """
        # Before running forward and backward pass through the model,
        # clear parameter gradients aggregated from the previous pass
        # TODO Set parameter gradient to zeros
        # Hint: using self.params() might be useful!
        for k in self.params().values():
            k.grad = np.zeros_like(k.grad)
        
        # TODO Compute loss and fill param gradients
        # by running forward and backward passes through the model
        X = self.fc_1.forward(X)
        X = self.relu_1.forward(X)
        X = self.fc_2.forward(X)
        
        loss, d_pred = softmax_with_cross_entropy(X, y)
        
        grad = self.fc_2.backward(d_pred)
        grad = self.relu_1.backward(grad)
        grad = self.fc_1.backward(grad)
        
        # After that, implement l2 regularization on all params
        # Hint: self.params() is useful again!
        for p in self.params().values():
            l, g = l2_regularization(p.value, self.reg)
            loss += l
            p.grad += g
        
        return loss

    def predict(self, X):
        """
        Produces classifier predictions on the set

        Arguments:
          X, np array (test_samples, num_features)

        Returns:
          y_pred, np.array of int (test_samples)
        """
        # TODO: Implement predict
        # Hint: some of the code of the compute_loss_and_gradients
        # can be reused
        pred = np.zeros(X.shape[0], np.int)
        X = self.fc_1.forward(X)
        X = self.relu_1.forward(X)
        X = self.fc_2.forward(X)
        probs = softmax(X)
        pred = np.argmax(probs, axis=1)
        
        return pred

    def params(self):
        # TODO Implement aggregating all of the params
        result = {
                  'fc1_W': self.fc_1.W,
                  'fc1_B': self.fc_1.B,
                  'fc2_W': self.fc_2.W,
                  'fc2_B': self.fc_2.B
                 }

        return result
