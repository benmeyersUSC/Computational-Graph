from value import Value
from vector import Vector
import numpy as np

class Perceptron:
    def __init__(self, in_dim : int, out_dim : int, activation='linear', random=True):
        # Choose initialization based on activation
        if activation in ['relu']:
            scale = np.sqrt(2.0 / in_dim)  # He initialization
        elif activation in ['tanh', 'sigmoid']:
            scale = np.sqrt(2.0 / (in_dim + out_dim))  # Xavier initialization
        else:
            scale = np.sqrt(1.0 / in_dim)  # Standard initialization

        weights = np.random.randn(in_dim, out_dim) * scale if random else np.zeros(shape=(in_dim, out_dim))
        self.W = Vector(weights)
        
        self.b = Vector(np.zeros(shape=(1, out_dim)))

        self.activation = self._get_activation(activation)
    
    def _get_activation(self, name):
        if name == 'relu':
            return lambda x: x.relu()
        elif name == 'tanh':
            return lambda x: x.tanh()
        elif name == 'sigmoid':
            return lambda x: x.sigmoid()
        elif name == 'softmax':
            return lambda x: x.softmax()
        else:
            return lambda x: x

    def __call__(self, x : Vector) -> Vector:
        z = (x @ self.W) + self.b
        a = self.activation(z)
        return a
    
    def params(self) -> list[Vector]:
        params = []
        params.extend(self.W.params())
        params.extend(self.b.params())
        return params


class MultiLayerPerceptron:
    def __init__(self, input_shape: tuple[int], layer_dims=(), activations=None):
        self.input_shape = input_shape if len(input_shape) > 1 else (1, input_shape[0])

        if activations is None:
            activations = ['relu'] * (len(layer_dims) - 1) + ['softmax']

        self.layers = []
        in_dim = self.input_shape[1]

        for dim, act in zip(layer_dims, activations):
            layer = Perceptron(in_dim, dim, activation=act)
            self.layers.append(layer)
            in_dim = dim

    def __call__(self, x : Vector) -> Vector:
        for layer in self.layers:
            x = layer(x)
        return x
    
    def params(self) -> list[list[Vector]]:
        params = []
        for layer in self.layers:
            params.extend(layer.params())
        return params
    
    def grad_step(self, lr=0.01):
        for param in self.params():
            param.data -= lr * param.grad
            param.grad = np.zeros_like(param.grad)
    
