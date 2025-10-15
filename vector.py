from value import Value
import numpy as np

class Vector(Value):
    def __init__(self, data : np.array, children=(), operation='', requires_grad=True):
        # force >1 dim
        self.data = data if data.ndim > 1 else data.reshape(1, -1)
        self.shape = self.data.shape
        self.grad = np.zeros(shape=self.data.shape)
        self._backward = lambda : None
        self._prev = set(children)
        self._op = operation
        self.requires_grad = requires_grad
    
    def squeeze(self):
        return self.data.squeeze()
    
    # overloaded arithmetic operators
    def __add__(self, other):
        # cast to Vector
        other = other if isinstance(other, Vector) else Vector(other)
        out = Vector(self.data + other.data, (self, other), '+')
        # addition just passes gradient back
        def _backward():
            self.grad += out.grad
            other.grad += out.grad
        # set backward method
        out._backward = _backward
        return out
    
    def __mul__(self, other):
        assert isinstance(other, (int, float, Value)), "only supporting int/float/Value scalars"
        other = other if isinstance(other, Value) else Value(other)
        out = Vector(self.data * other.data, (self, other), '*')
        def _backward():
            # vector gradient is just the scalar
            self.grad += other.data * out.grad
            # scalar gradient is the sum of its effects
            other.grad += np.sum(self.data * out.grad)
        out._backward = _backward
        return out
        
    def __matmul__(self, other):
        """with >1d, now everything is matmul!"""
        assert isinstance(other, Vector), "can only matmul with another Vector"
        out = Vector(np.matmul(self.data, other.data), (self, other), '@')
        
        def _backward():
            # d(A @ B) wrt A = out.grad @ B.T
            # A[0,0] impacts C[0,0] and C[0,1] by B[0,0] and B[0,1], respectively
            # so C's gradient's first row needs to dot with B's...that is why we transpose B
            self.grad += np.matmul(out.grad, other.data.T)
            other.grad += np.matmul(self.data.T, out.grad)
        out._backward = _backward
        return out
    
    def params(self):
        """Get trainable parameters (self if requires_grad)"""
        return [self] if self.requires_grad else []

    def backward(self):
        # recursively build graph...starting at end, though we essentially create a stack
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)
        
        # matmul case --> output is an array
        if isinstance(self.grad, np.ndarray):
            self.grad = np.ones_like(self.grad)  
        # dot case --> output is scalar
        else:
            self.grad = 1.0 
        
        for node in reversed(topo):
            node._backward()

    def relu(self):
        out = Vector(np.maximum(0, self.data), (self,), 'relu')
        def _backward():
            self.grad += (out.data > 0) * out.grad
        out._backward = _backward
        return out
    
    def tanh(self):
        out = Vector(np.tanh(self.data), (self,), 'tanh')
        def _backward():
            self.grad += (1 - np.tanh(self.data)**2) * out.grad
        out._backward = _backward
        return out
    
    def sigmoid(self):
        max_val = np.log(np.finfo(np.float64).max)
        x_clipped = np.clip(self.data, -max_val, max_val)
        
        mask = x_clipped < 0
        out_data = np.empty_like(x_clipped)
        out_data[mask] = np.exp(x_clipped[mask])
        out_data[~mask] = 1 / (1 + np.exp(-x_clipped[~mask]))
        
        out = Vector(out_data, (self,), 'sigmoid')
        def _backward():
            # ds/dx = -(1+e^-x)^-2 * d(1+e^-x)/dx
            #                                       d(1 + e^-x)/dx = e^-x
            #       = -(1+e^-x)^-2 * e^-x
            #       = e^-x / (1+e^-x)^2
            #                                       s(x) = 1 / (1 + e^-x)...
            #                                       1 - s(x) = e^-x / (1 + e^-x)
            #       = 1 / (1+e^-x) * e^-x / (1+e^-x)
            #       = s(x) * (1-s(x))
            self.grad += out.data * (1 - out.data) * out.grad
        out._backward = _backward
        return out
    
    def softmax(self):
        exps = np.exp(self.data - np.max(self.data, axis=-1, keepdims=True))
        softmax_out = exps / np.sum(exps, axis=-1, keepdims=True)
        out = Vector(softmax_out, (self,), 'softmax')
        
        def _backward():
            # Jacobian...f: R^n -> R^m
            # J = m,n matrix
            # J(i,j) = d(f(i)) / d(x(j))
            #        = partial derivative of a function with i...m outputs wrt the j-th...n index of input

            # for softmax, J is n,n where
            # J[i,i] = s[i] * (1 - s[i])            diagonal i=j
            # J[i,j] = -s[i] * s[j]                 off diagonal i!=j 

            # These ^ are the pairwise Jacobian results
            #   but for a single index j in input, we need to sum all of its effects
            #   so we sum J[i,j] for all i 
            #   and multiply by upstream gradient of j

            # upstream gradient g, we have
            # sum_j(J[i,j] * g[j]) 
            # = s[i] * (g[i] - sum_j(s[j] * g[j]))
            self.grad += out.data * (out.grad - np.sum(out.grad * out.data, axis=-1, keepdims=True))
        out._backward = _backward
        return out