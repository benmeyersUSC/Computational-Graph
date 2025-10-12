from value import Value
import numpy as np

class Vector(Value):
    def __init__(self, data : np.array, children=(), operation='', requires_grad=True):
        self.data = data
        self.grad = np.zeros(shape=self.data.shape)
        self._backward = lambda : None
        self._prev = set(children)
        self._op = operation
        self.requires_grad = requires_grad
    
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
        assert isinstance(other, Vector), "can only matmul with another Vector"
        # DOT PRODUCT
        if self.data.ndim == 1 and other.data.ndim == 1:
            assert self.data.shape[0] == other.data.shape[0], "vectors must have same length"
            # output is a Value!
            out = Value(np.dot(self.data, other.data), (self, other), '@')
            def _backward():
                # d(a @ b) wrt a = b
                self.grad += other.data * out.grad
                other.grad += self.data * out.grad
            out._backward = _backward
            return out
        # MATRIX MULT
        else:
            # ensure dimension agreement...[m x n] @ [n x k] --> [m x k]
            if self.data.ndim == 2 and other.data.ndim == 2:
                assert self.data.shape[1] == other.data.shape[0], \
                    f"incompatible shapes for matmul: {self.data.shape} @ {other.data.shape}"
            elif self.data.ndim == 2 and other.data.ndim == 1:
                assert self.data.shape[1] == other.data.shape[0], \
                    f"incompatible shapes for matmul: {self.data.shape} @ {other.data.shape}"
            elif self.data.ndim == 1 and other.data.ndim == 2:
                assert self.data.shape[0] == other.data.shape[0], \
                    f"incompatible shapes for matmul: {self.data.shape} @ {other.data.shape}"
            else:
                raise ValueError(f"unsupported dimensions for matmul: {self.data.ndim}D @ {other.data.ndim}D")
            out = Vector(np.matmul(self.data, other.data), (self, other), '@')
            def _backward():
                # d(A @ B) wrt A = out.grad @ B.T
                # A[0,0] impacts C[0,0] and C[0,1] by B[0,0] and B[0,1], respectively
                # so C's gradient's first row needs to dot with B's...that is why we transpose B
                self.grad += np.matmul(out.grad, other.data.T)

                # MatMat or MatVec
                if self.data.ndim == 2:
                    # Matrix @ Matrix
                    other.grad += np.matmul(self.data.T, out.grad)
                # VecMat
                elif self.data.ndim == 1:
                    other.grad += np.outer(self.data, out.grad)
            out._backward = _backward
            return out
        
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