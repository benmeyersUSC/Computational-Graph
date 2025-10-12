import numpy as np

class Value:
    def __init__(self, data, children=(), operation='', requires_grad=True):
        self.data = data              # the actual numerical value
        self.grad = 0.0               # gradient (defaults to 0)
        self._backward = lambda: None # function to propagate gradients
        self._prev = set(children)    # parent nodes in the graph
        self._op = operation          # operation that created this node
        self.requires_grad = requires_grad

    
    # overloaded arithmetic operators
    def __add__(self, other):
        # cast to Value
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')
        
        # addition just passes gradient back
        def _backward():
            self.grad += out.grad      
            other.grad += out.grad
        # set backward method
        out._backward = _backward
        return out
    
    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')
        
        # store values that will be filled in when backward is called
        # that is, *right now, as multiplication occurs*, keep track of two inputs
        def _backward():
            self.grad += other.data * out.grad  
            other.grad += self.data * out.grad
        out._backward = _backward
        return out
    
    def __pow__(self, other):
        assert isinstance(other, (int, float)), "only supporting int/float powers"
        out = Value(self.data ** other, (self,), f'**{other}')
        
        def _backward():
            self.grad += other * (self.data ** (other - 1)) * out.grad
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
        
        self.grad = 1.0  # gradient wrt self is 1
        # reverse the graph to start at end!
        for node in reversed(topo):
            node._backward()


    def parameters(self):
        """get leaf nodes in graph who are trainable...not products, but just internal values = leaves"""
        params = []
        visited = set()
        def collect(v):
            if v not in visited:
                visited.add(v)
                # leaf node AND 
                if len(v._prev) == 0 and v.requires_grad:  
                    params.append(v)
                for child in v._prev:
                    collect(child)
        collect(self)
        return params

    def grad_step(self, lr=0.01):
        """subtract gradient * learning rate for each parameter"""
        for param in self.parameters():
            param.data -= lr * param.grad
            param.grad = np.zeros_like(param.grad)
