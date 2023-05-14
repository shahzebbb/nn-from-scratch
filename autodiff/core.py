import math

class Value():

    def __init__(self, data: int, _previous=()):
        self.data = data
        self.grad = 0
        self._backward = lambda : None
        self._previous = _previous


    def __repr__(self):
        return f"Value(data={self.data}, grad={self.grad})"


    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other))

        def _backward():
            self.grad += out.grad * 1
            other.grad += out.grad * 1

        out._backward = _backward

        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other))

        def _backward():
            self.grad += out.grad * other.data
            other.grad += out.grad * self.data

        out._backward = _backward

        return out

    def __pow__(self, other):
        
        assert isinstance(other, (int, float))

        out = Value(self.data ** other, (self,))

        def _backward():
            self.grad += out.grad *  (other * self.data**(other - 1))

        out._backward = _backward

        return out

    def relu(self):

        out = Value(0 if self.data < 0 else self.data, (self,))

        def _backward():
            self.grad += out.grad * (0 if self.data < 0 else self.data)

        out._backward = _backward

        return out


    def exp(self):

        out = Value(math.exp(self.data), (self,))

        def _backward():
            self.grad += out.grad * math.exp(self.data)

        out._backward = _backward

        return out
    
    def log(self):

        out = Value(math.log(self.data), (self,))

        def _backward():

            self.grad += out.grad * (1 / self.data)

        out._backward = _backward
        
        return out


    def backward(self):

        topological_sort = []
        visited = set()
        def build_topological_sort(node):
            if node not in visited:
                visited.add(node)
                for child in node._previous:
                    build_topological_sort(child)
                topological_sort.append(node)

        build_topological_sort(self)

        self.grad = 1
        for node in reversed(topological_sort):
            node._backward()


    def __neg__(self):
        return self * -1

    def __sub__(self, other):
        return self + (-other)

    def __truediv__(self,other):
        return self * (other ** -1)

    def __radd__(self, other):
       return self + other

    def __rmul__(self,other):
        return self * other

    def __rsub__(self, other):
        return other + (-self)

    def __rtruediv__(self,other):
        return other * (self ** -1)
    
    def sigmoid(self):
        return 1 / (1+(-1*self).exp())
    
    






