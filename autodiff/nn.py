from autodiff.core import Value
import random

class Base:

    def zero_grad(self):
        for p in self.parameters():
            p.grad = 0

    def parameteres(self):
        return []

class Neuron(Base):

    def __init__(self, input_length, use_relu = True):
        #self.weights = [Value(random.uniform(-1,1)) for _ in range(number_of_inputs)]
        self.weights = list(map(lambda x: Value(random.uniform(-1,1)), range(input_length)))
        self.bias = Value(0)
        #self.use_sigmoid = use_sigmoid
        self.use_relu = use_relu


    def __call__(self, x):
        output = sum([weights_i * x_i for weights_i, x_i in zip (self.weights, x)], self.bias)
        
        if self.use_relu:
            return output.relu()
        else:
            return output
        
    def parameters(self):
        parameters = self.weights + [self.bias]
        return parameters
    
    def __repr__(self):
        return f"{'ReLU ' if self.use_relu else 'Linear '}Neuron({len(self.weights)})"



class Layer(Base):

    def __init__(self,input_length,output_length, **kwargs):
        #self.neurons = [Neuron(input_length, **kwargs) for _ in range(output_length)]
        self.neurons = list(map(lambda x: Neuron(input_length, **kwargs), range(output_length)))

    def __call__(self,x):
        out = list(map(lambda n: n(x), self.neurons))
        return out[0] if len(out) == 1 else out
    
    def parameters(self):
        return [p for n in self.neurons for p in n.parameters()]
    
    def __repr__(self):
        return f"Layer of [{', '.join(str(n) for n in self.neurons)}]"
    

class MLP(Base):
    
    def __init__(self, input_length, output_length_list):
        size = [input_length] + output_length_list
        self.layers = [Layer(size[i], size[i+1], use_relu = i != len(output_length_list)-1) for i in range(len(output_length_list))]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
    
    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]
    
    def load_weights(self, weights):
        assert len(self.parameters()) == len(weights)
        for p, n in zip(self.parameters(), weights):
            p.data = n

    def __repr__(self):
        return f"MLP of [{', '.join(str(layer) for layer in self.layers)}]"