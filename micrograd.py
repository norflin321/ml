## Backpropagation - is an algorithm that allows you to efficiently evaluate the gradient of a loss function,
# with respect to the weights of a neural network. Which allows us to iteratively tune the weights
# of that neural network to minimize the loss function and therefore improve the accuracy of the network.
#
## Derivative - это скорость изменения функции. Насколько круто идет вверх (или вниз) график функции.
# Другими словами — насколько быстро меняется "y" с изменением "x".
# Очевидно, что одна и та же функция в разных точках может иметь разное
# значение производной — то есть может меняться быстрее или медленнее.
# formula: (f(x+h) - f(x)) / h

import math
from utils import draw_dot
import matplotlib.pyplot as plt
import subprocess
import numpy as np

class Value:
    def __init__(self, data, _children=(), _op="", label=""):
        self.label = label
        self.data = data
        self.grad = 0.0 # deriv(self.data -> l)
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op
    def __repr__(self):
        return f"Value(data={self.data})"
    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), "+")
        def _backward():
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad
        out._backward = _backward
        return out
    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), "*")
        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward
        return out
    def __pow__(self, other):
        assert isinstance(other, (int, float)), "only supporting int/float powers now"
        out = Value(self.data**other, (self,), f"**{other}")
        def _backward():
            self.grad += other * (self.data ** (other - 1)) * out.grad
        out._backward = _backward
        return out
    def __truediv__(self, other):
        return self * other**-1
    def __neg__(self):
        return self * -1
    def __sub__(self, other):
        return self + (-other)
    def exp(self):
        x = self.data
        out = Value(math.exp(x), (self, ), "exp")
        def _backward():
            self.grad += out.data * out.grad
        out._backward = _backward
        return out
    def tanh(self):
        x = self.data
        t = (math.exp(2*x) - 1) / (math.exp(2*x) + 1)
        out = Value(t, (self, ), "tanh")
        def _backward():
            self.grad += (1 - t**2) * out.grad
        out._backward = _backward
        return out
    def __radd__(self, other):
        return self + other
    def __rsub__(self, other):
        return other + (-self)
    def __rmul__(self, other):
        return other * self
    def backward(self):
        # we don't want to call _backward for any node before we have done everything after it
        # this ordering of graphs can be achieved using "topological sort"
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev: build_topo(child)
                topo.append(v)
        build_topo(self)
        self.grad = 1.0
        for node in reversed(topo): node._backward();

# so we have multiple inputs here going into a methematical expression, that
# produces a single output (l), and function "draw_dot" is visualizing this "forward pass"
a = Value(2.0, label="a")
b = Value(-3.0, label="b")
c = Value(10.0, label="c")
f = Value(-2.0, label="f")
e = a * b; e.label = "e"
d = c + e; d.label = "d"
l = d * f; l.label = "l"
# print(l, l._prev, l._op)
# next we would like to run "backpropagation". we are going to start at the end (l),
# and going backwards, calculate the "gradient" along all these intermediate values.
# which means to compute the derivative of each node (a, b, c, f, e, d) with respect to (l).
# because we need to know how these weights are impacting the loss function (l).
#
# Value.grad will maintain the derivative of that value to (l), initially it will be zero,
# because at initialization we assumeing that every value does not impact the output (l).

# if we change (l) by an amount of (h), it will change by an amount of (h), obviously.
# it is proportional and therefore deriv(l -> l) = 1
l.grad = 1.0

# now we need to calculate the deriv(f -> l) and deriv(d -> l).
# because they are neighbors, we can use the formula.
# deriv(f -> l) = (d*(f+h) - d*f) / h = (d*f + d*h - d*f) / h = (d*h) / h = d
# deriv(d -> l) = f
f.grad = 4.0 
d.grad = -2.0

# now we know how (l) is sensitive to (f, d).
# but how is (l) sensitive to not neighboring nodes (c, e)?
# to know that, first we need to calculate how nodes (c, e) impacts (d).
# deriv(c -> d) = (e+(c+h) - (e+c)) / h = (e + c + h - e - c) / h = h / h = 1
# deriv(e -> d) = 1
#
## Chain Rule - states that knowing the rate of change of (a) relative to (b) and that of (b) relative
# to (x) allows one to calculate the rate of change of (a) relative to (x) as the product of the two
# rates of change. which means that deriv(a -> x) = deriv(a -> b) * deriv(b -> x)
#
# deriv(c -> l) = deriv(c -> d) * deriv(d -> l) = 1 * -2 = -2
# deriv(e -> l) = deriv(e -> d) * deriv(d -> l) = 1 * -2 = -2
c.grad = -2.0
e.grad = -2.0

# deriv(a -> l) = deriv(e -> l) * deriv(a -> d) = -2.0 * -3.0 
# deriv(a -> l) = deriv(e -> l) * deriv(b -> d) = -2.0 * 2.0
a.grad = -2.0 * -3.0
b.grad = -2.0 * 2.0
# draw_dot(l).render(".tmp/dot1")

## Neuron
# open docs/neuron.jpeg
# You have some inputs (x) and synapses that have weights (w) on them.
# The synapse interacts with the input (x) to this neuron multiplicatively, so what flows to the
# cell body of this neuron is (w * x). But there is multiple inputs (w * x) flowing to the cell body.
#
# The cell body also has some bias (b) which can make input, a bit more or a bit less,
# regardless of the input.
#
# Basically we are taking all the (w * x) of all the inputs, adding the bias (b), and then
# we take it through an activaction function. Which is usually some kind of
# squashing function like a Sigmoid or 10h.
# open docs/fn_10h.png

## Implement Multilayer Perceptron (MLP) model
class Neuron:
    # nin - number of inputs to a neuron
    def __init__(self, nin):
        # weight is an array, with just a random number between -1 and 1 for every one of input number (2.0, 3.0, -1.0)
        self.w = [Value(np.random.uniform(-1,1)) for _ in range(nin)]
        # bias is a random number between -1 and 1, that represents the overall trigger heppiness of this neuron
        self.b = Value(np.random.uniform(-1,1))
    def __call__(self, x):
        # (w * x) + b
        out = (wi * xi for wi, xi in zip(self.w, x))
        out = sum(out, self.b)
        out = out.tanh() # activation function
        return out
    def parameters(self):
        return self.w + [self.b]

class Layer:
    # nin - number of inputs to a layer
    # nout - number of neurons in a single layer
    def __init__(self, nin, nout):
        self.neurons = [Neuron(nin) for _ in range(nout)]
    def __call__(self, x):
        outs = [n(x) for n in self.neurons] # x = xs[i]
        return outs[0] if len(outs) == 1 else outs
    def parameters(self):
        params = []
        for neuron in self.neurons:
            ps = neuron.parameters()
            params += ps
        return params

class MLP:
    # nin - number of inputs to a layer
    # nouts - list of numbers of neurons in a layer
    def __init__(self, nin, nouts):
        arr = [nin] + nouts # concat 2 lists
        self.layers = [Layer(arr[i], arr[i+1]) for i in range(len(nouts))] # [Layer(3, 4), Layer(4, 4), Layer(4, 1)]
    def __call__(self, x):
        for layer in self.layers:
            x = layer(x) # x = xs[i]
        return x
    def parameters(self): # returns all the weights and biases from inside each neuron
        params = []
        for layer in self.layers:
            ps = layer.parameters()
            params += ps
        return params

## Binary Classification (-1.0 or 1.0)
# https://i.imgur.com/7sbLKV3.png
xs = [
    [2.0, 3.0, -1.0], # target: 1.0
    [3.0, -1.0, 0.5], # target: -1.0
    [0.5, 1.0, 1.0],  # target: -1.0
    [1.0, 1.0, -1.0], # target: 1.0
]
ys = [1.0, -1.0, -1.0, 1.0] # desired targets

model = MLP(3, [4, 4, 1])

steps = 100
ypred = []
for step in range(steps):
    ## forward pass
    ypred = [model(x) for x in xs] # ypred is not quite accurate
    # so how do we tune the weights to better predict the desired targets?
    # loss measures the total performance of nn
    ## calculate loss (using simple mean square error loss)
    loss = sum([(yout - ygt)**2 for ygt, yout in zip(ys, ypred)])
    ## backward pass (calculate gradients), but do zero grad before
    for p in model.parameters():
        p.grad = 0.0
    loss.backward()
    ## update (using simple stochastic gradient descent)
    for p in model.parameters():
        p.data += -0.1 * p.grad
    ## print loss
    if step % (steps/10) == 0:
        print(f'step: {step}, loss: {loss.data}')

for i in ypred: print(i)
