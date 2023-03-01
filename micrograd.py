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
        return Value(self.data + other.data, (self, other), "+")
    def __mul__(self, other):
        return Value(self.data * other.data, (self, other), "*")
    def tanh(self):
        x = self.data
        t = (math.exp(2*x) - 1) / (math.exp(2*x) + 1)
        return Value(t, (self, ), "tanh")

# so we have multiple inputs here going into a methematical expression, that
# produces a single output (l), and function "draw_dot" is visualizing this "forward pass"
a = Value(2.0, label="a")
b = Value(-3.0, label="b")
c = Value(10.0, label="c")
f = Value(-2.0, label="f")
e = a * b; e.label = "e"
d = c + e; d.label = "d"
l = d * f; l.label = "l"
print(l, l._prev, l._op)
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
# open .tmp/dot1

## Neuron
# open docs/neuron.jpeg
# You have some inputs (x) and synapses that have weights (w) on them.
# The synapse interacts with the input (x) to this neuron multiplicatively, so what flows to the
# cell body of this neuron is (w*x). But there is multiple inputs (w*x) flowing to the cell body.
#
# The cell body also has some bias (b) which can make input, a bit more or a bit less,
# regardless of the input.
#
# Basically we are taking all the (w*x) of all the inputs, adding the bias (b), and then
# we take it through an activaction function. Which is usually some kind of
# squashing function like a Sigmoid or 10h.
# open docs/fn_10h.png

# inputs x1, x2
x1 = Value(2.0, label="x1")
x2 = Value(0.0, label="x2")
# weights w1, w2 (synaptic strengths for each input)
w1 = Value(-3.0, label="w1")
w2 = Value(1.0, label="w2")
# bias of the neuron
b = Value(6.8813735870195432, label="b")
# x1*w1 + x2*w2 + b
x1w1 = x1*w1; x1w1.label = "x1*w1"
x2w2 = x2*w2; x2w2.label = "x2*w2"
x1w1x2w2 = x1w1 + x2w2; x1w1x2w2.label = "x1*w1 + x2*w2"
n = x1w1x2w2 + b; n.label = "n"
o = n.tanh(); o.label = "o"
draw_dot(o).render(".tmp/dot2")

# open .tmp/dot2
# now we are going to do "backpropagation" and fill in all the gradients.
# in a typical neural network setting, we mostly care bout derivatives of weights (w1, w2).
# because we are going to change them as part of the optimization
