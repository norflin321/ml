// Scalar level autograd engine (core of any neural network training, everything else is just efficiency).
// Port of micrograd (by Andrej Karpathy) in Go, in order to better understand how backpropagation works.

package main

import (
	"fmt"
	"math"
)

type Value struct {
	data     float64
	prev1    *Value
	prev2    *Value
	op       string
	grad     float64
	Backward func()
}

func initValue(n float64) *Value {
	return &Value{n, nil, nil, "init", 0, func() {}}
}

func (self Value) Info() string {
	return fmt.Sprintf("Value{data:%.4f, grad:%.4f}", self.data, self.grad)
}

func (self *Value) Add(other *Value) *Value {
	out := Value{self.data + other.data, self, other, "+", 0, nil}
	backward := func() {
		self.grad = 1.0 * out.grad
		other.grad = 1.0 * out.grad
	}
	out.Backward = backward
	return &out
}

func (self *Value) Mul(other *Value) *Value {
	out := Value{self.data * other.data, self, other, "*", 0, nil}
	backward := func() {
		self.grad = other.data * out.grad
		other.grad = self.data * out.grad
	}
	out.Backward = backward
	return &out
}

func (self *Value) Tanh() *Value {
	t := math.Tanh(self.data)
	out := Value{t, self, nil, "tanh", 0, nil}
	backward := func() {
		self.grad = (1 - math.Pow(t, 2)) * out.grad
	}
	out.Backward = backward
	return &out
}

// func buildTopo()  {
// 	return &Value{n, nil, nil, "init", 0, func() {}}
// }

// Binary Classification Example
func main() {
	xs := [4][3]float64{
		{2.0, 3.0, -1.0},
		{3.0, -1.0, 0.5},
		{0.5, 1.0, 1.0},
		{1.0, 1.0, -1.0},
	}
	ys := [4]float64{1.0, -1.0, -1.0, 1.0}
	fmt.Println(xs, ys)

	w2 := initValue(1.0)
	x2 := initValue(0.0)
	x1 := initValue(2.0)
	w1 := initValue(-3.0)
	b := initValue(6.8812735870195432)

	w2x2 := w2.Mul(x2)
	x1w1 := x1.Mul(w1)
	w2x2_add_x1w1 := w2x2.Add(x1w1)
	n := w2x2_add_x1w1.Add(b)
	o := n.Tanh()

	fmt.Println(o.Info())

	// topo := [...]*Value{}
	// visited := [...]*Value{}

	// steps := 100
	// for step := 0; step <= steps; step++ {
	// 	fmt.Println(step)
	// }
}
