package math_fx

import "math"

var Sigmoid ActivationFxSet = ActivationFxSet{
	Actual: func(x float64) float64 {
		return 1 / (1 + math.Exp(-x))
	},
	FirstDerivative: func(x float64) float64 {
		sigmoid := 1 / (1 + math.Exp(-x))
		return sigmoid * (1 - sigmoid)
	},
}
