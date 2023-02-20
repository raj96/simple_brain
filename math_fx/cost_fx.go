package math_fx

import "math"

var MSE CostFxSet = CostFxSet{
	Actual: func(target float64, output float64) float64 {
		return math.Pow(output-target, 2)
	},
	FirstDerivative: func(target, output float64) float64 {
		return (output - target)
	},
}

var Error CostFxSet = CostFxSet{
	Actual: func(target, output float64) float64 {
		return target - output
	},
	FirstDerivative: func(target, output float64) float64 {
		if output > target {
			return -1
		} else if output < target {
			return 1
		} else {
			return 0
		}
	},
}
