package math_fx

type RawActivationFx func(float64) float64
type ActivationFxSet struct {
	Actual          RawActivationFx
	FirstDerivative RawActivationFx
}

type RawCostFx func(target, output float64) float64
type CostFxSet struct {
	Actual          RawCostFx
	FirstDerivative RawCostFx
}
