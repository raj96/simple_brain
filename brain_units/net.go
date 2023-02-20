package brain_units

import (
	"math"
	"math/rand"
	"time"

	"github.com/raj96/simple_brain/math_fx"
	"github.com/raj96/simple_brain/misc"
)

type Net struct {
	Layers []*Layer

	costFxSet    math_fx.CostFxSet
	learningRate float64
	Loss         float64
}

func CreateNet(netMap []int, costFunction math_fx.CostFxSet, activationFunction math_fx.ActivationFxSet, learningRate float64) *Net {
	rand.Seed(time.Now().UnixNano())

	numberOfLayers := len(netMap)
	net := &Net{
		Layers:       make([]*Layer, numberOfLayers),
		costFxSet:    costFunction,
		learningRate: learningRate,
		Loss:         math.Inf(1),
	}

	// Initialize layers
	net.Layers[0] = CreateLayer(netMap[0], activationFunction, netMap[1])
	for i := 1; i < numberOfLayers-1; i++ {
		net.Layers[i] = CreateLayer(netMap[i], activationFunction, netMap[i+1])
	}
	net.Layers[numberOfLayers-1] = CreateLayer(netMap[numberOfLayers-1], activationFunction, 0)

	return net
}

func (net *Net) SetInput(input []float64) {
	for index, node := range net.Layers[0].Nodes {
		node.Value = input[index]
	}
}

func (net *Net) ForwardPropagate() {
	for i := 1; i < len(net.Layers); i++ {
		net.Layers[i].ForwardPropagate(net.Layers[i-1])
	}
}

func (net *Net) BackwardPropagate(target []float64) {
	numberOfLayers := len(net.Layers)
	net.Layers[numberOfLayers-1].BackwardCostPropagate(net.Layers[numberOfLayers-2], target, net.costFxSet)
	for i := numberOfLayers - 2; i > 0; i-- {
		net.Layers[i].applyBackProp(net.learningRate)
		net.Layers[i].BackwardPropagate(net.Layers[i-1])
	}
	net.Layers[0].applyBackProp(net.learningRate)
}

func (net *Net) GetOutput() []float64 {
	outputs := make([]float64, len(net.Layers[len(net.Layers)-1].Nodes))
	for index, node := range net.Layers[len(net.Layers)-1].Nodes {
		outputs[index] = node.Value
	}

	return outputs
}

func (net *Net) Train(trainingData []misc.TrainingData, epochs int) {
	var rIndex int
	net.Loss = 0
	for epochs > 0 {
		rIndex = rand.Intn(len(trainingData))

		net.SetInput(trainingData[rIndex].Input)
		net.ForwardPropagate()
		net.CalculateLoss(trainingData[rIndex].Output)
		net.BackwardPropagate(trainingData[rIndex].Output)

		epochs--
	}
}

func (net *Net) CalculateLoss(target []float64) {
	output := net.GetOutput()
	if len(target) != len(output) {
		panic("net.CalculateLoss: Dimension error")
	}

	for index := range target {
		net.Loss += net.costFxSet.Actual(target[index], output[index])
	}
}

func (net *Net) IsConverged(acceptableLoss float64) bool {
	return net.Loss <= acceptableLoss
}

func (net *Net) Predict(input []float64) []float64 {
	net.SetInput(input)
	net.ForwardPropagate()

	return net.GetOutput()
}
