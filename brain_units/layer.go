package brain_units

import (
	"sync"

	"github.com/raj96/simple_brain/math_fx"
)

type Layer struct {
	Nodes []*Node

	biasNode        *Node
	activationFxSet math_fx.ActivationFxSet
	backPropBases   []float64
}

func CreateLayer(numberOfNodes int, activationFunction math_fx.ActivationFxSet, nodesInNextLayer int) *Layer {
	layer := &Layer{
		Nodes:           make([]*Node, numberOfNodes),
		activationFxSet: activationFunction,
	}

	for i := 0; i < numberOfNodes; i++ {
		layer.Nodes[i] = CreateNode()
		layer.Nodes[i].InitializeWeights(nodesInNextLayer)
	}

	biasNode := CreateNode()
	biasNode.Value = 1
	biasNode.InitializeWeights(nodesInNextLayer)
	layer.biasNode = biasNode

	return layer
}

func (layer *Layer) ForwardPropagate(prevLayer *Layer) {
	var wg sync.WaitGroup

	wg.Add(len(layer.Nodes))
	for nodeIndex, node := range layer.Nodes {
		go func(node *Node, nodeIndex int) {
			node.BaseValue = 0
			for _, prevNode := range prevLayer.Nodes {
				node.BaseValue += prevNode.Value * prevNode.Weights[nodeIndex]
			}
			node.BaseValue += prevLayer.biasNode.Value * prevLayer.biasNode.Weights[nodeIndex]
			node.Value = layer.activationFxSet.Actual(node.BaseValue)
			wg.Done()
		}(node, nodeIndex)
	}
	wg.Wait()
}

// Only to be run on the last layer
func (layer *Layer) BackwardCostPropagate(prevLayer *Layer, target []float64, costFxSet math_fx.CostFxSet) {
	if len(layer.Nodes) != len(target) {
		panic("BackwardCostPropagate: Dimension mismatch of target and predicted value(s)")
	}

	backPropValues := make([]float64, len(target))
	for index, node := range layer.Nodes {
		backPropValues[index] = costFxSet.FirstDerivative(target[index], node.Value)
		backPropValues[index] *= layer.activationFxSet.FirstDerivative(node.BaseValue)
	}

	prevLayer.backPropBases = backPropValues
}

func (layer *Layer) applyBackProp(learningRate float64) {
	// Apply the backprop values
	for _, node := range layer.Nodes {
		for wIndex := range node.Weights {
			node.Weights[wIndex] -= learningRate * layer.backPropBases[wIndex] * node.Value
		}
	}
	for wIndex := range layer.biasNode.Weights {
		layer.biasNode.Weights[wIndex] -= learningRate * layer.backPropBases[wIndex] * layer.biasNode.Value
	}
}

func (layer *Layer) BackwardPropagate(prevLayer *Layer) {
	// Back propagate new value from this layer
	backPropValues := make([]float64, len(layer.Nodes))
	for bIndex := range backPropValues {
		backPropValues[bIndex] = 0
		for wIndex, weight := range layer.Nodes[bIndex].Weights {
			backPropValues[bIndex] += layer.backPropBases[wIndex] * weight
		}
		backPropValues[bIndex] *= layer.activationFxSet.FirstDerivative(layer.Nodes[bIndex].BaseValue)
	}
	prevLayer.backPropBases = backPropValues
}
