package brain_units

import (
	"math/rand"
	"sync"
)

type Node struct {
	BaseValue float64
	Value     float64
	Weights   []float64
}

func CreateNode() *Node {
	return &Node{}
}

func (node *Node) InitializeWeights(n int) {
	node.Weights = make([]float64, n)
	wg := sync.WaitGroup{}

	wg.Add(n)
	for i := 0; i < n; i++ {
		go func(index int) {
			node.Weights[index] = rand.Float64()
			wg.Done()
		}(i)
	}
	wg.Wait()
}
