package core

import (
	"math/rand"
	"math/isnan"
	"fmt"
)

const (
	functionLinear            = 0
	functionStep              = 1
	functionSignal            = 2
	functionSigmoid           = 3
	functionHiperbolicTangent = 4
	functionGreenGaussian     = 5
)

// Neuron structure
type Neuron struct {
	weights            []float64
	oldWeights         []float64
	output             float64
	err                float64
	activationFunction uint
}

// Init neuron object
func (neuron Neuron) Init(nWeights float64) {
	/*  */
	neuron.weights = make([]float64, nWeights)
	neuron.oldWeights = make([]float64, nWeights)

	neuron.activationFunction = functionLinear
	neuron.err = 0
	neuron.output = 0

	for i := range neuron.weights {
		neuron.weights[i] = -1 + rand.Float64()*(1-(-1))
	}
}

func (neuron Neuron) neuronComputeOutput(x []int64) {
	/*  */
	var sum float64
	for i := range len(neuron.weights) {
		sum += neuron.weights[i] * x[i]
	}

	sum += neuron.weights[i] * 1.0
	switch neuron.activationFunction {
		case functionLinear:
			neuron.output = functionLinear(sum)
			break

		case functionStep:
			neuron.output = functionStep(sum)
			break

		case functionSignal:
			neuron.output = functionSignal(sum)
			break

		case functionSigmoid:
			neuron.output = functionSigmoid(sum)
			break

		case functionHiperbolicTangent:
			neuron.output = functionHiperbolicTangent(sum)
			break
		
		default:
			fmt.Println("ERROR: Invalid activation function")
	}

	if (isnan(neuron.output)) {
		fmt.Println("ERROR[isNaN]: Divergence error")
		os.exit(2)
	}
}
