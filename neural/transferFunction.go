// Neural provides struct to represents most common neural networks model and algorithms to train / test them.
package neural

import (
	"math"
)

// define transfer function desiderata
type transferFunction func(float64) float64

// different type of transfer function

func HeavysideTransfer(d float64) float64 {

	if d >= 0.0 {
		return 1.0
	}
	return 0.0

}

func HeavysideTransferDerivate(d float64) float64 {

	return 1.0

}

func SigmoidalTransfer(d float64) float64 {

	return 1 / (1 + math.Exp(-d))

}

// SigmoidalTransferDerivate computes the derivative of the sigmoid function.
// It expects d to already be a sigmoid OUTPUT (i.e. the value returned by
// SigmoidalTransfer), as is the case everywhere it's called in BackPropagate:
// sigma'(x) = sigma(x) * (1 - sigma(x)).
func SigmoidalTransferDerivate(d float64) float64 {

	return d * (1 - d)

}

func HyperbolicTransfer(d float64) float64 {

	return math.Tanh(d)

}

func HyperbolicTransferDerivate(d float64) float64 {

	return 1 - math.Pow(d, 2)

}
