package neural

import "testing"

func TestPrepareMLPNetTopology(t *testing.T) {
	layers := []int{4, 5, 3}
	mlp := PrepareMLPNet(layers, 0.1, SigmoidalTransfer, SigmoidalTransferDerivate)

	if len(mlp.NeuralLayers) != len(layers) {
		t.Fatalf("expected %d layers, got %d", len(layers), len(mlp.NeuralLayers))
	}

	for i, want := range layers {
		if got := mlp.NeuralLayers[i].Length; got != want {
			t.Errorf("layer %d: expected %d neurons, got %d", i, want, got)
		}
		if got := len(mlp.NeuralLayers[i].NeuronUnits); got != want {
			t.Errorf("layer %d: expected %d NeuronUnits, got %d", i, want, got)
		}
	}

	// every neuron (besides the input layer) should hold one weight per
	// neuron in the previous layer
	for l := 1; l < len(layers); l++ {
		for _, neuron := range mlp.NeuralLayers[l].NeuronUnits {
			if got := len(neuron.Weights); got != layers[l-1] {
				t.Errorf("layer %d neuron: expected %d weights, got %d", l, layers[l-1], got)
			}
		}
	}
}

func TestExecuteReturnsOneValuePerOutputNeuron(t *testing.T) {
	mlp := PrepareMLPNet([]int{3, 4, 2}, 0.1, SigmoidalTransfer, SigmoidalTransferDerivate)
	pattern := Pattern{Features: []float64{0.1, 0.5, 0.9}}

	out := Execute(&mlp, &pattern)

	if len(out) != 2 {
		t.Fatalf("expected 2 output values, got %d", len(out))
	}
	for i, v := range out {
		if v < 0 || v > 1 {
			t.Errorf("output %d = %v, expected a sigmoid activation in [0, 1]", i, v)
		}
	}
}

// TestBackPropagateReducesErrorOnXOR trains a small MLP on the classic XOR
// problem (which requires a hidden layer to solve) and checks that the
// global error decreases as training progresses.
func TestBackPropagateReducesErrorOnXOR(t *testing.T) {
	mlp := PrepareMLPNet([]int{2, 4, 1}, 0.5, SigmoidalTransfer, SigmoidalTransferDerivate)

	type sample struct {
		in  []float64
		out []float64
	}
	xor := []sample{
		{[]float64{0, 0}, []float64{0}},
		{[]float64{0, 1}, []float64{1}},
		{[]float64{1, 0}, []float64{1}},
		{[]float64{1, 1}, []float64{0}},
	}

	epochError := func() float64 {
		total := 0.0
		for _, s := range xor {
			total += BackPropagate(&mlp, &Pattern{Features: s.in}, s.out)
		}
		return total / float64(len(xor))
	}

	initialError := epochError()
	for epoch := 1; epoch < 2000; epoch++ {
		epochError()
	}
	finalError := epochError()

	if finalError >= initialError {
		t.Fatalf("expected training to reduce average error, initial=%v final=%v", initialError, finalError)
	}
	if finalError > 0.2 {
		t.Errorf("expected the network to reasonably approximate XOR after 2000 epochs, got average error %v", finalError)
	}
}
