package neural

import (
	"math"
	"testing"
)

func TestPredict(t *testing.T) {
	neuron := NeuronUnit{Weights: []float64{1.0, -1.0}, Bias: 0.0}

	if got := Predict(&neuron, &Pattern{Features: []float64{1.0, 0.0}}); got != 1.0 {
		t.Errorf("expected positive activation to predict 1.0, got %v", got)
	}
	if got := Predict(&neuron, &Pattern{Features: []float64{0.0, 1.0}}); got != 0.0 {
		t.Errorf("expected negative activation to predict 0.0, got %v", got)
	}
}

func TestAccuracy(t *testing.T) {
	actual := []float64{1, 0, 1, 1}
	predicted := []float64{1, 0, 0, 1}

	correct, percentage := Accuracy(actual, predicted)
	if correct != 3 {
		t.Errorf("expected 3 correct predictions, got %d", correct)
	}
	if percentage != 75.0 {
		t.Errorf("expected 75%% accuracy, got %v", percentage)
	}
}

func TestAccuracyMismatchedLength(t *testing.T) {
	correct, percentage := Accuracy([]float64{1, 0}, []float64{1})
	if correct != -1 || percentage != -1.0 {
		t.Errorf("expected (-1, -1.0) for mismatched slices, got (%d, %v)", correct, percentage)
	}
}

// TestTrainNeuronLearnsAND checks that a single NeuronUnit trained with the
// perceptron learning rule can learn the linearly separable AND function.
func TestTrainNeuronLearnsAND(t *testing.T) {
	patterns := []Pattern{
		{Features: []float64{0, 0}, SingleExpectation: 0},
		{Features: []float64{0, 1}, SingleExpectation: 0},
		{Features: []float64{1, 0}, SingleExpectation: 0},
		{Features: []float64{1, 1}, SingleExpectation: 1},
	}

	neuron := NeuronUnit{Weights: make([]float64, 2), Bias: 0.0, Lrate: 0.1}
	TrainNeuron(&neuron, patterns, 50, 1)

	for _, p := range patterns {
		got := Predict(&neuron, &p)
		if got != p.SingleExpectation {
			t.Errorf("AND(%v) = %v, want %v", p.Features, got, p.SingleExpectation)
		}
	}
}

func TestUpdateWeightsReducesError(t *testing.T) {
	neuron := NeuronUnit{Weights: []float64{0.0, 0.0}, Bias: 0.0, Lrate: 0.1}
	// with zero weights and bias, Predict scores this as 1.0 (scalar product
	// + bias is not < 0.0); expecting 0.0 guarantees a non-zero initial error.
	pattern := Pattern{Features: []float64{1.0, 1.0}, SingleExpectation: 0.0}

	prevError, postError := UpdateWeights(&neuron, &pattern)

	if prevError == 0 {
		t.Fatalf("expected a non-zero error before any training")
	}
	if math.Abs(postError) > math.Abs(prevError) {
		t.Errorf("expected error magnitude to not increase after one weight update: prev=%v post=%v", prevError, postError)
	}
}
