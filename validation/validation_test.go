package validation

import (
	"testing"

	mn "multilayer-perceptron/neural"
)

func makePatterns(n int) []mn.Pattern {
	patterns := make([]mn.Pattern, n)
	for i := range patterns {
		patterns[i] = mn.Pattern{
			Features:          []float64{float64(i), float64(i) * 2},
			SingleExpectation: float64(i % 2),
		}
	}
	return patterns
}

func TestTrainTestPatternsSplitSizesAndDisjoint(t *testing.T) {
	patterns := makePatterns(20)

	train, test := TrainTestPatternsSplit(patterns, 0.7, 1)

	if len(train) != 14 {
		t.Fatalf("expected 14 training patterns (70%% of 20), got %d", len(train))
	}
	if len(test) != 6 {
		t.Fatalf("expected 6 testing patterns, got %d", len(test))
	}

	seen := make(map[float64]bool)
	for _, p := range train {
		seen[p.Features[0]] = true
	}
	for _, p := range test {
		if seen[p.Features[0]] {
			t.Errorf("pattern with feature %v present in both train and test sets", p.Features[0])
		}
	}
}

func TestKFoldPatternsSplitCoversAllPatterns(t *testing.T) {
	patterns := makePatterns(23) // not evenly divisible by k, exercises the remainder handling
	const k = 5

	folds := KFoldPatternsSplit(patterns, k, 1)

	if len(folds) != k {
		t.Fatalf("expected %d folds, got %d", k, len(folds))
	}

	total := 0
	seen := make(map[float64]int)
	for _, fold := range folds {
		total += len(fold)
		for _, p := range fold {
			seen[p.Features[0]]++
		}
	}

	if total != len(patterns) {
		t.Fatalf("expected folds to cover all %d patterns, got %d", len(patterns), total)
	}
	for feature, count := range seen {
		if count != 1 {
			t.Errorf("pattern with feature %v appeared in %d folds, want exactly 1", feature, count)
		}
	}
}

func TestKFoldValidationReturnsOneScorePerFold(t *testing.T) {
	patterns, err, _ := mn.LoadPatternsFromCSVFile("../res/sonar.all_data.csv")
	if err != nil {
		t.Fatalf("unexpected error loading dataset: %v", err)
	}

	neuron := mn.NeuronUnit{Weights: make([]float64, len(patterns[0].Features)), Bias: 0.0, Lrate: 0.01}
	const k = 4

	scores := KFoldValidation(&neuron, patterns, 20, k, 1)

	if len(scores) != k {
		t.Fatalf("expected %d scores, got %d", k, len(scores))
	}
	for i, s := range scores {
		if s < 0 || s > 100 {
			t.Errorf("score %d (%v) out of expected [0, 100] percentage range", i, s)
		}
	}
}

func TestMLPKFoldValidationReturnsOneScorePerFold(t *testing.T) {
	patterns, err, mapped := mn.LoadPatternsFromCSVFile("../res/iris.all_data.csv")
	if err != nil {
		t.Fatalf("unexpected error loading dataset: %v", err)
	}

	layers := []int{len(patterns[0].Features), 5, len(mapped)}
	mlp := mn.PrepareMLPNet(layers, 0.1, mn.SigmoidalTransfer, mn.SigmoidalTransferDerivate)
	const k = 3

	scores := MLPKFoldValidation(&mlp, patterns, 20, k, 1, mapped)

	if len(scores) != k {
		t.Fatalf("expected %d scores, got %d", k, len(scores))
	}
	for i, s := range scores {
		if s < 0 || s > 100 {
			t.Errorf("score %d (%v) out of expected [0, 100] percentage range", i, s)
		}
	}
}
