package neural

import (
	"math"
	"testing"
)

func TestSigmoidalTransferKnownValues(t *testing.T) {
	if got := SigmoidalTransfer(0); math.Abs(got-0.5) > 1e-9 {
		t.Errorf("SigmoidalTransfer(0) = %v, want 0.5", got)
	}
	// sigmoid should saturate towards 0 / 1 for large magnitude inputs
	if got := SigmoidalTransfer(50); got < 0.999 {
		t.Errorf("SigmoidalTransfer(50) = %v, want close to 1", got)
	}
	if got := SigmoidalTransfer(-50); got > 0.001 {
		t.Errorf("SigmoidalTransfer(-50) = %v, want close to 0", got)
	}
}

// SigmoidalTransferDerivate is called with an already-sigmoided value
// throughout BackPropagate, so it must implement sigma(x) * (1 - sigma(x))
// rather than a constant.
func TestSigmoidalTransferDerivateIsNotConstant(t *testing.T) {
	d1 := SigmoidalTransferDerivate(SigmoidalTransfer(0))
	d2 := SigmoidalTransferDerivate(SigmoidalTransfer(5))

	if math.Abs(d1-0.25) > 1e-9 {
		t.Errorf("derivative at sigmoid(0)=0.5 should be 0.25, got %v", d1)
	}
	if d1 == d2 {
		t.Errorf("derivative should vary with input, got the same value %v for both", d1)
	}
}

func TestHeavysideTransfer(t *testing.T) {
	if got := HeavysideTransfer(0.5); got != 1.0 {
		t.Errorf("HeavysideTransfer(0.5) = %v, want 1.0", got)
	}
	if got := HeavysideTransfer(-0.5); got != 0.0 {
		t.Errorf("HeavysideTransfer(-0.5) = %v, want 0.0", got)
	}
}

func TestHyperbolicTransfer(t *testing.T) {
	if got := HyperbolicTransfer(0); got != 0.0 {
		t.Errorf("HyperbolicTransfer(0) = %v, want 0.0", got)
	}
	want := math.Tanh(1)
	if got := HyperbolicTransfer(1); math.Abs(got-want) > 1e-9 {
		t.Errorf("HyperbolicTransfer(1) = %v, want %v", got, want)
	}
}
