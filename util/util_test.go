package util

import (
	"math"
	"testing"
)

func TestStringInSlice(t *testing.T) {
	slice := []string{"a", "b", "c"}

	if found, index := StringInSlice("b", slice); !found || index != 1 {
		t.Fatalf("expected to find %q at index 1, got found=%v index=%d", "b", found, index)
	}

	if found, index := StringInSlice("z", slice); found || index != -1 {
		t.Fatalf("expected %q not to be found, got found=%v index=%d", "z", found, index)
	}
}

func TestStringToFloatMode0KeepsLengthAndUsesDefault(t *testing.T) {
	result := StringToFloat([]string{"1.5", "oops", "3"}, 0, -1.0)

	want := []float64{1.5, -1.0, 3.0}
	if len(result) != len(want) {
		t.Fatalf("expected %d values, got %d (%v)", len(want), len(result), result)
	}
	for i := range want {
		if result[i] != want[i] {
			t.Errorf("index %d: expected %v, got %v", i, want[i], result[i])
		}
	}
}

func TestStringToFloatMode1DropsInvalidEntries(t *testing.T) {
	result := StringToFloat([]string{"1.5", "oops", "3"}, 1, -1.0)

	want := []float64{1.5, 3.0}
	if len(result) != len(want) {
		t.Fatalf("expected %d values, got %d (%v)", len(want), len(result), result)
	}
	for i := range want {
		if result[i] != want[i] {
			t.Errorf("index %d: expected %v, got %v", i, want[i], result[i])
		}
	}
}

func TestScalarProduct(t *testing.T) {
	got := ScalarProduct([]float64{1, 2, 3}, []float64{4, 5, 6})
	want := 1*4 + 2*5 + 3*6

	if got != float64(want) {
		t.Fatalf("expected %v, got %v", want, got)
	}
}

func TestScalarProductMismatchedLength(t *testing.T) {
	got := ScalarProduct([]float64{1, 2}, []float64{1, 2, 3})
	if got != -1.0 {
		t.Fatalf("expected -1.0 for mismatched slices, got %v", got)
	}
}

func TestMaxInSlice(t *testing.T) {
	max, index := MaxInSlice([]float64{0.1, 0.7, 0.3, 0.2})
	if index != 1 {
		t.Fatalf("expected max at index 1, got index %d (value %v)", index, max)
	}
	if math.Abs(max-0.7) > 1e-9 {
		t.Fatalf("expected max value 0.7, got %v", max)
	}
}

func TestGenerateRandomIntWithBinaryDimIsWithinBounds(t *testing.T) {
	const d = 8
	upperBound := int64(1) << uint(d) // 256

	for i := 0; i < 200; i++ {
		v := GenerateRandomIntWithBinaryDim(d)
		if v < 0 || v >= upperBound {
			t.Fatalf("value %d out of expected range [0, %d)", v, upperBound)
		}
	}
}

func TestConvertIntToBinaryAndBackRoundTrip(t *testing.T) {
	cases := []struct {
		n int64
		d int
	}{
		{0, 8},
		{1, 8},
		{255, 8},
		{42, 8},
		{433, 9},
	}

	for _, c := range cases {
		bits := ConvertIntToBinary(c.n, c.d)
		if len(bits) != c.d {
			t.Fatalf("ConvertIntToBinary(%d, %d): expected %d bits, got %d", c.n, c.d, c.d, len(bits))
		}
		back := ConvertBinToInt(bits)
		if int64(back) != c.n {
			t.Fatalf("round trip failed: ConvertBinToInt(ConvertIntToBinary(%d, %d)) = %d", c.n, c.d, back)
		}
	}
}

func TestRound(t *testing.T) {
	cases := []struct {
		val    float64
		want   float64
		places int
	}{
		{0.6, 1.0, 0},
		{0.4, 0.0, 0},
		{0.123456, 0.12, 2},
	}

	for _, c := range cases {
		got := Round(c.val, .5, c.places)
		if math.Abs(got-c.want) > 1e-9 {
			t.Errorf("Round(%v, .5, %d): expected %v, got %v", c.val, c.places, c.want, got)
		}
	}
}
