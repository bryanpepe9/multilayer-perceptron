package neural

import "testing"

func TestLoadPatternsFromCSVFile(t *testing.T) {
	patterns, err, mapped := LoadPatternsFromCSVFile("../res/iris.all_data.csv")
	if err != nil {
		t.Fatalf("unexpected error loading dataset: %v", err)
	}

	if len(patterns) == 0 {
		t.Fatal("expected at least one pattern to be loaded")
	}

	if len(mapped) != 3 {
		t.Fatalf("expected 3 distinct iris classes, got %d (%v)", len(mapped), mapped)
	}

	for i, p := range patterns {
		if len(p.Features) != 4 {
			t.Fatalf("pattern %d: expected 4 features, got %d", i, len(p.Features))
		}
		if p.SingleRawExpectation == "" {
			t.Fatalf("pattern %d: expected a non-empty raw expectation", i)
		}
		if p.SingleExpectation < 0 || int(p.SingleExpectation) >= len(mapped) {
			t.Fatalf("pattern %d: SingleExpectation %v out of range for %d classes", i, p.SingleExpectation, len(mapped))
		}
	}
}

func TestLoadPatternsFromCSVFileMissingFile(t *testing.T) {
	_, err, _ := LoadPatternsFromCSVFile("../res/does-not-exist.csv")
	if err == nil {
		t.Fatal("expected an error for a missing file")
	}
}

func TestRawExpectedConversion(t *testing.T) {
	patterns := []Pattern{
		{SingleRawExpectation: "A"},
		{SingleRawExpectation: "B"},
		{SingleRawExpectation: "A"},
	}

	mapped := RawExpectedConversion(patterns)

	if len(mapped) != 2 {
		t.Fatalf("expected 2 distinct classes, got %d (%v)", len(mapped), mapped)
	}
	if patterns[0].SingleExpectation != patterns[2].SingleExpectation {
		t.Errorf("expected identical raw classes to map to the same numeric value, got %v and %v",
			patterns[0].SingleExpectation, patterns[2].SingleExpectation)
	}
	if patterns[0].SingleExpectation == patterns[1].SingleExpectation {
		t.Errorf("expected different raw classes to map to different numeric values")
	}
}

func TestCreateRandomPatternArray(t *testing.T) {
	const dim = 8
	const count = 10

	patterns := CreateRandomPatternArray(dim, count)

	if len(patterns) != count {
		t.Fatalf("expected %d patterns, got %d", count, len(patterns))
	}

	for i, p := range patterns {
		if len(p.Features) != 2*dim {
			t.Fatalf("pattern %d: expected %d features (two %d-bit numbers), got %d", i, 2*dim, dim, len(p.Features))
		}
		if len(p.MultipleExpectation) != dim+1 {
			t.Fatalf("pattern %d: expected sum encoded in %d bits, got %d", i, dim+1, len(p.MultipleExpectation))
		}
	}
}
