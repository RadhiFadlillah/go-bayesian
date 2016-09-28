package bayesian

import (
	"testing"
)

const (
	Good Class = "good"
	Bad  Class = "bad"
)

func TestClassifier(t *testing.T) {
	classifier := NewClassifier(MultinomialBoolean)
	classifier.Learn(
		NewDocument(Good, "tall", "handsome", "rich"),
		NewDocument(Bad, "bald", "poor", "ugly"),
	)

	_, class, certain := classifier.Classify("the", "tall", "man")
	if class != Good || certain != true {
		t.Fatal("Text 1:", class, "-", certain, "should be Good - true")
	}

	_, class, certain = classifier.Classify("poor", "ugly", "girl")
	if class != Bad || certain != true {
		t.Fatal("Text 2:", class, "-", certain, "should be Bad - true")
	}

	_, class, certain = classifier.Classify("the", "bad", "man")
	if certain != false {
		t.Fatal("Text 1:", class, "-", certain, "should be Good - false")
	}
}
