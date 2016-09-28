# Go-Bayesian

[![GoDoc](https://godoc.org/github.com/RadhiFadlillah/go-bayesian?status.png)](https://godoc.org/github.com/RadhiFadlillah/go-bayesian)

Go-Bayesian is a Go package for doing classification using Naive-Bayes algorithm. There are two Naive-Bayes models that implemented in this package, which are Multinomial TF and Multinomial Boolean.

## Usage Examples

For basic classifying, you can do it like this:

```go
import (
	"fmt"
	"github.com/RadhiFadlillah/go-bayesian"
)

// Declare class
const (
	Good bayesian.Class = "good"
	Bad  bayesian.Class = "bad"
)

func main() {
	// New Multinomial TF classifier
	classifier := bayesian.NewClassifier(bayesian.MultinomialTf)

	// Do learning using two documents
	classifier.Learn(
		NewDocument(Good, "tall", "handsome", "rich"),
		NewDocument(Bad, "bald", "poor", "ugly"),
	)

	// Classify tokens from a document
	allScores, class, certain := classifier.Classify("the", "tall", "man")
	fmt.Println(allScores, class, certain)
}
```

You also can save the classifier to a file for later use. Useful to avoid repeating learning process :

```go
func main() {
	// New Multinomial TF classifier
	classifier := bayesian.NewClassifier(bayesian.MultinomialTf)
	classifier.Learn(
		NewDocument(Good, "tall", "handsome", "rich"),
		NewDocument(Bad, "bald", "poor", "ugly"),
	)

	// Save classifier to file
	err := classifier.SaveClassifierToFile("./my-classifier")
	if err != nil {
		panic(err)
	}
}
```

Later, you can create a new Classifier from that file :

```go
func main() {
	// New classifier from file
	classifier, err := bayesian.NewClassifierFromFile("./my-classifier")
	if err != nil {
		panic(err)
	}
}
```

## Resource

1. Nazief and Adriani Algorith
2. Raschka, S. 2014. ___Naive Bayes and Text Classification I - Introduction and Theory___. ([PDF](https://arxiv.org/abs/1410.5329v3) and [Website](http://sebastianraschka.com/Articles/2014_naive_bayes_1.html#3_3_multivariate))
3. Metsis, V.,  Androutsopoulos, I., and Paliouras, G. 2006. ___Spam Filtering with Naive Bayes – Which Naive Bayes ?___. Proceeding of CEAS 2006 - Third Conference on Email and Anti-Spam. California, USA, July 27-28, 2006. ([PDF](http://nlp.cs.aueb.gr/pubs/ceas2006_paper.pdf))
4. Lecture [slides](https://web.stanford.edu/~jurafsky/NLPCourseraSlides.html) from the Stanford Coursera course by [Dan Jurafsky](http://web.stanford.edu/~jurafsky/) and [Christopher Manning](http://nlp.stanford.edu/manning/).

## License

Go-Bayesian is distributed using [MIT](http://choosealicense.com/licenses/mit/) license.

## Other Naive-Bayes Implementation

- [Bayesian](https://github.com/jbrukh/bayesian) by [Jake Brukhman](https://github.com/jbrukh)
- [Go-NaiveBayes](https://github.com/lestrrat/go-naivebayes) by [Lestrrat](https://github.com/lestrrat)
