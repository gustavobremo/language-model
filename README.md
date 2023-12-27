# N-Gram Language Model

This project implements an N-Gram language model using Python's NLTK library to analyze and generate text based on a given corpus. The model supports tokenization, N-Gram splitting, and different two options for obtaining probabilities such as empirical or with Laplace smoothing.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Functions](#functions)
- [Examples](#examples)

## Installation

1. **Clone this repository:**

    git clone https://github.com/username/n-gram-language-model.git

2.  **Install the required dependencies:**

    pip install -r requirements.txt

## Usage

The main functionalities of this project include:

-    Tokenization of text using NLTK's **word_tokenize**.
-    N-Gram splitting of tokens.
-    Building N-Gram language models with **empirical** or **Laplace** smoothing.

To use the project:
1.  Ensure NLTK is installed:
    ```python
    import nltk

    nltk.download('punkt')

2.  Modify **example_corpus.txt** or replace it with your own corpus for analysis.

3.  Run **main()** from **ngram_model.py**:
    ```bash
    python ngram_model.py
    
## Functions

-   **tokenizer(phrase)**: Tokenizes input text using NLTK's word_tokenize.
-   **read_file(filename)**: Reads a file and tokenizes it into a list of sentences.
-   **split_tokens(tokens, n=2)**: Splits a list of tokens into n-grams.
-   **get_vocabulary_size(sentence_list)**: Calculates the vocabulary size of a given corpus.
-   **build_ngram_model(sentence_tokens_list, n, smoothing="empirical", V=0)**: Builds an N-Gram language model with -   optional empirical or Laplace smoothing.

## Examples
1.  Tokenization

    ```python
    from ngram_model import tokenizer
    
    text = "This is an example sentence for tokenization."
    tokens = tokenizer(text)
    print(tokens)
    ```

2.  N-Gram Model
    ```python
        from ngram_model import build_ngram_model, read_file
        
        corpus = read_file("example_corpus.txt")

        # Empirical model
        empirical_model = build_ngram_model(corpus, n=3, smoothing="empirical")

        # Laplace smoothed model
        vocabulary_size = get_vocabulary_size(corpus)
        laplace_model = build_ngram_model(corpus, n=3, smoothing="laplace", V=vocabulary_size)
    ```
