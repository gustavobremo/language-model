import nltk
from nltk.tokenize import word_tokenize


def test_fun(tokens_set1, tokens_set2):
    """
    Tests the split_tokens function with various assertions.

    Args:
    - tokens_set1: Set of tokens for testing.
    - tokens_set2: Another set of tokens for testing.
    """

    # Tests for tokens_set1 with n=3
    assert len(split_tokens(tokens_set1, n=3)) == 21
    assert split_tokens(tokens_set1, 3)[0] == ("<BOS>", "<BOS>", "It")
    assert split_tokens(tokens_set1, n=3)[10] == ("we", "are", "dealing")

    # Tests for tokens_set1 with n=2
    assert len(split_tokens(tokens_set1, n=2)) == 20
    assert split_tokens(tokens_set1, n=2)[0] == ("<BOS>", "It")
    assert split_tokens(tokens_set1, n=2)[10] == ("are", "dealing")

    # Tests for tokens_set2 with n=2
    assert len(split_tokens(tokens_set2, n=2)) == 10
    assert split_tokens(tokens_set2, n=2)[0] == ("<BOS>", "How")
    assert split_tokens(tokens_set2, n=2)[9] == ("?", "<EOS>")


def tokenizer(phrase):
    """
    Tokenizes the input phrase using word_tokenize from NLTK.

    Args:
    - phrase: A string containing the text to be tokenized.

    Returns:
    - tokens: A list of tokens extracted from the input phrase.
    """
    tokens = word_tokenize(phrase)

    return tokens


def split_tokens(tokens, n=2):
    """
    Splits a list of tokens into n-grams.

    Args:
    - tokens: A list containing tokens to be split into n-grams.
    - n: An integer indicating the size of the n-grams. Default is 2.

    Returns:
    - split_tokens: A list of tuples representing the generated n-grams.
    """
    # Creating beginning-of-sentence and end-of-sentence tokens
    bos_list = ["<BOS>"] * (n - 1)
    eos_list = ["<EOS>"] * (n - 1)

    # Adding beginning and end tokens to the input tokens
    tokens = bos_list + tokens + eos_list

    k = len(tokens)
    split_tokens = []

    if n > k:
        print("Tokens too small for n value splits")
    elif n < k:
        # Generate n-grams from the tokens
        for i in range(k - n + 1):
            j = i + n

            split_tokens.append(tuple(tokens[i:j]))

    elif n == k:
        print("Same size as tokens")

    return split_tokens


def get_vocabulary_size(corpus):
    # Get unique words by converting the corpus to a set
    unique_words = set(corpus)

    # Return the size of the unique words set
    return len(unique_words)


def build_ngram_model(sentence_tokens_list, n, smoothing="empirical", V=0):
    """
    Build an n-gram language model based on the given sentence tokens.

    Args:
    - sentence_tokens_list (list): List of tokenized sentences.
    - n (int): The order of the n-gram model.
    - smoothing (str): The type of smoothing to be applied. Default is 'empirical'.
    - V (int): Vocabulary size used in Laplace smoothing. Default is 0.

    Returns:
    - dict: A dictionary representing the n-gram model with probabilities.

    """
    # Initialize an empty dictionary to store probabilities
    probabilities_dict = {}

    # Initialize an empty list to store n-grams
    ngrams_lists = []

    # Generate n-grams for each sentence in the list of tokens
    for sentence in sentence_tokens_list:
        ngrams = split_tokens(sentence, n)
        ngrams_lists.append(ngrams)

    # Flatten the list of n-grams
    ngrams_list = [ngram_item for ngram_list in ngrams_lists for ngram_item in ngram_list]

    # Populate dictionary with keys
    for ngram in ngrams_list:
        n_min_1_gram = ngram[0 : n - 1]
        following_word = ngram[-1]
        if n_min_1_gram in probabilities_dict:
            probabilities_dict[n_min_1_gram].append(following_word)
        else:
            probabilities_dict[n_min_1_gram] = [following_word]

    # Apply smoothing based on the specified method
    if smoothing == "empirical":
        for key in probabilities_dict.keys():
            word_list = probabilities_dict[key]
            my_dict = {w: word_list.count(w) / len(word_list) for w in word_list}
            probabilities_dict[key] = my_dict

    elif smoothing == "laplace":
        for key in probabilities_dict.keys():
            word_list = probabilities_dict[key]
            # Calculate Laplace smoothed probabilities
            my_dict = {w: (word_list.count(w) + 1) / (len(word_list) + V) for w in word_list}
            probabilities_dict[key] = my_dict

    return probabilities_dict


def main():
    """
    Main function to create the model and apply different smoothing techniques.
    """

    # Create tokens list per sentence
    sentence_list = []
    with open("example_corpus.txt", "rt") as file:
        for line in file:
            sentence_list.append(list(line.split()))

    corpus = [word for words in sentence_list for word in words]

    # Get the vocabulary size of the corpus
    vocabulary_size = get_vocabulary_size(corpus)

    # Build empirical n-gram model
    model_empirical = build_ngram_model(sentence_list, n=3)

    # Build n-gram model with Laplace smoothing
    model_laplace = build_ngram_model(sentence_list, n=3, smoothing="laplace", V=vocabulary_size)


main()
