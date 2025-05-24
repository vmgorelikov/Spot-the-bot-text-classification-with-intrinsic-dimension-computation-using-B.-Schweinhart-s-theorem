import os
import sys
import numpy as np

def tokens_to_vectors(text: list, dictionary: dict, n: int = 1, d = None, output_mode = "sequence", dictionary_set = set()):
    assert len(list(dictionary.keys())), "Empty dictionary"
    if output_mode == "sequence":
        n = 1
    ngrams_vectors = []
    delete_dictionary_set = False
    if not dictionary_set:
        dictionary_set = set(dictionary.keys())
        delete_dictionary_set = True
    unknown_words = set()
    for position in range(len(text)-n+1):
        ngram = text[position:position+n]
        if set(ngram) - dictionary_set:
            unknown_words = unknown_words | (set(ngram) - dictionary_set)
            continue
        ngram_vectors = []
        for word in ngram:
            ngram_vectors.append(dictionary[word][:d])
        ngrams_vectors.append(np.array(ngram_vectors).flatten())
    if delete_dictionary_set: del dictionary_set
    if output_mode == "sequence":
        return np.array(ngrams_vectors), unknown_words
    elif output_mode == "ngrams":
        return np.unique(np.array(ngrams_vectors), axis=0), unknown_words

# assert output_mode == "ngrams"
# assert len(input_file_paths) == 1