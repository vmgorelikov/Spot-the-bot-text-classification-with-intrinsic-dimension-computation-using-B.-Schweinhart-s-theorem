import joblib
import text_to_vectors as ttv
import vectors_to_schweinhart as vts
import numpy as np
from random import random
import sys
import spacy

print("Usage: python classifier.py <text> [model] [dictionary]")

ru_core_news = spacy.load("ru_core_news_sm")

def preprocess(text: str, nlp=ru_core_news) -> list[str]:
    doc = nlp(text.replace("\n", " "))
    lemmata = []
    for token in doc:
        if token.pos_ in ["PRON", "DET"]:
            lemmata.append("1")
        elif token.pos_ in ["NUM"]:
            lemmata.append("0")
        elif (not token.is_punct) and (not token.is_space):
            lemmata.append(token.lemma_.lower())
    return lemmata

model_path = "models/separated_test_2_3_4_LinearSVC_balanced_accuracy_0.9911_alpha_0.0010.jl"
dictionary_path = "dictionaries/russian_dict_SVD_100.npy"
try:
    model_path = sys.argv[2]
    dictionary_path = sys.argv[3]
except:
    pass

text = preprocess(open(sys.argv[1], "r", encoding="utf8").read())
clf = joblib.load(open(model_path, "rb"))
vectors, _ = ttv.tokens_to_vectors(text, np.load(dictionary_path, allow_pickle=True).item(), 1, 100, output_mode="sequence")
dimensions = vts.compute_dimensions(vectors, vts.build_mst(vectors), (0.001, 2, 10))
p = clf.predict(dimensions[0].reshape(1, -1))
print("The text is probably written by a LLM." if p else "The text is probably written by a human.")
