# BAMBARA PROCESSOR
# Usage: python bambara_processor.py <Raw input directory> <Output file> [Documents output directory]
# The input texts are supposed to be in lines of reasonable length, with no line-breaks which would break sentences apart.
# Vladimir Gorelikov, HSE University <vmgorelikov@edu.hse.ru>
# May 2025

import os
import sys
import re
import unicodedata
import time

bypass_lemmatization = False # for lemmatized texts

if len(sys.argv) < 3:
    print("Usage: python bambara_processor.py <Raw input directory> <Output file> [Documents output directory]")
    print("The input texts are supposed to be in lines of reasonable length, with no line-breaks which would break sentences apart.")
    exit(1)

sentence_boundary_punctuation = [". ", "? ", "! ", ": "]
orphographic_nonletters = ["'", "-"]

suppletive_forms_byclass = {"fɔ": ["fo"], "ninlu": ["ninnu"], "minlu": ["minnu"], "kelen": ["kelen"]}
suppletive_forms = {w: lemma for lemma, ws in suppletive_forms_byclass.items() for w in ws}

affixes = ["([aiuoeɔɛ])la#", "([aiuoeɔɛ])nna#", "(.{2,})lu#", "([aiuoeɔɛ])nnu#",
"([rl][aiuoeɔɛ])la#", "(n[aiuoeɔɛ]|[aiuoeɔɛ]n)na#", "(.{2,})w#",
"([aiuoeɔɛ])len#", "([aiuoeɔɛ]n)nen#"] # regex, groups are what should be preserved
#, "(.{2,})ra#", "(.{2,})la#"

stopwords_byclass = {
    "XNUMERAL": ["kelen", "fila", "saba", "naani", "tan", "bila", "kɛmɛ",
    "duuru", "bi", "nan", "wɔɔrɔ", "b'", "wolonwula", "mugan",
    "kelenkelen", "segin", "seegin", "kɔnɔntɔn", "wolonfila", "biwɔɔrɔ"],
    "XPRONOUN": ["a", "u", "i", "an", "n", "olu", "ne", "aw", "ale", "e",
    "anw", "a'", "o", "min", "ɲɔgɔn", "nin", "bɛɛ", "minlu", "dɔ", "di", "mun", "jumɛn", "ninlu"]
}

stopwords = {w: tag for tag, ws in stopwords_byclass.items() for w in ws}

# tokenizer
def tokenizer(line: str,
sentence_boundary_punctuation: list=sentence_boundary_punctuation) -> list[list[str]]:
    '''Splits a string into a list of sentences (i.e. lists of tokens).'''

    # split into sentences by punctuation
    sentences = re.split("|".join([re.escape(mark) for mark in sentence_boundary_punctuation]), line)

    for i, sentence in enumerate(sentences):
        # remove non-orphographic characters (mostly non-letters),
        # split into tokens by spaces and convert to lowercase
        sentence = ''.join([char.lower() for char in sentence
        if (unicodedata.category(char).startswith("L") or
        char in orphographic_nonletters or char == " ")]).split(" ")
        sentences[i] = sentence
    return sentences

# lemmatizer
def lemmatizer(token: str, suppletive_forms: dict = suppletive_forms,
    affixes: dict = affixes) -> str:
    '''Removes inflection marking and returns a lemma for a given token.'''
    if bypass_lemmatization:
        return token
    if token in suppletive_forms:
        return suppletive_forms[token]
    else:
        token += "#" # word boundary for regexes
        for pattern in affixes:
            match = re.search(pattern, token)
            if match:
                if len(match.groups()) == 1:
                    token = token.replace(match.group(0), match.group(1), 1)
                else:
                    print("Matching error while analyzing %s: %i groups found"%(token,len(match.groups())))
        token = token.strip("#") # if nothing is found, truncate the #
        return token

# semantic filter
def semantic_filter(token: str, stopwords: dict = stopwords) -> str:
    '''Returns a given token if it is not a stop-word or its replacement if it is.'''
    if token in stopwords:
        return stopwords[token]
    else:
        return token

if not os.path.isdir(sys.argv[-1]):
    big_file = open(sys.argv[-1], "w", encoding="utf8")
    outdir, _ = os.path.split(os.path.abspath(sys.argv[-1]))
else:
    big_file = open(sys.argv[-2], "w", encoding="utf8")
    outdir = sys.argv[-1]
filelist = [f for f in os.listdir(sys.argv[1])]
total_documents_size = sum([os.path.getsize(sys.argv[1] + f) for f in filelist])
print("%i files found with total size of %i bytes" % (len(filelist), total_documents_size))
progress_5pc = len(filelist)//20 if len(filelist) > 19 else 1
last_5pc_point = -1
progress = 0
progress_bytes = 0
token_number = 0
start_time = time.time()
# input and output
for filename in filelist:
    input_file = open(sys.argv[1]+filename, "r", encoding="utf8")
    print(sys.argv[1]+filename)
    print(os.path.basename(filename))
    doc_file = open(outdir + os.path.basename(filename) + "_PROC.txt", "w", encoding="utf8")
    while True:
        line = input_file.readline()
        if not line:
            break
        line = line.strip()
        for sentence in tokenizer(line):
            # the pipeline
            sentence = \
            [semantic_filter(lemmatizer(token)) for token in sentence]
            sentence = [x for x in sentence if x] # remove empty lemmata
            token_number += len(sentence)
            sentence = " ".join(sentence)
            big_file.write(sentence + " ")
            doc_file.write(sentence + "\n")
    big_file.write("\n")
    progress += 1
    progress_bytes += input_file.tell()
    doc_file.close()
    if progress // progress_5pc != last_5pc_point:
        last_5pc_point = progress // progress_5pc
        print("Processed %i files, wrote %i tokens, approx. %.2f minutes left."
        % (progress, token_number,
        (total_documents_size-progress_bytes)/(progress_bytes*60/(time.time()-start_time))))

big_file.close()