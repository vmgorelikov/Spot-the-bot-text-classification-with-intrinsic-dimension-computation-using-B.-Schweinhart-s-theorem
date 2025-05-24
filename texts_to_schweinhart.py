import text_to_vectors as ttv
import vectors_to_schweinhart as vts
import numpy as np
import sys
import os

print("Usage: python texts_to_schweinhart.py <human corpus> <generated corpus> <dictionary>")
human_lined_corpus = open(sys.argv[1], "r", encoding="utf8")
bot_lined_corpus = open(sys.argv[2], "r", encoding="utf8")

alpha_max = 2
alpha_quantity = 10
batch_X, batch_y = np.empty((0,alpha_quantity), int), np.array([])

human_text_offsets = []
while human_lined_corpus.readline():
    human_text_offsets.append(human_lined_corpus.tell())

bot_text_offsets = []
while bot_lined_corpus.readline():
    bot_text_offsets.append(bot_lined_corpus.tell())

start_batch_number = int(sys.argv[4]) if len(sys.argv) > 4 else 0
batch_number = start_batch_number
point_number = 60 * batch_number
batch_quantity = int(sys.argv[5]) if len(sys.argv) > 5 else 999999

if point_number < len(human_text_offsets):
    human_text_offsets = human_text_offsets[point_number:]
    human_file_ended = False
else:
    bot_text_offsets = bot_text_offsets[point_number-len(human_text_offsets):]
    human_file_ended = True

batch_files = os.listdir("batches/")
dictionary = np.load(sys.argv[3], allow_pickle=True).item()

while True:
    if batch_number >= start_batch_number + batch_quantity:
        print("Finished with %i batches" % batch_number)
        break
    if "%i_%i_%i_X.pkl.npy" % (alpha_max, alpha_quantity, batch_number) in batch_files \
    and "%i_%i_%i_y.pkl.npy" % (alpha_max, alpha_quantity, batch_number) in batch_files:
        print("Duplicate file risk at batch %i. Please resolve the conflict before proceeding" % batch_number)
        quit()
    else:
        if human_file_ended:
            # print(bot_text_offsets.pop())
            offset = bot_text_offsets.pop(0)
            bot_lined_corpus.seek(offset)
            text = bot_lined_corpus.readline().strip()
            if offset < (107379490 - 3):
                category = 1  # balaboba
            elif offset < (107379490 + 66772967 - 3):
                category = 2  # mgpt
            elif offset < (107379490 + 66772967 + 125045989 - 3):
                category = 3  # lstm
            else:
                category = 4  # gpt2
        else:
            if not len(human_text_offsets):
                human_file_ended = True
                print("Human file ended (empty list)")
                continue
            human_lined_corpus.seek(human_text_offsets.pop(0))
            text = human_lined_corpus.readline()
            if not text:
                human_file_ended = True
                print("Human file ended (no text)")
                continue
            text = text.strip()
            category = 0
        if not text:
            print("One of the files ended unexpectedly")
            continue
        print("category", category)
        text = text.split(" ")
        vectors, unknown_words = ttv.tokens_to_vectors(text, dictionary, 1, 100, output_mode="sequence")
        if vectors.shape[0] < 10: continue
        dimensions = vts.compute_dimensions(vectors, vts.build_mst(vectors), (0.001, alpha_max, alpha_quantity))
        if np.any(dimensions < 0): continue
        batch_X = np.vstack((batch_X, dimensions))
        batch_y = np.append(batch_y, category)
        point_number += 1
    if point_number % 60 == 0:
        np.save("batches/%i_%i_%i_X.pkl" % (alpha_max, alpha_quantity, batch_number), batch_X)
        np.save("batches/%i_%i_%i_y.pkl" % (alpha_max, alpha_quantity, batch_number), batch_y)
        print("Batch %i ready\x07" % batch_number)
        del batch_X, batch_y
        batch_number += 1
        batch_X, batch_y =  np.empty((0,alpha_quantity), int), np.array([])





