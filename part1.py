from sentence_transformers import SentenceTransformer, util
import numpy as np
import pickle
import os

model = SentenceTransformer('stsb-roberta-large')

sentence1 = "I like Python because I can build AI applications"
sentence2 = "I like Python because I can do data analytics"

embedding1 = model.encode(sentence1, convert_to_tensor=True)
embedding2 = model.encode(sentence2, convert_to_tensor=True)

cosine_scores = util.pytorch_cos_sim(embedding1, embedding2)
print("Sentence 1:", sentence1)
print("Sentence 2:", sentence2)
print("Similarity score:", cosine_scores.item())

filename = os.path.join('perline_model.sav')
pickle.dump(model , open(filename, 'wb'))

