# We are using a pre trained BERT (a transformer which is an architecture of deep learning) model to predict a word
# We will be using the Hugging face transformer Library for it

from transformers import pipeline

nlp = pipeline("fill-mask", model = "bert-base-uncased")
result = nlp("The capital of Italy is [MASK].")

print(result[0])