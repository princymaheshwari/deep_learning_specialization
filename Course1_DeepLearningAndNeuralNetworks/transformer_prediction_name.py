# I will use hugging face transformer library to make a model that predicts my name
# This will not predict my actual name but a name that has the highest probability of showing up in it's system
# for twist I will make it predict and give an output of all top names with the highest probabilites of showing up in it's system.

from transformers import pipeline

nlp = pipeline("fill-mask", model = "bert-base-uncased")
result = nlp("My age is [MASK].") # by default bert-base-uncased prints the top 5 tokens with the highest probability

for r in result:
    print(r)



# Now let's create a similar program where we can change the number of tokens or high probability outputs 
# that are being printed using the top_k= argument while making the pipeline

nlp2 = pipeline("fill-mask", model = "bert-base-uncased", top_k=8)
result2 = nlp2("My age is [MASK].")

for r2 in result2:
    print(r2)