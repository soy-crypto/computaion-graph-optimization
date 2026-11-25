from transformers import AutoTokenizer

model_name = "google/bert_uncased_L-2_H-128_A-2"
tokenier = AutoTokenizer.from_pretrained(model_name)
print(tokenier.tokenize("I like dogs!"))
