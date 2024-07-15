from transformers import BertTokenizer
import torch 

tokenizer = BertTokenizer.from_pretrained("google-bert/bert-base-uncased")
text = tokenizer.tokenize("I don't like cat but I love dog.")
print(text)
sent_vec = tokenizer(text, return_tensors='pt')
print(sent_vec)
print(tokenizer.decode(sent_vec))