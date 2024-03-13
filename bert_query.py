
import logging, sys
import csv
logging.disable(sys.maxsize)
import faiss
import torch
from transformers import AutoTokenizer, AutoModel

tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-distilroberta-v1') 
model = AutoModel.from_pretrained('sentence-transformers/all-distilroberta-v1')

def read_csv_file(file_path):
    with open(file_path, 'r') as file:
        reader = csv.DictReader(file)
        data = [row for row in reader]
    return data

sample_doc = read_csv_file('/home/cs242/dataset/natureLover_clean.csv')
sentences = []
for p in range(200):
    sentences.append(sample_doc[p]['tweet'])
def convert_to_embedding(query):
    tokens = {'input_ids': [], 'attention_mask': []}
    new_tokens = tokenizer.encode_plus(query, max_length=512,truncation=True, padding='max_length',return_tensors='pt')
    tokens['input_ids'].append(new_tokens['input_ids'][0])
    tokens['attention_mask'].append(new_tokens['attention_mask'][0])
    tokens['input_ids'] = torch.stack(tokens['input_ids'])
    tokens['attention_mask'] = torch.stack(tokens['attention_mask'])
    with torch.no_grad():
        outputs = model(**tokens)
        embeddings = outputs.last_hidden_state
        attention_mask = tokens['attention_mask']
        mask = attention_mask.unsqueeze(-1).expand(embeddings.size()).float()
        masked_embeddings = embeddings * mask
        summed = torch.sum(masked_embeddings, 1)
        summed_mask = torch.clamp(mask.sum(1), min=1e-9)
        mean_pooled = summed / summed_mask
                                                                            
    return mean_pooled[0] 


query=sys.argv[1]
query_embedding = convert_to_embedding(query)
index_loaded = faiss.read_index("/home/cs242/dataset/sample_code.index")
D, I = index_loaded.search(query_embedding[None, :], 5)
x = list(I[0])
sentences = list(sentences)
for i in range(len(x)):
    print("scores: ", D[0][i])
    print(sentences[x[i]])

