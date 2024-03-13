import logging, sys
import csv
logging.disable(sys.maxsize)

import time
import lucene
import os
from transformers import AutoTokenizer, AutoModel
import torch
import faiss


def read_csv_file(file_path):
    with open(file_path, 'r') as file:
        reader = csv.DictReader(file)
        data = [row for row in reader]
    return data

sample_doc = read_csv_file('/home/cs242/dataset/ir_proj_data_clean.csv')
def convert_to_embedding(query):
        tokens = {'input_ids': [], 'attention_mask': []}
        new_tokens = tokenizer.encode_plus(query, max_length=512,
                                                           truncation=True, padding='max_length',return_tensors='pt')
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

tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-distilroberta-v1') 
model = AutoModel.from_pretrained('sentence-transformers/all-distilroberta-v1')
index = faiss.IndexFlatIP(768)   # build the index  
sentences = []
for p in range(len(sample_doc)):
    sentences.append(sample_doc[p]['tweet'])
# initialize dictionary to store tokenized sentences
tokens = {'input_ids': [], 'attention_mask': []}
time_lis=[]
count = 0
start = time.time()
for sentence in sentences:
    count +=1
    new_tokens = tokenizer.encode_plus(sentence, max_length=512,truncation=True, padding='max_length',return_tensors='pt')
    tokens['input_ids'].append(new_tokens['input_ids'][0])
    tokens['attention_mask'].append(new_tokens['attention_mask'][0])
    if count > 50:
        count = 0
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
            index.add(mean_pooled) 
            val_time = time.time()-start
            print("The time to index" , index.ntotal , "is :",val_time)
            time_lis.append([index.ntotal,val_time])
            tokens = {'input_ids': [], 'attention_mask': []}


faiss.write_index(index,"/home/cs242/dataset/sample_code_full.index")

