from django.shortcuts import render
from django.http import HttpResponse
import subprocess
import logging, sys
import csv
logging.disable(sys.maxsize)
import faiss
import torch
from transformers import AutoTokenizer, AutoModel
import json
import logging, sys
import csv
logging.disable(sys.maxsize)

import time as tm
import lucene
import os
from org.apache.lucene.store import MMapDirectory, SimpleFSDirectory, NIOFSDirectory
from java.nio.file import Paths
from org.apache.lucene.analysis.standard import StandardAnalyzer
from org.apache.lucene.document import Document, Field, FieldType
from org.apache.lucene.queryparser.classic import QueryParser
from org.apache.lucene.index import FieldInfo, IndexWriter, IndexWriterConfig, IndexOptions, DirectoryReader
from org.apache.lucene.search import IndexSearcher, BoostQuery, Query
from org.apache.lucene.search.similarities import BM25Similarity

# Create your views here.
def results(request):
    if request.method == 'POST':
        data = request.POST.get('name')
        print(data)
        context= {}
        system=request.POST.get('req_method', None )
        param = {}
        #context['req_method']=system
        if system=="BERT":
            result_j=create_json(data)
            json_records = json.dumps(result_j)
            data = [] 
            data = json.loads(json_records) 
            param = {'d': data} 
        if system=="PyLucene":
            param=pylucene_retrieval(data)
        param['req_method'] = system
        print(param) 
        return render(request,'results.html',param)
    else:
        return render(request, 'results.html')
def index(request):
    data = request.POST
    print(data)
    return render(request, '../templates/index.html')

tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-distilroberta-v1')
model = AutoModel.from_pretrained('sentence-transformers/all-distilroberta-v1')

def pylucene_retrieval( query):
    lucene.initVM()
    searchDir = NIOFSDirectory(Paths.get('sample_lucene_index/'))
    searcher = IndexSearcher(DirectoryReader.open(searchDir))
    
    parser = QueryParser('Context', StandardAnalyzer())
    parsed_query = parser.parse(query)
    print('Inside PyLucene Logic')

    topDocs = searcher.search(parsed_query, 10).scoreDocs
    topkdocs = []
    for hit in topDocs:
        doc = searcher.doc(hit.doc)
        
        topkdocs.append({
            "tweet": doc.get("Context"),
            "score": hit.score
            })
    return {'d': topkdocs}
  
        
def read_csv_file(file_path):
    with open(file_path, 'r',encoding="utf8") as file:
        reader = csv.DictReader(file)
        data = [row for row in reader]
    return data

def read_doc():
    sample_doc = read_csv_file('natureLover_clean.csv')
    sentences = []
    for p in range(len(sample_doc)):
        sentences.append(sample_doc[p]['tweet'])
    return sentences

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

def create_json(query):
    query_embedding = convert_to_embedding(query)
    index_loaded = faiss.read_index("sample_code.index")
    D, I = index_loaded.search(query_embedding[None, :], 10)
    x = list(I[0])
    res=[]
    sentences=read_doc()
    sentences = list(sentences)
    for i in range(len(x)):
        print("scores: ", D[0][i])
        print(sentences[x[i]])
        res.append(sentences[x[i]])
    return res
