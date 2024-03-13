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


def read_csv_file(file_path):

    with open(file_path, 'r') as file:

        reader = csv.DictReader(file)

        data = [row for row in reader]

    return data



sample_doc = read_csv_file('/home/cs242/dataset/ir_proj_data_clean.csv')

def create_index(dir):
    start_time = tm.time()
    cnt = 0
    if not os.path.exists(dir):
        os.mkdir(dir)
    store = SimpleFSDirectory(Paths.get(dir))
    analyzer = StandardAnalyzer()
    config = IndexWriterConfig(analyzer)
    config.setOpenMode(IndexWriterConfig.OpenMode.CREATE)
    writer = IndexWriter(store, config)

    metaType = FieldType()
    metaType.setStored(True)
    metaType.setTokenized(False)

    contextType = FieldType()
    contextType.setStored(True)
    contextType.setTokenized(True)
    contextType.setIndexOptions(IndexOptions.DOCS_AND_FREQS_AND_POSITIONS)

    for sample in sample_doc:
        
        idnum = sample['id']
        author_name=sample['author.name']

        screen_name=sample['author.screen_name']
        context=sample['tweet']
        time=sample['time']
        geo=sample['geo']
        tweet_link=sample['tweet_link']
        title=sample['hashtags']
        enabled=sample['enabled']

        doc = Document()
        doc.add(Field('id', str(idnum), metaType))
        doc.add(Field('Context', str(context), contextType))
        doc.add(Field('author.name', str(author_name), contextType))
        doc.add(Field('author.screen_name', str(screen_name), contextType))
        doc.add(Field('time', str(time), metaType))
        doc.add(Field('geo', str(geo), metaType))
        doc.add(Field('tweet_link', str(tweet_link), metaType))
        doc.add(Field('Title', str(title), metaType))
        doc.add(Field('enabled', str(enabled), metaType))
        writer.addDocument(doc)
        cnt = cnt+1
        if cnt == 1200000 or cnt == 1300000 or cnt == 1400000 or cnt == 1500000 or cnt == 1600000 or cnt == 1700000 or cnt == 1800000 or  cnt == 1900000 or cnt == 2000000:
            end  = tm.time() - start_time
            
    writer.close()
    end_time = tm.time()
    difference_time = end_time-start_time
    

def retrieve(storedir, query):
    searchDir = NIOFSDirectory(Paths.get(storedir))
    searcher = IndexSearcher(DirectoryReader.open(searchDir))
    
    parser = QueryParser('Context', StandardAnalyzer())
    parsed_query = parser.parse(query)

    topDocs = searcher.search(parsed_query, 10).scoreDocs
    topkdocs = []
    for hit in topDocs:
        doc = searcher.doc(hit.doc)
        topkdocs.append({
            "score": hit.score,
            "text": doc.get("Context"),
            "time": doc.get("time"),
            "geo": doc.get("geo"),
            "tweet_link":doc.get("tweet_link"),
            

        })
    
    for i in range(len(topkdocs)):
        print(topkdocs[i])

lucene.initVM(vmargs=['-Djava.awt.headless=true'])
create_index('sample_lucene_index/')
retrieve('sample_lucene_index/', sys.argv[1])

