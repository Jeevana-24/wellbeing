# Search Engine for Lifestyle Tweets

## Overview of the System

### 1. Architecture

This system is a search engine designed to retrieve Lifestyle tweets extracted from Twitter. It compares the search results generated from Lucene and Faiss Indexing. The system provides an interactive user interface allowing users to enter their queries and preview relevant tweets from both Lucene and Faiss indexes.

### 2. Details of How BERT Was Used

To accurately retrieve tweets from crawled data from Twitter, BERT (Bidirectional Encoder Representations from Transformers) was utilized as the major tool for developing the system. The following steps were taken:

- **Preparing the Data:** The tweets retrieved from the crawling system were cleaned, tokenized, and converted into a BERT-compatible format. This step is critical as it ensures that the input data is in a standardized format that the model can process.

- **Fine-Tuning BERT:** BERT was then fine-tuned on tweets to construct a language model that could understand the context and meaning of the given tweet. A pre-trained BERT model is fine-tuned by training it on a specific task or domain-specific data.

- **Indexing:** After creating the language model, the next step was to generate an index of the tweets. An index is a type of data structure that allows for the quick and easy retrieval of items based on specific keywords or phrases. The tweets were indexed using BERT embeddings.

- **Query Processing:** The final step was to process user queries and retrieve relevant tweets. When a user enters a query, BERT encodes it into a vector representation. A similarity metric, such as cosine similarity, is then used to compare this vector representation to the indexed tweets. Tweets with the highest similarity score are then returned to the user.

## Getting Started

### BERT:
We have split the BERT code into two parts where the first part of the code `bert_faiss.py` creates BERT embeddings and creates the Faiss indexing for the tweets which are stored in the `sample_code.index` so as to avoid running every time to create indices for our static data which takes a large amount of time to run.
To run this file: `python3 bert_faiss.py`

The second part of the code `bert_query.py` reads in the `sample_code.index` and retrieves the top 10 results from FAISS indexer for the respective query. The query is passed as the argument.
To run this file: `python3 bert_query.py "query"`.
Note: Since BERT Indexing was taking a lot of time to index, we only used 2 MB of the whole scrapped Twitter data (which mainly contain tweets related to nature). Hence the BERT search engine works properly only for queries related to nature.

### Webpage:
We have created a simple webpage for our search engine and used Django, HTML, CSS, and Python for it. 
We created a folder called `mysite_final` and it has all the required documents.
The main files related to PyLucene and BERT are `Indextest.py`, `bert_faiss.py`, and `bert_query.py`.
To run:
- Navigate to `mysite_final` folder in the terminal.
- Run the command: `python3 manage.py runserver 0.0.0.0:8888`

Once the above command is executed, go to `http://class-046.cs.ucr.edu:8888` to view the webpage.
## Screenshots of webpage
![Screenshot](screenshots/Screenshot%202024-03-13%20121503.png)
![Screenshot 1](screenshots/Screenshot%202024-03-13%20121537.png)
![Screenshot 2](screenshots/Screenshot%202024-03-13%20121559.png)


