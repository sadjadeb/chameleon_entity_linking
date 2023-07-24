import faiss
from transformers import AutoModel, AutoTokenizer
from model_chatgpt import BiEncoder
import os
import torch
import sys
from tqdm import tqdm, trange
import time
import ir_datasets, ir_measures
from ir_measures import *
import numpy as np


LOCAL = True if sys.platform == 'win32' else False
model_name = "bert-base-uncased"
model_save_path = f'output/bi-encoder_cosine-embedding_{model_name.split("/")[-1]}'
run_output_path = model_save_path + '/Run.txt'
device = 'cpu' if LOCAL else 'cuda:1'
batch_size = 4 if LOCAL else 128

# We use the Bi-Encoder to encode all passages, so that we can use it with sematic search
# Load the pre-trained BERT model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)
encoder = AutoModel.from_pretrained(model_name).to(device)
embedding_dim = encoder.config.hidden_size

# Load the trained BiEncoder model
model = BiEncoder(encoder).to(device)
model.load_state_dict(torch.load(model_save_path + '/BiEncoder_ChatGPT.pt'))
model.eval()
print(f'{model_save_path} model loaded.')

### Data files
if LOCAL:
    data_folder = r'C:\Users\sajad\PycharmProjects\chameleon_entity_linking\msmarco'
else:
    data_folder = '/home/sajadeb/msmarco'

### Read the corpus files, that contain all the passages. Store them in the corpus dict
print('Loading collection...')
# corpus = {}
pids = []
corpus = []
corpus_filepath = os.path.join(data_folder, 'collection.tsv')
with open(corpus_filepath, 'r', encoding='utf8') as f:
    for line in tqdm(f):
        pid, passage = line.strip().split("\t")
        # corpus[pid] = passage.strip()
        pids.append(pid)
        corpus.append(passage.strip())

all_embeddings = []
# check whether faiss index exists
if os.path.exists(os.path.join(model_save_path, 'faiss.index')):
    index = faiss.read_index(os.path.join(model_save_path, 'faiss.index'))
    print(f'Index loaded. Index size: {index.ntotal}')
else:
    for start_index in trange(0, len(corpus), batch_size, desc="Batches"):
        sentences_batch = corpus[start_index:start_index + batch_size]
        features = tokenizer(sentences_batch, return_tensors='pt', padding=True, truncation=True, max_length=512)

        with torch.no_grad():
            out_features = model(features['input_ids'].to(device), features['attention_mask'].to(device)).detach()

        # torch.save(out_features, os.path.join(model_save_path, f'corpus_tensor_{start_index}.pt'))
        all_embeddings.append(out_features.detach().cpu().numpy())
    all_embeddings = np.concatenate(all_embeddings, axis=0)

    index = faiss.IndexFlatL2(embedding_dim)

    # for i in trange(0, len(corpus), batch_size, desc='Creating index'):
    #     all_corpus = torch.load(os.path.join(model_save_path, f'corpus_tensor_{i}.pt'), map_location=torch.device(device)).detach().cpu().numpy()
    #     index.add(all_corpus)
    index.add(all_embeddings)

    faiss.write_index(index, os.path.join(model_save_path, 'faiss.index'))
    print(f'Index saved. Index size: {index.ntotal}')

queries_filepath = os.path.join(data_folder, 'queries.dev.small.tsv')
qids = []
queries = []
with open(queries_filepath, 'r', encoding='utf-8') as fIn:
    for line in fIn:
        qid, query = line.strip().split("\t")
        qids.append(qid)
        queries.append(query.strip())

top_k = 1000
print('Encoding queries...')
query_embeddings = []
for start_index in trange(0, len(queries), batch_size, desc="Batches"):
    sentences_batch = queries[start_index:start_index + batch_size]
    features = tokenizer(sentences_batch, return_tensors='pt', padding=True, truncation=True, max_length=512)

    with torch.no_grad():
        out_features = model(features['input_ids'].to(device), features['attention_mask'].to(device)).detach()

    query_embeddings.append(out_features.detach().cpu().numpy())
query_embeddings = np.concatenate(query_embeddings, axis=0)

start_time = time.time()
D, I = index.search(query_embeddings, top_k)
print(f'Search time: {time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time))}')

print('Writing the result to file...')
with open(run_output_path, 'w', encoding='utf-8') as fOut:
    for qid in range(len(I)):
        for rank in range(10):
            fOut.write(f'{qids[qid]} Q0 {pids[I[qid][rank]]} {rank+1} {1/(rank+1):.7f} BiEncoder_Retrieval\n')

print('Evaluation...')
qrels = ir_datasets.load('msmarco-passage/dev/small').qrels_iter()
run = ir_measures.read_trec_run(run_output_path)
print(ir_measures.calc_aggregate([nDCG@10, P@10, AP@10, RR@10, R@10], qrels, run))
