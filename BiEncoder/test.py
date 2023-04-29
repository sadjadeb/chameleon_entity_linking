from sentence_transformers import SentenceTransformer
import torch
from tqdm import tqdm
import os
import sys
import pickle

LOCAL = True if sys.platform == 'win32' else False
model_name = "studio-ousia/luke-base"
model_save_path = f'output/bi-encoder_margin-mse_{model_name.split("/")[-1]}'
run_output_path = model_save_path + '/Run.txt'
device = 'cpu' if LOCAL else 'cuda:2'

####  Load model
model = SentenceTransformer(model_save_path, device=device)
print(f'{model_save_path} model loaded.')

### Data files
if LOCAL:
    data_folder = r'C:\Users\sajad\PycharmProjects\chameleon_entity_linking\msmarco'
else:
    data_folder = '/home/sajadeb/msmarco'

### Read the corpus files, that contain all the passages. Store them in the corpus dict
print('Loading collection...')
collection = {}
collection_filepath = os.path.join(data_folder, 'collection.tsv')
with open(collection_filepath, 'r', encoding='utf8') as f:
    for line in tqdm(f):
        pid, passage = line.strip().split("\t")
        collection[pid] = passage.strip()

### Read the test queries, store in queries dict
print('Loading queries...')
queries = {}
queries_filepath = os.path.join(data_folder, 'queries.dev.small.tsv')
with open(queries_filepath, 'r', encoding='utf8') as f:
    for line in f:
        qid, query = line.strip().split("\t")
        queries[qid] = torch.tensor(model.encode(query.strip())).unsqueeze(0)

print('Loading qrels...')
qrels = {}
pids = set()
if LOCAL:
    qrels_filepath = os.path.join(data_folder, 'runbm25anserini.dev')
else:
    qrels_filepath = os.path.join(data_folder, 'runbm25anserini_notnull.dev')
with open(qrels_filepath, 'r', encoding='utf8') as f:
    for line in f:
        qrel = line.strip().split(" ")
        qid = qrel[0]
        pid = qrel[2]
        pids.add(pid)
        if qid in qrels:
            qrels[qid].append(pid)
        else:
            qrels[qid] = [pid]

embedded_collection = {}
for pid in tqdm(pids):
    embedded_collection[pid] = torch.tensor(model.encode(collection[pid])).unsqueeze(0)

# Search in a loop for the individual queries
ranks = {}
for qid, passages in tqdm(qrels.items()):
    query_embedding = queries[qid]

    scores = [float(torch.cosine_similarity(query_embedding, embedded_collection[pid])) for pid in passages]

    # Sort the scores in decreasing order
    results = [{'pid': pid, 'score': score} for pid, score in zip(passages, scores)]
    results = sorted(results, key=lambda x: x['score'], reverse=True)

    ranks[qid] = results

print('Writing the result to file...')
with open(run_output_path, 'w', encoding='utf-8') as out:
    for qid, results in ranks.items():
        for rank, hit in enumerate(results):
            out.write(f'{qid} Q0 {hit["pid"]} {rank + 1} {hit["score"]} BiEncoder\n')
