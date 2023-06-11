from model import SentenceTransformer
import torch
from tqdm import tqdm
import os
import sys
import gc

LOCAL = True if sys.platform == 'win32' else False
model_name = "studio-ousia/luke-base"
model_save_path = f'output/bi-encoder_margin-mse_{model_name.split("/")[-1]}'
run_output_path = model_save_path + '/Run.txt'
device = 'cpu' if LOCAL else 'cuda:2'

####  Load model
model = SentenceTransformer(model_save_path, device=device, mode='entity')
print(f'{model_save_path} model loaded.')

### Data files
if LOCAL:
    data_folder = r'C:\Users\sajad\PycharmProjects\chameleon_entity_linking\msmarco'
else:
    data_folder = '/home/sajadeb/msmarco'

### Read the corpus files, that contain all the passages. Store them in the corpus dict
print('Loading collection...')
corpus = {}
corpus_filepath = os.path.join(data_folder, 'collection.tsv')
with open(corpus_filepath, 'r', encoding='utf8') as f:
    for line in tqdm(f):
        pid, passage = line.strip().split("\t")
        corpus[pid] = passage.strip()

### Read the test queries, store in queries dict
print('Loading queries...')
queries = {}
queries_filepath = os.path.join(data_folder, 'queries.dev.small.tsv')
with open(queries_filepath, 'r', encoding='utf8') as f:
    for line in f:
        qid, query = line.strip().split("\t")
        queries[qid] = query.strip()

### Read the train passages entities, store in passages_entities dict
passages_entities = {}
passages_entities_filepath = os.path.join(data_folder, 'entities', 'docs_entities.tsv')
with open(passages_entities_filepath, 'r', encoding='utf8') as fIn:
    print('Loading passages entities...')
    for line in fIn:
        pid, entities = line.strip().split("\t")
        passages_entities[pid] = eval(entities)

### Read the train queries entities, store in queries_entities dict
queries_entities = {}
queries_entities_filepath = os.path.join(data_folder, 'entities', 'dev_small_queries_entities.tsv')
with open(queries_entities_filepath, 'r', encoding='utf8') as fIn:
    print('Loading queries entities...')
    for line in fIn:
        qid, entities = line.strip().split("\t")
        queries_entities[qid] = eval(entities)

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

embedded_corpus = {}
for pid in tqdm(pids):
    passage_text = corpus[pid]
    passage_entity_spans = [(entity['start'], entity['end']) for entity in passages_entities[pid]]
    passage_entities = [entity.get('title', entity.get('spot')) for entity in passages_entities[pid]]
    embedded_corpus[pid] = torch.tensor(model.encode(passage_text, passage_entity_spans, passage_entities)).unsqueeze(0)

embedded_queries = {}
for qid in tqdm(queries):
    query_text = queries[qid]
    query_entity_spans = [(entity['start'], entity['end']) for entity in queries_entities[qid]]
    query_entities = [entity.get('title', entity.get('spot')) for entity in queries_entities[qid]]
    embedded_queries[qid] = torch.tensor(model.encode(query_text, query_entity_spans, query_entities)).unsqueeze(0)

del corpus
del queries
del passages_entities
del queries_entities
gc.collect()

# Search in a loop for the individual queries
ranks = {}
for qid, passages in tqdm(qrels.items()):
    query_embedding = embedded_queries[qid]

    scores = [float(torch.cosine_similarity(query_embedding, embedded_corpus[pid])) for pid in passages]

    # Sort the scores in decreasing order
    results = [{'pid': pid, 'score': score} for pid, score in zip(passages, scores)]
    results = sorted(results, key=lambda x: x['score'], reverse=True)

    ranks[qid] = results

print('Writing the result to file...')
with open(run_output_path, 'w', encoding='utf-8') as out:
    for qid, results in ranks.items():
        for rank, hit in enumerate(results):
            out.write(f'{qid} Q0 {hit["pid"]} {rank + 1} {hit["score"]} BiEncoder\n')
