from wikipedia2vec import Wikipedia2Vec
from tqdm import tqdm
import os
import sys
import numpy as np

LOCAL = True if sys.platform == 'win32' else False
run_output_path = 'Run.txt'

####  Load model
model = Wikipedia2Vec.load('../resources/enwiki_20180420_100d.pkl')
# print(f'{model_save_path} model loaded.')

### Data files
if LOCAL:
    data_folder = r'C:\Users\sajad\PycharmProjects\chameleon_entity_linking\msmarco'
else:
    data_folder = '/home/sajadeb/msmarco'

### Read the train passages entities, store in passages_entities dict
passages_entities = {}
passages_entities_filepath = os.path.join(data_folder, 'entities', 'docs_entities_0.1.tsv')
with open(passages_entities_filepath, 'r', encoding='utf8') as fIn:
    print('Loading passages entities...')
    for line in fIn:
        pid, entities = line.strip().split("\t")
        passages_entities[pid] = eval(entities)

### Read the train queries entities, store in queries_entities dict
queries_entities = {}
queries_entities_filepath = os.path.join(data_folder, 'entities', 'dev_small_queries_entities_0.1.tsv')
with open(queries_entities_filepath, 'r', encoding='utf8') as fIn:
    print('Loading queries entities...')
    for line in fIn:
        qid, entities = line.strip().split("\t")
        queries_entities[qid] = eval(entities)

print('Loading qrels...')
qrels = {}
pids = set()
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


def get_pooled_entity_embedding(entities):
    entity_embeddings = []
    for entity in entities:
        try:
            entity_embeddings.append(model.get_entity_vector(entity['title']))
        except KeyError:
            pass

    if len(entity_embeddings) == 0:
        return np.zeros(100, dtype='float32')
    else:
        return np.mean(np.array(entity_embeddings), axis=0)


def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


# Search in a loop for the individual queries
ranks = {}
for qid, passages in tqdm(qrels.items()):
    query_entity_embedding = get_pooled_entity_embedding(queries_entities[qid])

    scores = [cosine_similarity(query_entity_embedding, get_pooled_entity_embedding(passages_entities[pid])) for pid in passages]

    # Sort the scores in decreasing order
    results = [{'pid': pid, 'score': score} for pid, score in zip(passages, scores)]
    results = sorted(results, key=lambda x: x['score'], reverse=True)

    ranks[qid] = results

print('Writing the result to file...')
with open(run_output_path, 'w', encoding='utf-8') as out:
    for qid, results in ranks.items():
        for rank, hit in enumerate(results):
            out.write(f'{qid} Q0 {hit["pid"]} {rank + 1} {hit["score"]} BiEncoder_W2V\n')