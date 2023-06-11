import sys
import json
from torch.utils.data import DataLoader
from sentence_transformers import models
from model import SentenceTransformer, MSMARCODataset, MarginMSELossOnTextAndEntity
import gzip
import os
import tqdm
import pickle
from transformers import logging
import gc

logging.set_verbosity_error()

# The  model we want to fine-tune
LOCAL = True if sys.platform == 'win32' else False
model_name = 'studio-ousia/luke-base'
model_save_path = f'output/bi-encoder_margin-mse_{model_name.split("/")[-1]}'
train_batch_size = 4 if LOCAL else 32
device = 'cpu' if LOCAL else 'cuda:0'
max_passages = 2e6
max_seq_length = 512
num_negs_per_system = 4
num_epochs = 3
warmup_steps = 1000
use_pre_trained_model = False

os.makedirs(model_save_path, exist_ok=True)

# Load our embedding model
if use_pre_trained_model:
    print("use pretrained SBERT model")
    model = SentenceTransformer(model_name, device=device)
    model.max_seq_length = max_seq_length
else:
    print("Create new SBERT model")
    word_embedding_model = models.Transformer(model_name, max_seq_length=max_seq_length)
    entity_embedding_model = models.Transformer(model_name, max_seq_length=max_seq_length)
    pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(), "mean")
    model = SentenceTransformer(modules=[word_embedding_model, entity_embedding_model, pooling_model], device=device, mode='text')

### Now we read the MS Marco dataset
if LOCAL:
    data_folder = r'C:\Users\sajad\PycharmProjects\chameleon_entity_linking\msmarco'
else:
    data_folder = '/home/sajadeb/msmarco'

#### Read the corpus files, that contain all the passages. Store them in the corpus dict
corpus = {}
collection_filepath = os.path.join(data_folder, 'collection.tsv')
with open(collection_filepath, 'r', encoding='utf8') as f:
    print('Loading collection...')
    for line in f:
        pid, passage = line.strip().split("\t")
        corpus[int(pid)] = passage

### Read the train queries, store in queries dict
queries = {}
queries_filepath = os.path.join(data_folder, 'queries.train.tsv')
with open(queries_filepath, 'r', encoding='utf8') as fIn:
    print('Loading queries...')
    for line in fIn:
        qid, query = line.strip().split("\t")
        queries[int(qid)] = query

### Read the train passages entities, store in passages_entities dict
passages_entities = {}
passages_entities_filepath = os.path.join(data_folder, 'entities', 'docs_entities.tsv')
with open(passages_entities_filepath, 'r', encoding='utf8') as fIn:
    print('Loading passages entities...')
    for line in fIn:
        pid, entities = line.strip().split("\t")
        passages_entities[int(pid)] = eval(entities)

### Read the train queries entities, store in queries_entities dict
queries_entities = {}
queries_entities_filepath = os.path.join(data_folder, 'entities', 'queries_entities.tsv')
with open(queries_entities_filepath, 'r', encoding='utf8') as fIn:
    print('Loading queries entities...')
    for line in fIn:
        qid, entities = line.strip().split("\t")
        queries_entities[int(qid)] = eval(entities)

# Load a dict (qid, pid) -> ce_score that maps query-ids (qid) and paragraph-ids (pid)
# to the CrossEncoder score computed by the cross-encoder/ms-marco-MiniLM-L-6-v2 model
ce_scores_file = os.path.join(data_folder, 'cross-encoder-ms-marco-MiniLM-L-6-v2-scores.pkl.gz')
print("Load CrossEncoder scores dict")
with gzip.open(ce_scores_file, 'rb') as fIn:
    ce_scores = pickle.load(fIn)

# As training data we use hard-negatives that have been mined using various systems
hard_negatives_filepath = os.path.join(data_folder, 'msmarco-hard-negatives.jsonl.gz')
print("Read hard negatives train file")
train_queries = {}
negs_to_use = ['msmarco-distilbert-base-v3']
with gzip.open(hard_negatives_filepath, 'rt') as fIn:
    for line in tqdm.tqdm(fIn):
        if 0 < max_passages <= len(train_queries):
            break
        data = eval(line) if LOCAL else json.loads(line)

        # Get the positive passage ids
        pos_pids = data['pos']

        # Get the hard negatives
        neg_pids = set()
        if negs_to_use is None:
            negs_to_use = list(data['neg'].keys())
            print("Using negatives from the following systems:", negs_to_use)

        for system_name in negs_to_use:
            if system_name not in data['neg']:
                continue

            system_negs = data['neg'][system_name]
            negs_added = 0
            for pid in system_negs:
                if pid not in neg_pids:
                    neg_pids.add(pid)
                    negs_added += 1
                    if negs_added >= num_negs_per_system:
                        break

        if len(pos_pids) > 0 and len(neg_pids) > 0:
            train_queries[data['qid']] = {'qid': data['qid'], 'query': queries[data['qid']], 'pos': pos_pids, 'neg': neg_pids}

print("Train queries: {}".format(len(train_queries)))


# For training the SentenceTransformer model, we need a dataset, a dataloader, and a loss used for training.
train_dataset = MSMARCODataset(queries=train_queries, corpus=corpus, ce_scores=ce_scores, queries_entities=queries_entities, passages_entities=passages_entities)
train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=train_batch_size, drop_last=True)
train_loss = MarginMSELossOnTextAndEntity(model=model)

del corpus
del queries
del train_queries
del train_dataset
del queries_entities
del passages_entities
del ce_scores
gc.collect()

# Train the model
model.fit(train_objectives=[(train_dataloader, train_loss)],
          epochs=num_epochs,
          warmup_steps=warmup_steps,
          use_amp=True,
          optimizer_params={'lr': 2e-5},
          )

# Train latest model
model.save(model_save_path)
