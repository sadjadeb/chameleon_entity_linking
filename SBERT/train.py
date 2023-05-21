from torch.utils.data import DataLoader
from sentence_transformers import LoggingHandler
from model import CrossEncoder, InputExample
import logging
import os
import random
import gc
import sys

#### Just some code to print debug information to stdout
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])

LOCAL = True if sys.platform == 'win32' else False
# First, we define the transformer model we want to fine-tune
# model_name = 'distilroberta-base'
model_name = "studio-ousia/luke-base"
train_batch_size = 32
num_epochs = 1
warmup_steps = 5000
device = 'cpu' if LOCAL else 'cuda:1'
# Maximal number of training samples we want to use
max_train_samples = 2e6
# We use a positive-to-negative ratio: For 1 positive sample (label 1) we include 4 negative samples (label 0)
pos_neg_ration = 4

model_save_path = f'output/cross-encoder_{model_name.split("/")[-1]}_with-entities-entities'

# We set num_labels=1, which predicts a continuous score between 0 and 1
model = CrossEncoder(model_name, num_labels=1, max_length=512, device=device)

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
        corpus[pid] = passage

### Read the train queries, store in queries dict
queries = {}
queries_filepath = os.path.join(data_folder, 'queries.train.tsv')
with open(queries_filepath, 'r', encoding='utf8') as fIn:
    print('Loading queries...')
    for line in fIn:
        qid, query = line.strip().split("\t")
        queries[qid] = query

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
queries_entities_filepath = os.path.join(data_folder, 'entities', 'queries_entities.tsv')
with open(queries_entities_filepath, 'r', encoding='utf8') as fIn:
    print('Loading queries entities...')
    for line in fIn:
        qid, entities = line.strip().split("\t")
        queries_entities[qid] = eval(entities)

# Read our training file
cnt = 0
train_samples = []
if LOCAL:
    train_filepath = os.path.join(data_folder, 'qidpidtriples.train.full.2.tsv')
else:
    train_filepath = os.path.join(data_folder, 'qidpidtriples.train.full.2.notnull.tsv')
with open(train_filepath, 'r', encoding='utf8') as fIn:
    print('Loading triples...')
    lines = fIn.readlines()
    random.shuffle(lines)

    for line in lines:
        qid, pos_id, neg_id = line.strip().split()

        query = queries[qid]
        query_entity_spans = [(entity['start'], entity['end']) for entity in queries_entities[qid]]
        query_entities = [entity.get('title', 'spot') for entity in queries_entities[qid]]

        if (cnt % (pos_neg_ration + 1)) == 0:
            passage = corpus[pos_id]
            label = 1
            passage_entity_spans = [(entity['start'], entity['end']) for entity in passages_entities[pos_id]]
            passage_entities = [entity.get('title', 'spot') for entity in passages_entities[pos_id]]
        else:
            passage = corpus[neg_id]
            label = 0
            passage_entity_spans = [(entity['start'], entity['end']) for entity in passages_entities[neg_id]]
            passage_entities = [entity.get('title', 'spot') for entity in passages_entities[neg_id]]

        train_samples.append(InputExample(texts=[query, passage], label=label,
                                          entity_spans=[query_entity_spans, passage_entity_spans],
                                          entities=[query_entities, passage_entities]))
        cnt += 1

        if cnt >= max_train_samples:
            break

del corpus
del queries
del passages_entities
del queries_entities
gc.collect()

# We create a DataLoader to load our train samples
train_dataloader = DataLoader(train_samples, shuffle=True, batch_size=train_batch_size)

del train_samples

# Train the model
model.fit(train_dataloader=train_dataloader,
          epochs=num_epochs,
          warmup_steps=warmup_steps,
          use_amp=True)

# Save latest model
model.save(model_save_path)
