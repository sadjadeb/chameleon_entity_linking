from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup
from model import FullyCrossEncoder, InputExample
import torch
from torch.cuda.amp import autocast
import os
import random
import gc
import sys
from tqdm import tqdm

LOCAL = True if sys.platform == 'win32' else False
# First, we define the transformer model we want to fine-tune
model_name = "studio-ousia/luke-base"
train_batch_size = 4 if LOCAL else 32
device = 'cpu' if LOCAL else 'cuda:0'
# Maximal number of training samples we want to use
max_train_samples = 2e6
# We use a positive-to-negative ratio: For 1 positive sample (label 1) we include 4 negative samples (label 0)
pos_neg_ration = 4

model_save_path = f'output/fully-cross-encoder_{model_name.split("/")[-1]}_with-entities-entities'
os.makedirs(model_save_path, exist_ok=True)

# We set num_labels=1, which predicts a continuous score between 0 and 1
model = FullyCrossEncoder(model_name, max_length=512, device=device)
model.to(model.target_device)

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
train_dataloader = DataLoader(train_samples, shuffle=True, batch_size=train_batch_size,
                              collate_fn=model.batching_collate)

total_steps = len(train_dataloader)
warmup_steps = total_steps // 10
print(f'Total steps: {total_steps}, warmup steps: {warmup_steps}')
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-5)
criterion = torch.nn.CrossEntropyLoss()
scaler = torch.cuda.amp.GradScaler()
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps,
                                            num_training_steps=total_steps - warmup_steps)

step = 0
SCALER = False
model.zero_grad()
model.train()
for batch in tqdm(train_dataloader):
    queries, passages, labels = batch

    if SCALER:
        with autocast():
            outputs = model(queries, passages)
            loss = criterion(outputs.squeeze(), labels)

        scale_before_step = scaler.get_scale()
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()

        skip_scheduler = scaler.get_scale() != scale_before_step

        optimizer.zero_grad()

        if not skip_scheduler:
            scheduler.step()
    else:
        outputs = model(queries, passages)
        loss = criterion(outputs.squeeze(), labels)

        loss.backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

    step += 1
    if step % 10000 == 0:
        torch.save(model.state_dict(), model_save_path + f"/FullyCrossEncoder_{step}.pt")

torch.save(model.state_dict(), model_save_path + '/FullyCrossEncoder.pt')
print(f'Finished training. Model saved to {model_save_path}')
