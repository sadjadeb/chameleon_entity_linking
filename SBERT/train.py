from torch.utils.data import DataLoader
from sentence_transformers import LoggingHandler, InputExample
from sentence_transformers.cross_encoder import CrossEncoder
import logging
import os
import random

#### Just some code to print debug information to stdout
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])

# First, we define the transformer model we want to fine-tune
# model_name = 'distilroberta-base'
model_name = "studio-ousia/luke-base"
train_batch_size = 32
train_batch_size = 4
num_epochs = 1
warmup_steps = 5000
device = 'cpu'
# Maximal number of training samples we want to use
max_train_samples = 2e7
# We use a positive-to-negative ratio: For 1 positive sample (label 1) we include 4 negative samples (label 0)
pos_neg_ration = 4

model_save_path = f'output/cross-encoder_{model_name.split("/")[-1]}'

# We set num_labels=1, which predicts a continuous score between 0 and 1
model = CrossEncoder(model_name, num_labels=1, max_length=512, device=device)

### Now we read the MS Marco dataset
data_folder = '/home/sajadeb/msmarco'
data_folder = r'C:\Users\sajad\PycharmProjects\chameleon_entity_linking\msmarco'

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

# Read our training file
cnt = 0
train_samples = []
train_filepath = os.path.join(data_folder, 'qidpidtriples.train.full.2.tsv')
with open(train_filepath, 'r', encoding='utf8') as fIn:
    print('Loading triples...')
    lines = fIn.readlines()
    random.shuffle(lines)

    for line in lines:
        qid, pos_id, neg_id = line.strip().split()

        query = queries[qid]
        if (cnt % (pos_neg_ration + 1)) == 0:
            passage = corpus[pos_id]
            label = 1
        else:
            passage = corpus[neg_id]
            label = 0

        train_samples.append(InputExample(texts=[query, passage], label=label))
        cnt += 1

        if cnt >= max_train_samples:
            break

# We create a DataLoader to load our train samples
train_dataloader = DataLoader(train_samples, shuffle=True, batch_size=train_batch_size)

# Train the model
model.fit(train_dataloader=train_dataloader,
          epochs=num_epochs,
          warmup_steps=warmup_steps,
          use_amp=True)

# Save latest model
model.save(model_save_path + '-latest')
