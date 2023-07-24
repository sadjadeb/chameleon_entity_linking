import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer
from tqdm import tqdm, trange
import os
import sys
from model_chatgpt import BiEncoder

LOCAL = True if sys.platform == 'win32' else False
model_name = 'bert-base-uncased'
model_save_path = f'output/bi-encoder_cosine-embedding_{model_name.split("/")[-1]}'
train_batch_size = 4 if LOCAL else 32
device = 'cpu' if LOCAL else 'cuda:1'
max_seq_length = 512
num_epochs = 1

os.makedirs(model_save_path, exist_ok=True)


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


# Define the dataset class for MSMARCO
class MSMARCODataset(Dataset):
    def __init__(self, data_file, tokenizer):
        self.data_file = data_file
        self.tokenizer = tokenizer
        self.queries, self.passages = list(queries.values()), list(corpus.values())

    def __len__(self):
        return len(self.queries)

    def __getitem__(self, index):
        query = self.queries[index]
        passage = self.passages[index]
        encoding = self.tokenizer.encode_plus(query, passage, padding='max_length', truncation=True, max_length=512)
        return encoding['input_ids'], encoding['attention_mask']


# Load queries and passages from the data file
# Return a list of queries and passages


# Set the device
device = torch.device(device)

# Load the pre-trained BERT model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)
encoder = AutoModel.from_pretrained(model_name).to(device)

# Create the BiEncoder model
model = BiEncoder(encoder).to(device)

# Load the MSMARCO dataset
dataset = MSMARCODataset('msmarco_data.txt', tokenizer)
dataloader = DataLoader(dataset, batch_size=train_batch_size, shuffle=True)

# Define the optimizer and loss function
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
criterion = torch.nn.CosineEmbeddingLoss()

# Training loop
model.train()
for epoch in trange(num_epochs):
    total_loss = 0
    for input_ids, attention_mask in tqdm(dataloader):
        input_ids = torch.stack(input_ids).to(device)
        attention_mask = torch.stack(attention_mask).to(device)

        optimizer.zero_grad()

        # Forward pass
        query_embeddings = model(input_ids, attention_mask)

        # Create target tensor
        target = torch.ones(query_embeddings.size(0)).to(device)

        # Compute the loss
        loss = criterion(query_embeddings, query_embeddings, target)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss}")

# Save the trained model
torch.save(model.state_dict(), model_save_path + '/BiEncoder_ChatGPT.pt')
print(f'Finished training. Model saved to {model_save_path}')
