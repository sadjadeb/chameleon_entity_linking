import json
from trainDataset import Dataset
from model import KnowledgeAwareTextMatching
from loss import NegativeSamplingInBatchLoss
from PadCollate import PadCollate
import torch
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup, LukeTokenizer
from tqdm import tqdm

import warnings

warnings.filterwarnings("ignore")

if torch.cuda.is_available():
    device = torch.device("cuda")
    torch.cuda.set_device(1)
else:
    device = torch.device("cpu")


def train(config):
    step = 0

    model = KnowledgeAwareTextMatching(device)
    model.to(device)
    tokenizer = LukeTokenizer.from_pretrained("studio-ousia/luke-base")

    optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'])

    datasetTrain = Dataset(dataPath=config['dataPath'], tokenizerName=config['Model'], type='train')
    trainDataloader = DataLoader(dataset=datasetTrain, shuffle=True, batch_size=config['batch'],
                                 collate_fn=PadCollate(pad_token_id=tokenizer.pad_token_id,
                                                       entity_pad_token_id=tokenizer.entity_pad_token_id), drop_last=True)

    total_steps = len(trainDataloader) * config['epochs']
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(0.01 * total_steps),
                                                num_training_steps=total_steps)

    criterion = NegativeSamplingInBatchLoss(device)
    model.zero_grad()
    for i in range(config['epochs']):
        for batch in tqdm(trainDataloader):
            query_inputs, doc_inputs, queryID, docID, relevance_grade = batch

            model.train()
            outputs = model(query_inputs, doc_inputs, 'train')

            loss = criterion(outputs)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            step += 1
            if step % 10000 == 0:
                torch.save(model, f"{config['output']}KnowledgeAwareTextMatching_TEXT_{i}_{step}.model")

    torch.save(model, config['output'] + 'KnowledgeAwareTextMatching_TEXT.model')


if __name__ == '__main__':
    with open("local_config.json", "r") as jsonfile:
        config = json.load(jsonfile)  # Reading the file
        print(config)
    train(config)
