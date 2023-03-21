import json
from torch.utils.data import DataLoader
from PadCollate import PadCollate
from testDataset import Dataset
import torch
from operator import itemgetter
from tqdm import tqdm
from transformers import LukeTokenizer

import warnings

warnings.filterwarnings("ignore")

if torch.cuda.is_available():
    device = torch.device("cuda")
    torch.cuda.set_device(1)
else:
    device = torch.device("cpu")


def test(config):
    model = torch.load(config['output'] + 'KnowledgeAwareTextMatching_TEXT.model')
    model.to(device)
    tokenizer = LukeTokenizer.from_pretrained(config['Model'])

    datasetTest = Dataset(dataPath=config['dataPath'], tokenizerName=config['Model'], type='dev')
    print('{:>5,} validation samples'.format(len(datasetTest)))
    TestDataloader = DataLoader(dataset=datasetTest, batch_size=config['batch'],
                                collate_fn=PadCollate(tokenizer.pad_token_id, tokenizer.pad_token_type_id))

    validationRun = {}
    for batch in tqdm(TestDataloader):
        query_inputs, docs_inputs, queryID, docID, relevance_grade = batch

        with torch.no_grad():
            outputs = model(query_inputs, docs_inputs, 'test')

        for q in range(len(queryID)):
            if queryID[q] in validationRun:
                validationRun[queryID[q]].append((docID[q], float(outputs[q].item())))
            else:
                validationRun[queryID[q]] = [(docID[q], float(outputs[q].item()))]

    runFileName = config['output'] + 'Run.txt'
    runFile = open(runFileName, 'w')
    for query, retrievalList in validationRun.items():
        rank = 1
        retrievalList = sorted(retrievalList, key=itemgetter(1), reverse=True)
        for doc, retrievalScore in retrievalList:
            runFile.write(str(query) + '\t' + 'Q0' + '\t' + str(doc) + '\t' + str(rank) + '\t' + str(
                retrievalScore) + '\t' + 'Luke_TEXT\n')
            rank = rank + 1

    runFile.close()


if __name__ == '__main__':
    with open("local_configTest.json", "r") as jsonfile:
        config = json.load(jsonfile)  # Reading the file
        print(config)
    test(config)
