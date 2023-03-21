from torch.utils.data import Dataset
from transformers import LukeTokenizer
import torch
from typing import List, Dict
from irtools.pad import pad_batch


class Dataset(Dataset):
    def __init__(self, dataPath, tokenizerName, type):
        self.dataPath = dataPath
        self.type = type
        self.tokenizer = LukeTokenizer.from_pretrained(tokenizerName)
        self._pad_values = {
            "input_ids": self.tokenizer.pad_token_id,
            "attention_mask": 0,
        }
        self.read()

    def __getitem__(self, index):
        doc_outputs: List[Dict[str, torch.Tensor]] = []
        query_outputs: List[Dict[str, torch.Tensor]] = []

        sample = self.qrels[index]
        queryID = sample[0]
        queryText = self.query_dict[queryID]
        query_inputs = self.tokenizer(queryText, add_prefix_space=True, return_tensors="pt", truncation=True)
        assert all(v.size(0) == 1 for v in query_inputs.values())
        query_outputs.append({k: v[0] for k, v in query_inputs.items()})

        query_encoded_output: Dict[str, torch.Tensor] = {}
        for key in query_outputs[0]:
            padded = pad_batch(
                [one[key].type(torch.int64) for one in query_outputs], value=self._pad_values[key]
            )
            tensor = torch.stack(padded)
            query_encoded_output[key] = tensor

        docID = sample[1]
        relevance_grade = sample[2]

        doc_inputs = self.tokenizer(self.collection_dict[docID], add_prefix_space=True, return_tensors="pt",
                                    truncation=True)
        assert all(v.size(0) == 1 for v in doc_inputs.values())
        doc_outputs.append({k: v[0] for k, v in doc_inputs.items()})
        doc_encoded_output: Dict[str, torch.Tensor] = {}
        for key in doc_outputs[0]:
            padded = pad_batch(
                [one[key].type(torch.int64) for one in doc_outputs], value=self._pad_values[key]
            )
            tensor = torch.stack(padded)
            doc_encoded_output[key] = tensor

        return query_encoded_output, doc_encoded_output, queryID, docID, relevance_grade

    def __len__(self):
        return len(self.qrels)

    def read(self):
        import csv
        self.query_dict = {}
        self.collection_dict = {}
        self.qrels = []
        self.lable = {}

        collection_file = open(self.dataPath + "/collection.tsv")
        read_tsv = csv.reader(collection_file, delimiter="\t")
        for row in read_tsv:
            self.collection_dict[int(row[0])] = row[1]

        query_file = open(self.dataPath + "/queries.train.tsv")
        read_tsv = csv.reader(query_file, delimiter="\t")
        for row in read_tsv:
            self.query_dict[int(row[0])] = row[1]

        # qrel_file = open(self.dataPath + "/RetrievedwithRelevancetrain")
        # read_tsv = csv.reader(qrel_file, delimiter="\t")
        # for row in read_tsv:
        #     self.qrels.append([int(row[0]), int(row[1]), float(row[2])])

        import ir_datasets
        dataset = ir_datasets.load("msmarco-passage/train")
        for qrel in dataset.qrels_iter():
            self.qrels.append([int(qrel.query_id), int(qrel.doc_id), float(qrel.relevance)])
