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

        sample = self.Qrel[index]
        queryID = sample[0]
        queryText = self.query_dic[queryID]

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

        doc_inputs = self.tokenizer(self.collection_dic[docID], add_prefix_space=True, return_tensors="pt", truncation=True)
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
        return (len(self.Qrel))

    def read(self):
        import csv
        self.query_dic = {}
        self.collection_dic = {}
        self.query_entity_dic = {}
        self.doc_entity_dic = {}
        self.Qrel = []
        self.lable = {}

        collection_file = open(self.dataPath + "/collection.tsv")
        read_tsv = csv.reader(collection_file, delimiter="\t")
        for row in read_tsv:
            self.collection_dic[int(row[0])] = row[1]

        query_file = open(self.dataPath + "/queries.dev.small.tsv")
        read_tsv = csv.reader(query_file, delimiter="\t")
        for row in read_tsv:
            self.query_dic[int(row[0])] = row[1]

        qrel_file = open(self.dataPath + "/RetrievedwithRelevancedev")
        read_tsv = csv.reader(qrel_file, delimiter="\t")
        for row in read_tsv:
            self.Qrel.append([int(row[0]), int(row[1]), float(row[2])])

        # read_tsv = csv.reader(qrel_file, delimiter=" ")
        # for row in read_tsv:
        #     self.Qrel.append([int(row[0]), int(row[2]), float(row[4])])
