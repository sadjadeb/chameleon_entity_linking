from transformers import LukeTokenizer
import torch
from torch.utils.data import Dataset
from typing import List, Dict
from irtools.pad import pad_batch
import ir_datasets


class Dataset(Dataset):
    def __init__(self, dataPath, tokenizerName, type):
        self.dataPath = dataPath
        self.type = type
        self.tokenizer = LukeTokenizer.from_pretrained(tokenizerName)
        self._pad_values = {
            "input_ids": self.tokenizer.pad_token_id,
            "attention_mask": 0,
        }

        self.query_dict = {}
        self.collection_dict = {}
        self.qrels = []

        self.read()

    def __getitem__(self, index):
        doc_outputs: List[Dict[str, torch.Tensor]] = []
        query_outputs: List[Dict[str, torch.Tensor]] = []

        sample = self.qrels[index]
        queryID = sample[0]
        queryText = self.query_dict[queryID]

        query_inputs = self.tokenizer(queryText, add_prefix_space=True, return_tensors="pt", truncation=True)
        query_inputs = {k: v.squeeze() for k, v in query_inputs.items()}


        query_encoded_output: Dict[str, torch.Tensor] = {}
        for key in query_inputs.keys():
            padded = pad_batch(
                [out[key].type(torch.int64) for out in query_outputs], value=self._pad_values[key]
            )
            tensor = torch.stack(padded)
            query_encoded_output[key] = tensor
        print(query_encoded_output)

        docID = sample[1]
        relevance_grade = sample[2]


        doc_inputs = self.tokenizer(self.collection_dict[docID], add_prefix_space=True, return_tensors="pt", truncation=True)
        assert all(v.size(0) == 1 for v in doc_inputs.values())
        doc_outputs.append({k: v[0] for k, v in doc_inputs.items()})
        doc_encoded_output: Dict[str, torch.Tensor] = {}
        for key in doc_outputs[0]:
            padded = pad_batch(
                [one[key].type(torch.int64) for one in doc_outputs], value=self._pad_values[key]
            )
            tensor = torch.stack(padded)
            doc_encoded_output[key] = tensor
        print(doc_encoded_output)

        return query_encoded_output, doc_encoded_output, queryID, docID, relevance_grade

    def __len__(self):
        return len(self.qrels)

    def read(self):
        if self.type == 'train':
            dataset = ir_datasets.load("msmarco-passage/train")
        elif self.type == 'dev' or self.type == 'test':
            dataset = ir_datasets.load("msmarco-passage/dev/small")
        else:
            raise KeyError('Run type is not defined')

        for doc in dataset.docs_iter():
            self.collection_dict[int(doc.doc_id)] = doc.text

        for query in dataset.queries_iter():
            self.query_dict[int(query.query_id)] = query.text

        for qrel in dataset.qrels_iter():
            self.qrels.append([int(qrel.query_id), int(qrel.doc_id), float(qrel.relevance)])
