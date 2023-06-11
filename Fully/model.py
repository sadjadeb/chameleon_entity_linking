from typing import List, Union
import torch
from torch import nn
from transformers import AutoModel, AutoTokenizer, LukeModel, LukeTokenizer
from transformers import logging
import numpy as np

logging.set_verbosity_error()


class FullyCrossEncoder(nn.Module):
    def __init__(self, model_name: str = None, max_length: int = 512, device: str = None, mode: str = 'text_only'):
        if model_name is None:
            raise ValueError("model_name must be provided")

        super().__init__()
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.target_device = torch.device(device)
        self.max_length = max_length
        self.mode = mode

        self.text_tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.text_language_model = AutoModel.from_pretrained(model_name)
        self.entity_tokenizer = LukeTokenizer.from_pretrained('studio-ousia/luke-base')
        self.entity_language_model = LukeModel.from_pretrained('studio-ousia/luke-base')
        self.cosine_similarity = nn.CosineSimilarity()

    def forward(self, queries, passages, queries_entities, passages_entities):
        output = 0
        if self.mode == 'text_only':
            queries_outputs = self.text_language_model(**queries)
            passages_outputs = self.text_language_model(**passages)
            queries_representation = queries_outputs.pooler_output
            passages_representation = passages_outputs.pooler_output
            output = self.cosine_similarity(queries_representation, passages_representation)
        elif self.mode == 'entity_only':
            if queries_entities == {}:
                queries_entities_outputs = self.entity_language_model(**queries)
                passages_entities_outputs = self.entity_language_model(**passages)
            else:
                queries_entities_outputs = self.entity_language_model(**queries_entities)
                passages_entities_outputs = self.entity_language_model(**passages_entities)

            queries_entities_representation = queries_entities_outputs.pooler_output
            passages_entities_representation = passages_entities_outputs.pooler_output
            output = self.cosine_similarity(queries_entities_representation, passages_entities_representation)
        elif self.mode == 'text_entity':
            if queries_entities == {}:
                queries_entities_outputs = self.entity_language_model(**queries)
                passages_entities_outputs = self.entity_language_model(**passages)
            else:
                queries_entities_outputs = self.entity_language_model(**queries_entities)
                passages_entities_outputs = self.entity_language_model(**passages_entities)

            queries_entities_representation = queries_entities_outputs.pooler_output
            passages_entities_representation = passages_entities_outputs.pooler_output

            queries_outputs = self.text_language_model(**queries)
            passages_outputs = self.text_language_model(**passages)

            queries_representation = queries_outputs.pooler_output
            passages_representation = passages_outputs.pooler_output

            text_score = self.cosine_similarity(queries_entities_representation, passages_entities_representation)
            entity_score = self.cosine_similarity(queries_representation, passages_representation)

            output = torch.mean(torch.stack([text_score, entity_score]), dim=0)

        return output

    def entities_only_tokenizer(self, entities: List):
        entity_texts = []
        entity_entity_spans = []
        entity_entities = []
        for list_of_entities in entities:
            entity_texts.append(' '.join(list_of_entities))
            each_entity_entity_spans = []
            for idx, each_entity in enumerate(list_of_entities):
                if idx == 0:
                    each_entity_entity_spans.append((0, len(each_entity) - 1))
                else:
                    elen = len(list_of_entities[idx - 1]) + 1
                    each_entity_entity_spans.append((elen, elen + len(each_entity) - 1))
            entity_entity_spans.append(each_entity_entity_spans)
            entity_entities.append(list_of_entities)
        entities_tokenized = self.entity_tokenizer(entity_texts, entity_spans=entity_entity_spans,
                                                   entities=entity_entities, padding=True, truncation='longest_first',
                                                   return_tensors="pt", max_length=self.max_length)
        return entities_tokenized

    def batching_collate(self, batch):
        texts = [[] for _ in range(len(batch[0].texts))]
        entity_spans = [[] for _ in range(len(batch[0].entity_spans))]
        entities = [[] for _ in range(len(batch[0].entities))]
        labels = []

        for example in batch:
            for idx, text in enumerate(example.texts):
                texts[idx].append(text.strip())

            for idx, each_entity_spans in enumerate(example.entity_spans):
                entity_spans[idx].append(each_entity_spans)

            for idx, each_entities in enumerate(example.entities):
                entities[idx].append(each_entities)

            labels.append(example.label)

        try:
            queries_tokenized = self.text_tokenizer(texts[0], entity_spans=entity_spans[0], entities=entities[0],
                                                    padding=True, truncation='longest_first', return_tensors="pt",
                                                    max_length=self.max_length)
            passages_tokenized = self.text_tokenizer(texts[1], entity_spans=entity_spans[1], entities=entities[1],
                                                     padding=True, truncation='longest_first', return_tensors="pt",
                                                     max_length=self.max_length)
            # tokenize entities separately
            queries_entities_tokenized = self.entities_only_tokenizer(entities[0])
            passages_entities_tokenized = self.entities_only_tokenizer(entities[1])

        except:
            queries_tokenized = self.text_tokenizer(texts[0], padding=True, truncation='longest_first',
                                                    return_tensors="pt",
                                                    max_length=self.max_length)
            passages_tokenized = self.text_tokenizer(texts[1], padding=True, truncation='longest_first',
                                                     return_tensors="pt",
                                                     max_length=self.max_length)
            queries_entities_tokenized = {}
            passages_entities_tokenized = {}

        for name in queries_tokenized:
            queries_tokenized[name] = queries_tokenized[name].to(self.target_device)
        for name in passages_tokenized:
            passages_tokenized[name] = passages_tokenized[name].to(self.target_device)
        for name in queries_entities_tokenized:
            queries_entities_tokenized[name] = queries_entities_tokenized[name].to(self.target_device)
        for name in passages_entities_tokenized:
            passages_entities_tokenized[name] = passages_entities_tokenized[name].to(self.target_device)
        labels = torch.tensor(labels, dtype=torch.float).to(self.target_device)

        return queries_tokenized, passages_tokenized, queries_entities_tokenized, passages_entities_tokenized, labels

    def batching_collate_without_label(self, batch):
        texts = [[], []]
        entity_spans = [[], []]
        entities = [[], []]

        for example in batch:
            for idx, text in enumerate(example[0]):
                texts[idx].append(text.strip())

            for idx, each_entity_spans in enumerate(example[1]):
                entity_spans[idx].append(each_entity_spans)

            for idx, each_entities in enumerate(example[2]):
                entities[idx].append(each_entities)

        try:
            queries_tokenized = self.text_tokenizer(texts[0], entity_spans=entity_spans[0], entities=entities[0],
                                                    padding=True, truncation='longest_first', return_tensors="pt",
                                                    max_length=self.max_length)
            passages_tokenized = self.text_tokenizer(texts[1], entity_spans=entity_spans[1], entities=entities[1],
                                                     padding=True, truncation='longest_first', return_tensors="pt",
                                                     max_length=self.max_length)
            # tokenize entities separately
            queries_entities_tokenized = self.entities_only_tokenizer(entities[0])
            passages_entities_tokenized = self.entities_only_tokenizer(entities[1])

        except:
            queries_tokenized = self.text_tokenizer(texts[0], padding=True, truncation='longest_first',
                                                    return_tensors="pt",
                                                    max_length=self.max_length)
            passages_tokenized = self.text_tokenizer(texts[1], padding=True, truncation='longest_first',
                                                     return_tensors="pt",
                                                     max_length=self.max_length)
            queries_entities_tokenized = {}
            passages_entities_tokenized = {}

        for name in queries_tokenized:
            queries_tokenized[name] = queries_tokenized[name].to(self.target_device)
        for name in passages_tokenized:
            passages_tokenized[name] = passages_tokenized[name].to(self.target_device)
        for name in queries_entities_tokenized:
            queries_entities_tokenized[name] = queries_entities_tokenized[name].to(self.target_device)
        for name in passages_entities_tokenized:
            passages_entities_tokenized[name] = passages_entities_tokenized[name].to(self.target_device)

        return queries_tokenized, passages_tokenized, queries_entities_tokenized, passages_entities_tokenized

    def batching_collate_merge(self, batch):
        texts = [[] for _ in range(len(batch[0].texts))]
        labels = []

        for example in batch:
            for idx, text in enumerate(example.texts):
                texts[idx].append(text.strip())
            labels.append(example.label)

        tokenized = self.text_tokenizer(*texts, padding=True, truncation='longest_first', return_tensors="pt",
                                        max_length=self.max_length)

        for name in tokenized:
            tokenized[name] = tokenized[name].to(self.target_device)
        labels = torch.tensor(labels, dtype=torch.float).to(self.target_device)

        return tokenized, labels

    def batching_collate_merge_without_label(self, batch):
        texts = [[], []]

        for example in batch:
            for idx, text in enumerate(example[0]):
                texts[idx].append(text.strip())
        tokenized = self.text_tokenizer(*texts, padding=True, truncation='longest_first', return_tensors="pt",
                                        max_length=self.max_length)

        for name in tokenized:
            tokenized[name] = tokenized[name].to(self.target_device)

        return tokenized


class InputExample:
    """
    Structure for one input example with texts, the label and a unique id
    """

    def __init__(self, guid: str = '', texts: List[str] = None, label: Union[int, float] = 0,
                 entity_spans: List = [], entities: List = []):
        """
        Creates one InputExample with the given texts, guid, label, entity_spans, entities
        :param guid
            id for the example
        :param texts
            the texts for the example.
        :param label
            the label for the example
        """
        self.guid = guid
        self.texts = texts
        self.label = label
        self.entity_spans = entity_spans
        self.entities = entities

    def __str__(self):
        return "<InputExample> label: {}, texts: {}".format(str(self.label), "; ".join(self.texts))
