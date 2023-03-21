from sentence_transformers import CrossEncoder
import logging
from typing import List, Union
import torch


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


logger = logging.getLogger(__name__)


class CrossEncoder(CrossEncoder):
    def smart_batching_collate(self, batch):
        texts = [[] for _ in range(len(batch[0].texts))]
        labels = []
        entity_spans = [[] for _ in range(len(batch[0].entity_spans))]
        entities = [[] for _ in range(len(batch[0].entities))]

        for example in batch:
            for idx, text in enumerate(example.texts):
                texts[idx].append(text.strip())

            labels.append(example.label)

            for idx, each_entity_spans in enumerate(example.entity_spans):
                entity_spans[idx].append(each_entity_spans)

            for idx, each_entities in enumerate(example.entities):
                entities[idx].append(each_entities)

        try:
            tokenized = self.tokenizer(*texts, *entity_spans, *entities, padding=True, truncation='longest_first',
                                       return_tensors="pt", max_length=self.max_length)
        except:
            tokenized = self.tokenizer(*texts, padding=True, truncation='longest_first', return_tensors="pt",
                                       max_length=self.max_length)
        labels = torch.tensor(labels, dtype=torch.float).to(self._target_device)

        for name in tokenized:
            tokenized[name] = tokenized[name].to(self._target_device)

        return tokenized, labels

    def smart_batching_collate_text_only(self, batch):
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
            tokenized = self.tokenizer(*texts, *entity_spans, *entities, padding=True, truncation='longest_first',
                                       return_tensors="pt", max_length=self.max_length)
        except:
            entity_spans = [[], []]
            entities = [[], []]
            tokenized = self.tokenizer(*texts, *entity_spans, *entities, padding=True, truncation='longest_first',
                                       return_tensors="pt", max_length=self.max_length)
            # tokenized = self.tokenizer(*texts, padding=True, truncation='longest_first', return_tensors="pt", max_length=self.max_length)

        for name in tokenized:
            tokenized[name] = tokenized[name].to(self._target_device)

        return tokenized
