from sentence_transformers import SentenceTransformer
from typing import List, Union
import torch
import numpy as np
from tqdm import trange
from sentence_transformers.util import batch_to_device
from torch.utils.data import Dataset
import random


class SentenceTransformer(SentenceTransformer):
    def __init__(self, model_name_or_path: str = None, modules: List = None, device: str = None, mode: str = 'text'):
        super().__init__(model_name_or_path, modules, device)
        self.mode = mode

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

        sentence_features = []
        for idx in range(len(batch[0].texts)):
            if self.mode == 'text':
                try:
                    tokenized = self.tokenizer(texts[idx], entity_spans=entity_spans[idx], entities=entities[idx],
                                               padding=True, truncation='longest_first', return_tensors="pt",
                                               max_length=512)
                except:
                    tokenized = self.tokenizer(texts[idx], padding=True, truncation='longest_first',
                                               return_tensors="pt", max_length=512)
                sentence_features.append(tokenized)
            elif self.mode == 'entity':
                entities_tokenized = self.entities_only_tokenizer(entities[idx])
                sentence_features.append(entities_tokenized)

        labels = torch.tensor(labels)

        return sentence_features, labels

    def encode(self, sentences: Union[str, List[str]],
               entity_spans: List = None,
               entities: List = None,
               batch_size: int = 32,
               show_progress_bar: bool = None,
               output_value: str = 'sentence_embedding',
               convert_to_numpy: bool = True,
               convert_to_tensor: bool = False,
               device: str = None,
               normalize_embeddings: bool = False):

        self.eval()
        if convert_to_tensor:
            convert_to_numpy = False

        if output_value != 'sentence_embedding':
            convert_to_tensor = False
            convert_to_numpy = False

        input_was_string = False
        if isinstance(sentences, str) or not hasattr(sentences, '__len__'):
            sentences = [sentences]
            input_was_string = True

        if device is None:
            device = self._target_device

        self.to(device)

        all_embeddings = []
        length_sorted_idx = np.argsort([-self._text_length(sen) for sen in sentences])
        sentences_sorted = [sentences[idx] for idx in length_sorted_idx]

        for start_index in trange(0, len(sentences), batch_size, desc="Batches", disable=not show_progress_bar):
            sentences_batch = sentences_sorted[start_index:start_index + batch_size]
            if self.mode == 'text':
                try:
                    features = self.tokenizer(sentences_batch, entity_spans=[entity_spans], entities=[entities],
                                              padding=True, truncation='longest_first', return_tensors="pt",
                                              max_length=512)

                except:
                    features = self.tokenizer(sentences_batch, padding=True, truncation='longest_first',
                                              return_tensors="pt", max_length=512)
            elif self.mode == 'entity':
                features = self.entities_only_tokenizer([entities])

            features_dict = {}
            for key in features:
                features_dict[key] = features[key].clone().detach()

            features = batch_to_device(features_dict, device)

            with torch.no_grad():
                out_features = self.forward(features)

                if output_value == 'token_embeddings':
                    embeddings = []
                    for token_emb, attention in zip(out_features[output_value], out_features['attention_mask']):
                        last_mask_id = len(attention) - 1
                        while last_mask_id > 0 and attention[last_mask_id].item() == 0:
                            last_mask_id -= 1

                        embeddings.append(token_emb[0:last_mask_id + 1])
                elif output_value is None:  # Return all outputs
                    embeddings = []
                    for sent_idx in range(len(out_features['sentence_embedding'])):
                        row = {name: out_features[name][sent_idx] for name in out_features}
                        embeddings.append(row)
                else:  # Sentence embeddings
                    embeddings = out_features[output_value]
                    embeddings = embeddings.detach()
                    if normalize_embeddings:
                        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

                    # fixes for #522 and #487 to avoid oom problems on gpu with large datasets
                    if convert_to_numpy:
                        embeddings = embeddings.cpu()

                all_embeddings.extend(embeddings)

        all_embeddings = [all_embeddings[idx] for idx in np.argsort(length_sorted_idx)]

        if convert_to_tensor:
            all_embeddings = torch.stack(all_embeddings)
        elif convert_to_numpy:
            all_embeddings = np.asarray([emb.numpy() for emb in all_embeddings])

        if input_was_string:
            all_embeddings = all_embeddings[0]

        return all_embeddings

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
        entities_tokenized = self.tokenizer(entity_texts, entity_spans=entity_entity_spans,
                                            entities=entity_entities, padding=True, truncation='longest_first',
                                            return_tensors="pt", max_length=512)
        return entities_tokenized


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


# We create a custom MSMARCO dataset that returns triplets (query, positive, negative)
# on-the-fly based on the information from the mined-hard-negatives jsonl file.
class MSMARCODataset(Dataset):
    def __init__(self, queries, corpus, ce_scores, queries_entities, passages_entities):
        self.queries = queries
        self.queries_ids = list(queries.keys())
        self.corpus = corpus
        self.ce_scores = ce_scores
        self.queries_entities = queries_entities
        self.passages_entities = passages_entities

        for qid in self.queries:
            self.queries[qid]['pos'] = list(self.queries[qid]['pos'])
            self.queries[qid]['neg'] = list(self.queries[qid]['neg'])
            random.shuffle(self.queries[qid]['neg'])

    def __getitem__(self, item):
        query = self.queries[self.queries_ids[item]]
        query_text = query['query']
        qid = query['qid']
        query_entity_spans = [(entity['start'], entity['end']) for entity in self.queries_entities[qid]]
        query_entities = [entity.get('title', entity.get('spot')) for entity in self.queries_entities[qid]]

        if len(query['pos']) > 0:
            pos_id = query['pos'].pop(0)  # Pop positive and add at end
            pos_text = self.corpus[pos_id]
            pos_entity_spans = [(entity['start'], entity['end']) for entity in self.passages_entities[pos_id]]
            pos_entities = [entity.get('title', entity.get('spot')) for entity in self.passages_entities[pos_id]]
            query['pos'].append(pos_id)
        else:  # We only have negatives, use two negs
            pos_id = query['neg'].pop(0)  # Pop negative and add at end
            pos_text = self.corpus[pos_id]
            pos_entity_spans = [(entity['start'], entity['end']) for entity in self.passages_entities[pos_id]]
            pos_entities = [entity.get('title', entity.get('spot')) for entity in self.passages_entities[pos_id]]
            query['neg'].append(pos_id)

        # Get a negative passage
        neg_id = query['neg'].pop(0)  # Pop negative and add at end
        neg_text = self.corpus[neg_id]
        neg_entity_spans = [(entity['start'], entity['end']) for entity in self.passages_entities[neg_id]]
        neg_entities = [entity.get('title', entity.get('spot')) for entity in self.passages_entities[neg_id]]
        query['neg'].append(neg_id)

        pos_score = self.ce_scores[qid][pos_id]
        neg_score = self.ce_scores[qid][neg_id]

        return InputExample(texts=[query_text, pos_text, neg_text], label=pos_score - neg_score,
                            entity_spans=[query_entity_spans, pos_entity_spans, neg_entity_spans],
                            entities=[query_entities, pos_entities, neg_entities])

    def __len__(self):
        return len(self.queries)
