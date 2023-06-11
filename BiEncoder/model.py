from sentence_transformers import SentenceTransformer
from typing import List, Dict, Tuple, Iterable, Type, Union, Callable, Optional
import torch
import numpy as np
from tqdm import trange
from sentence_transformers.util import batch_to_device
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import Optimizer
from torch import Tensor
from sentence_transformers.util import fullname, pairwise_dot_score
from sentence_transformers.model_card_templates import ModelCardTemplate
from sentence_transformers.evaluation import SentenceEvaluator
from sentence_transformers.losses import MarginMSELoss
import random
import json


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
        entity_features = []
        for idx in range(len(batch[0].texts)):
            if self.mode == 'text':
                tokenized = self.tokenizer(texts[idx], padding=True, truncation='longest_first', return_tensors="pt",
                                           max_length=512)
                sentence_features.append(tokenized)
            elif self.mode == 'entity':
                entities_tokenized = self.entities_only_tokenizer(entities[idx])
                entity_features.append(entities_tokenized)
            elif self.mode == 'text_entity':
                entities_tokenized = self.entities_only_tokenizer(entities[idx])
                tokenized = self.tokenizer(texts[idx], padding=True, truncation='longest_first', return_tensors="pt",
                                           max_length=512)
                sentence_features.append(tokenized)
                entity_features.append(entities_tokenized)

        labels = torch.tensor(labels)

        if self.mode == 'text':
            return sentence_features, labels
        elif self.mode == 'entity':
            return entity_features, labels
        elif self.mode == 'text_entity':
            return sentence_features, entity_features, labels

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

        all_sentence_embeddings = []
        all_entity_embeddings = []
        length_sorted_idx = np.argsort([-self._text_length(sen) for sen in sentences])
        sentences_sorted = [sentences[idx] for idx in length_sorted_idx]

        for start_index in trange(0, len(sentences), batch_size, desc="Batches", disable=not show_progress_bar):
            def get_inner_embeddings(features):
                all_embeddings = []
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

        sentences_batch = sentences_sorted[start_index:start_index + batch_size]
        if self.mode == 'text':
            features = self.tokenizer(sentences_batch, padding=True, truncation='longest_first', return_tensors="pt",
                                      max_length=512)
            all_sentence_embeddings = get_inner_embeddings(features)
        elif self.mode == 'entity':
            features = self.entities_only_tokenizer([entities])
            all_entity_embeddings = get_inner_embeddings(features)
        elif self.mode == 'text_entity':
            sentence_features = self.tokenizer(sentences_batch, padding=True, truncation='longest_first',
                                               return_tensors="pt", max_length=512)
            entity_features = self.entities_only_tokenizer([entities])
            all_sentence_embeddings = get_inner_embeddings(sentence_features)
            all_entity_embeddings = get_inner_embeddings(entity_features)

        return all_sentence_embeddings, all_entity_embeddings

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

    def fit(self,
            train_objectives: Iterable[Tuple[DataLoader, nn.Module]],
            evaluator: SentenceEvaluator = None,
            epochs: int = 1,
            steps_per_epoch=None,
            scheduler: str = 'WarmupLinear',
            warmup_steps: int = 10000,
            optimizer_class: Type[Optimizer] = torch.optim.AdamW,
            optimizer_params: Dict[str, object] = {'lr': 2e-5},
            weight_decay: float = 0.01,
            evaluation_steps: int = 0,
            output_path: str = None,
            save_best_model: bool = True,
            max_grad_norm: float = 1,
            use_amp: bool = False,
            callback: Callable[[float, int, int], None] = None,
            show_progress_bar: bool = True,
            checkpoint_path: str = None,
            checkpoint_save_steps: int = 500,
            checkpoint_save_total_limit: int = 0
            ):

        ##Add info to model card
        # info_loss_functions = "\n".join(["- {} with {} training examples".format(str(loss), len(dataloader)) for dataloader, loss in train_objectives])
        info_loss_functions = []
        for dataloader, loss in train_objectives:
            info_loss_functions.extend(ModelCardTemplate.get_train_objective_info(dataloader, loss))
        info_loss_functions = "\n\n".join([text for text in info_loss_functions])

        info_fit_parameters = json.dumps(
            {"evaluator": fullname(evaluator), "epochs": epochs, "steps_per_epoch": steps_per_epoch,
             "scheduler": scheduler, "warmup_steps": warmup_steps, "optimizer_class": str(optimizer_class),
             "optimizer_params": optimizer_params, "weight_decay": weight_decay, "evaluation_steps": evaluation_steps,
             "max_grad_norm": max_grad_norm}, indent=4, sort_keys=True)
        self._model_card_text = None
        self._model_card_vars['{TRAINING_SECTION}'] = ModelCardTemplate.__TRAINING_SECTION__.replace("{LOSS_FUNCTIONS}",
                                                                                                     info_loss_functions).replace(
            "{FIT_PARAMETERS}", info_fit_parameters)

        if use_amp:
            from torch.cuda.amp import autocast
            scaler = torch.cuda.amp.GradScaler()

        self.to(self._target_device)

        dataloaders = [dataloader for dataloader, _ in train_objectives]

        # Use smart batching
        for dataloader in dataloaders:
            dataloader.collate_fn = self.smart_batching_collate

        loss_models = [loss for _, loss in train_objectives]
        for loss_model in loss_models:
            loss_model.to(self._target_device)

        self.best_score = -9999999

        if steps_per_epoch is None or steps_per_epoch == 0:
            steps_per_epoch = min([len(dataloader) for dataloader in dataloaders])

        num_train_steps = int(steps_per_epoch * epochs)

        # Prepare optimizers
        optimizers = []
        schedulers = []
        for loss_model in loss_models:
            param_optimizer = list(loss_model.named_parameters())

            no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
            optimizer_grouped_parameters = [
                {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
                 'weight_decay': weight_decay},
                {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
            ]

            optimizer = optimizer_class(optimizer_grouped_parameters, **optimizer_params)
            scheduler_obj = self._get_scheduler(optimizer, scheduler=scheduler, warmup_steps=warmup_steps,
                                                t_total=num_train_steps)

            optimizers.append(optimizer)
            schedulers.append(scheduler_obj)

        global_step = 0
        data_iterators = [iter(dataloader) for dataloader in dataloaders]

        num_train_objectives = len(train_objectives)

        skip_scheduler = False
        for epoch in trange(epochs, desc="Epoch", disable=not show_progress_bar):
            training_steps = 0

            for loss_model in loss_models:
                loss_model.zero_grad()
                loss_model.train()

            for _ in trange(steps_per_epoch, desc="Iteration", smoothing=0.05, disable=not show_progress_bar):
                for train_idx in range(num_train_objectives):
                    loss_model = loss_models[train_idx]
                    optimizer = optimizers[train_idx]
                    scheduler = schedulers[train_idx]
                    data_iterator = data_iterators[train_idx]

                    try:
                        data = next(data_iterator)
                    except StopIteration:
                        data_iterator = iter(dataloaders[train_idx])
                        data_iterators[train_idx] = data_iterator
                        data = next(data_iterator)

                    if self.mode == 'text_entity':
                        sentence_features, entity_features, labels = data
                        sentence_features = list(map(lambda batch: batch_to_device(batch, self._target_device), sentence_features))
                        entity_features = list(map(lambda batch: batch_to_device(batch, self._target_device), entity_features))
                    else:
                        features, labels = data
                        features = list(map(lambda batch: batch_to_device(batch, self._target_device), features))
                    labels = labels.to(self._target_device)

                    if use_amp:
                        with autocast():
                            if self.mode == 'text_entity':
                                loss_value = loss_model(sentence_features, labels, entity_features)
                            else:
                                loss_value = loss_model(features, labels)

                        scale_before_step = scaler.get_scale()
                        scaler.scale(loss_value).backward()
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(loss_model.parameters(), max_grad_norm)
                        scaler.step(optimizer)
                        scaler.update()

                        skip_scheduler = scaler.get_scale() != scale_before_step
                    else:
                        if self.mode == 'text_entity':
                            loss_value = loss_model(sentence_features, labels, entity_features)
                        else:
                            loss_value = loss_model(features, labels)
                        loss_value.backward()
                        torch.nn.utils.clip_grad_norm_(loss_model.parameters(), max_grad_norm)
                        optimizer.step()

                    optimizer.zero_grad()

                    if not skip_scheduler:
                        scheduler.step()

                training_steps += 1
                global_step += 1

                if evaluation_steps > 0 and training_steps % evaluation_steps == 0:
                    self._eval_during_training(evaluator, output_path, save_best_model, epoch, training_steps, callback)

                    for loss_model in loss_models:
                        loss_model.zero_grad()
                        loss_model.train()

                if checkpoint_path is not None and checkpoint_save_steps is not None and checkpoint_save_steps > 0 and global_step % checkpoint_save_steps == 0:
                    self._save_checkpoint(checkpoint_path, checkpoint_save_total_limit, global_step)

            self._eval_during_training(evaluator, output_path, save_best_model, epoch, -1, callback)

        if evaluator is None and output_path is not None:  # No evaluator, but output path: save final model version
            self.save(output_path)

        if checkpoint_path is not None:
            self._save_checkpoint(checkpoint_path, checkpoint_save_total_limit, global_step)


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


class MarginMSELossOnTextAndEntity(nn.Module):
    def __init__(self, model, similarity_fct=pairwise_dot_score):
        """
        :param model: SentenceTransformerModel
        :param similarity_fct:  Which similarity function to use.
        """
        super(MarginMSELossOnTextAndEntity, self).__init__()
        self.model = model
        self.similarity_fct = similarity_fct
        self.loss_fct = nn.MSELoss()

    def forward(self, sentence_features: Iterable[Dict[str, Tensor]], labels: Tensor, entity_features: Iterable[Dict[str, Tensor]] = None):
        # sentence_features: query, positive passage, negative passage
        if entity_features is None:
            reps = [self.model(sentence_feature)['sentence_embedding'] for sentence_feature in sentence_features]
            embeddings_query = reps[0]
            embeddings_pos = reps[1]
            embeddings_neg = reps[2]
            scores_pos = self.similarity_fct(embeddings_query, embeddings_pos)
            scores_neg = self.similarity_fct(embeddings_query, embeddings_neg)
            margin_pred = scores_pos - scores_neg
        else:
            sentence_reps = [self.model(sentence_feature)['sentence_embedding'] for sentence_feature in sentence_features]
            entity_reps = [self.model(entity_feature)['sentence_embedding'] for entity_feature in entity_features]

            sentence_embeddings_query = sentence_reps[0]
            sentence_embeddings_pos = sentence_reps[1]
            sentence_embeddings_neg = sentence_reps[2]
            entity_embeddings_query = entity_reps[0]
            entity_embeddings_pos = entity_reps[1]
            entity_embeddings_neg = entity_reps[2]

            sentence_scores_pos = self.similarity_fct(sentence_embeddings_query, sentence_embeddings_pos)
            sentence_scores_neg = self.similarity_fct(sentence_embeddings_query, sentence_embeddings_neg)
            entity_scores_pos = self.similarity_fct(entity_embeddings_query, entity_embeddings_pos)
            entity_scores_neg = self.similarity_fct(entity_embeddings_query, entity_embeddings_neg)

            margin_pred = (0.9 * (sentence_scores_pos - sentence_scores_neg)) + (0.1 * (entity_scores_pos - entity_scores_neg))

        return self.loss_fct(margin_pred, labels)
