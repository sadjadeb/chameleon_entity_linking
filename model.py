import math
import torch
import torch.nn.functional as F
from torch import nn
from transformers.models.bert.modeling_bert import BertModel, BertSelfOutput, BertIntermediate, BertOutput


class WordEmbeddings(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.device = config.device
        self.embeddings = BertModel.from_pretrained(config.bertModel)

    def forward(self, text_inputs):
        input_ids = text_inputs["input_ids"].to(self.device)
        attention_mask = text_inputs["attention_mask"].to(self.device)
        token_type_ids = text_inputs.get("token_type_ids").to(self.device)
        model_outputs = self.embeddings(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )

        return model_outputs


class EntityEmbeddings(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.entity_embeddings = nn.Embedding(config.entity_vocab_size, config.entity_emb_size, padding_idx=0)

    def forward(self, entity_ids: torch.LongTensor):
        entity_embeddings = self.entity_embeddings(entity_ids)
        return entity_embeddings


class EntityAwareSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.w2e_query = nn.Linear(config.hidden_size, self.all_head_size)
        self.e2w_query = nn.Linear(config.hidden_size, self.all_head_size)
        self.e2e_query = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        return x.view(*new_x_shape).permute(0, 2, 1, 3)

    # def forward(self, word_hidden_states, entity_hidden_states, attention_mask):
    def forward(self, word_hidden_states, entity_hidden_states):
        word_size = word_hidden_states.size(1)

        w2w_query_layer = self.transpose_for_scores(self.query(word_hidden_states))
        w2e_query_layer = self.transpose_for_scores(self.w2e_query(word_hidden_states))
        e2w_query_layer = self.transpose_for_scores(self.e2w_query(entity_hidden_states))
        e2e_query_layer = self.transpose_for_scores(self.e2e_query(entity_hidden_states))

        key_layer = self.transpose_for_scores(self.key(torch.cat([word_hidden_states, entity_hidden_states], dim=1)))

        w2w_key_layer = key_layer[:, :, :word_size, :]
        e2w_key_layer = key_layer[:, :, :word_size, :]
        w2e_key_layer = key_layer[:, :, word_size:, :]
        e2e_key_layer = key_layer[:, :, word_size:, :]

        w2w_attention_scores = torch.matmul(w2w_query_layer, w2w_key_layer.transpose(-1, -2))
        w2e_attention_scores = torch.matmul(w2e_query_layer, w2e_key_layer.transpose(-1, -2))
        e2w_attention_scores = torch.matmul(e2w_query_layer, e2w_key_layer.transpose(-1, -2))
        e2e_attention_scores = torch.matmul(e2e_query_layer, e2e_key_layer.transpose(-1, -2))

        word_attention_scores = torch.cat([w2w_attention_scores, w2e_attention_scores], dim=3)
        entity_attention_scores = torch.cat([e2w_attention_scores, e2e_attention_scores], dim=3)
        attention_scores = torch.cat([word_attention_scores, entity_attention_scores], dim=2)

        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        # attention_scores = attention_scores + attention_mask

        attention_probs = F.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)

        value_layer = self.transpose_for_scores(
            self.value(torch.cat([word_hidden_states, entity_hidden_states], dim=1))
        )
        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        return context_layer[:, :word_size, :], context_layer[:, word_size:, :]


class EntityAwareAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.self = EntityAwareSelfAttention(config)
        self.output = BertSelfOutput(config)

    # def forward(self, word_hidden_states, entity_hidden_states, attention_mask):
    def forward(self, word_hidden_states, entity_hidden_states):
        # word_self_output, entity_self_output = self.self(word_hidden_states, entity_hidden_states, attention_mask)
        word_self_output, entity_self_output = self.self(word_hidden_states, entity_hidden_states)

        hidden_states = torch.cat([word_hidden_states, entity_hidden_states], dim=1)
        self_output = torch.cat([word_self_output, entity_self_output], dim=1)
        output = self.output(self_output, hidden_states)
        return output[:, : word_hidden_states.size(1), :], output[:, word_hidden_states.size(1):, :]


class EntityAwareLayer(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.attention = EntityAwareAttention(config)
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

    def forward(self, word_hidden_states, entity_hidden_states):
        word_attention_output, entity_attention_output = self.attention(
            # word_hidden_states, entity_hidden_states, attention_mask
            word_hidden_states, entity_hidden_states

        )
        attention_output = torch.cat([word_attention_output, entity_attention_output], dim=1)
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)

        return layer_output[:, : word_hidden_states.size(1), :], layer_output[:, word_hidden_states.size(1):, :]


class EntityAwareEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layer = nn.ModuleList([EntityAwareLayer(config) for _ in range(config.num_hidden_layers)])

    def forward(self, word_hidden_states, entity_hidden_states):
        for layer_module in self.layer:
            word_hidden_states, entity_hidden_states = layer_module(
                # word_hidden_states, entity_hidden_states, attention_mask
                word_hidden_states, entity_hidden_states

            )
        return word_hidden_states, entity_hidden_states


class KRQEntityAwareAttentionModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.word_embedding = WordEmbeddings(config)
        self.entity_embeddings = EntityEmbeddings(config)
        self.encoder = EntityAwareEncoder(config)

    def forward(self, text_inputs, query_entity):
        word_embeddings = self.word_embedding(text_inputs)
        word_embeddings = word_embeddings.last_hidden_state[:, 1:-1]
        entity_embeddings = self.entity_embeddings(query_entity)
        encoder = self.encoder(word_embeddings, entity_embeddings)
        return encoder

    # def load_state_dict(self, state_dict, *args, **kwargs):
    #     new_state_dict = state_dict.copy()
    #
    #     for num in range(self.config.num_hidden_layers):
    #         for attr_name in ("weight", "bias"):
    #             if f"encoder.layer.{num}.attention.self.w2e_query.{attr_name}" not in state_dict:
    #                 new_state_dict[f"encoder.layer.{num}.attention.self.w2e_query.{attr_name}"] = state_dict[
    #                     f"encoder.layer.{num}.attention.self.query.{attr_name}"
    #                 ]
    #             if f"encoder.layer.{num}.attention.self.e2w_query.{attr_name}" not in state_dict:
    #                 new_state_dict[f"encoder.layer.{num}.attention.self.e2w_query.{attr_name}"] = state_dict[
    #                     f"encoder.layer.{num}.attention.self.query.{attr_name}"
    #                 ]
    #             if f"encoder.layer.{num}.attention.self.e2e_query.{attr_name}" not in state_dict:
    #                 new_state_dict[f"encoder.layer.{num}.attention.self.e2e_query.{attr_name}"] = state_dict[
    #                     f"encoder.layer.{num}.attention.self.query.{attr_name}"
    #                 ]
    #
    #     kwargs["strict"] = False
    #     super().load_state_dict(new_state_dict, *args, **kwargs)


class A(nn.Module):
    def __init__(self, config):
        super().__init__()
        one_of_second_hidden_size = int(config.hidden_size / 2)
        self.linear1 = nn.Linear(1, one_of_second_hidden_size, bias=False)
        self.tanh = nn.Tanh()
        self.linear2 = nn.Linear(one_of_second_hidden_size, config.hidden_size, bias=False)
        self.softMax = nn.Softmax(dim=-1)

    def forward(self, text_hidden_representation, entity_hidden_representation):
        Query_hidden_representation_concat = torch.cat((text_hidden_representation, entity_hidden_representation),
                                                       dim=1)
        Query_hidden_representation_concat_transpose = torch.transpose(Query_hidden_representation_concat, dim0=1,
                                                                       dim1=-1)
        linear2_output = torch.matmul(self.linear2.weight.t(), Query_hidden_representation_concat_transpose)
        tanh_output = self.tanh(linear2_output)
        linear1_output = torch.matmul(self.linear1.weight.t(), tanh_output)
        softMax_output = self.softMax(linear1_output)
        return softMax_output


class KnowledgeAwareTextMatching(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.Hidden_representation = KRQEntityAwareAttentionModel(config)
        self.A = A(config)
        self.cos_similarity = nn.CosineSimilarity(dim=2)

    def forward(self, query_text_inputs, query_entity, doc_text_inputs, doc_entity):
        Query_text_hidden_representation, Query_entity_hidden_representation = self.Hidden_representation(
            query_text_inputs, query_entity)
        doc_text_hidden_representation, doc_entity_hidden_representation = self.Hidden_representation(doc_text_inputs,
                                                                                                      doc_entity)
        Query_A = self.A(Query_text_hidden_representation, Query_entity_hidden_representation)
        Query_y = torch.matmul(Query_A, (
            torch.cat((Query_text_hidden_representation, Query_entity_hidden_representation), dim=1)))
        doc_A = self.A(doc_text_hidden_representation, doc_entity_hidden_representation)
        doc_y = torch.matmul(doc_A,
                             (torch.cat((doc_text_hidden_representation, doc_entity_hidden_representation), dim=1)))
        cos_similarity = self.cos_similarity(Query_y, doc_y)
        return cos_similarity
