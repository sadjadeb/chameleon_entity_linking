import torch
from torch import nn
from transformers import LukeModel, LukeTokenizer


class KnowledgeAwareTextMatching(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.device = device
        self.language_model = LukeModel.from_pretrained("studio-ousia/luke-base")
        self.cos_similarity = nn.CosineSimilarity()

    def forward(self, queries_inputs, docs_inputs, phase):
        queries_inputs = {k: v.squeeze().to(self.device) for k, v in queries_inputs.items()}
        docs_inputs = {k: v.squeeze().to(self.device) for k, v in docs_inputs.items()}

        queries_outputs = self.language_model(**queries_inputs)
        queries_representation = queries_outputs.pooler_output
        docs_outputs = self.language_model(**docs_inputs)
        docs_representation = docs_outputs.pooler_output

        if phase == 'train':
            batch_size = queries_representation.shape[0]
            queries_representation = queries_representation.repeat_interleave(batch_size, dim=0)
            docs_representation = torch.cat(batch_size * [docs_representation])
            cos_similarity = self.cos_similarity(queries_representation, docs_representation)
            cos_similarity = cos_similarity.view(batch_size, batch_size)
        else:
            cos_similarity = self.cos_similarity(queries_representation, docs_representation)

        return cos_similarity
