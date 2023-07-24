import torch

# Define the BiEncoder model
class BiEncoder(torch.nn.Module):
    def __init__(self, encoder):
        super(BiEncoder, self).__init__()
        self.encoder = encoder

    def forward(self, input_ids, attention_mask):
        return self.encoder(input_ids=input_ids, attention_mask=attention_mask)['pooler_output']
