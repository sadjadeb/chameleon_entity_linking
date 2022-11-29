import transformers
from transformers import BertTokenizer
from transformers.models.bert.modeling_bert import BertConfig
import torch
from model import KnowledgeAwareTextMatching
from loss import NegativeSamplingInBatchLoss

# prevent transformers from printing warnings
transformers.logging.set_verbosity_error()

# Set Hyperparameters
MAX_LEN = 128
BATCH_SIZE = 32
EPOCHS = 5
LEARNING_RATE = 1e-5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_NAME = 'prajjwal1/bert-tiny'

config = BertConfig.from_pretrained(MODEL_NAME, num_attention_heads=32, hidden_size=128)
tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)

config.__dict__['device'] = DEVICE
config.__dict__['output_hidden_states'] = 128
config.__dict__['num_hidden_layers'] = 4
config.__dict__['entity_vocab_size'] = 5
config.__dict__['entity_emb_size'] = 128
config.__dict__['bertModel'] = MODEL_NAME

queryTexts = ["This is a book."]
queryText_IDs = tokenizer(queryTexts, add_special_tokens=True, max_length=MAX_LEN, padding=True, return_tensors="pt",
                          return_token_type_ids=True, return_attention_mask=True, truncation="longest_first",
                          return_special_tokens_mask=False)

docTexts = ["This is a text that is related to book.", "my notebook is very beautiful."]
docText_IDs = tokenizer(docTexts, add_special_tokens=True, max_length=MAX_LEN, padding=True, return_tensors="pt",
                        return_token_type_ids=True, return_attention_mask=True, truncation="longest_first",
                        return_special_tokens_mask=False)
print(queryText_IDs)

queryEntity_IDs = torch.LongTensor([[2, 3]])
docEntity_IDs = torch.LongTensor([[2, 3], [4, 1]])
labels = torch.tensor([1, 0])

model = KnowledgeAwareTextMatching(config)
# print(model)
model.to(config.device)
model.train()

criterion = NegativeSamplingInBatchLoss(DEVICE)
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

for epoch in range(EPOCHS):
    outputs = model(queryText_IDs, queryEntity_IDs, docText_IDs, docEntity_IDs)
    loss = criterion(outputs, labels)
    # optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()

    print(f"loss: {loss.item():>7f}  [{epoch+1:>2d}/{EPOCHS:>2d}]")

# torch.save(model.state_dict(), 'model.pt')
