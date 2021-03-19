# -*- coding: utf-8 -*-
"""
"""
from transformers import BertModel, BertTokenizer
from transformers import RobertaModel, RobertaTokenizer
import torch
import numpy as np
np.random.seed(42)
import os
path = os.path.join(os.path.dirname(__file__), 'models')

MAX_LEN = 500
model_name = 'bert-base-uncased'
if model_name == 'roberta-large':
    tokenizer = RobertaTokenizer.from_pretrained(model_name, do_lower_case = True)
else:
    tokenizer = BertTokenizer.from_pretrained(model_name, do_lower_case=True)

if model_name == 'roberta-large':
    transformer_model = RobertaModel.from_pretrained(model_name)
else:
    transformer_model = BertModel.from_pretrained(model_name)

def transformer_tokenizer(content, evidence, tokenizer):
    tokenizer_op = tokenizer.encode_plus(content, evidence,
                            max_length=MAX_LEN,
                            padding = 'max_length',
                            truncation = 'longest_first',
                            return_token_type_ids = True,
                            return_attention_mask = True)
    return torch.tensor(tokenizer_op['input_ids']), torch.tensor(tokenizer_op['attention_mask']), torch.tensor(tokenizer_op['token_type_ids'])

class Classifier(torch.nn.Module):
    def __init__(self, transformer_model):
        super().__init__()
        self.transformer = transformer_model
        self.out = torch.nn.Linear(768, 5)

    def forward(self, input_ids, attention_mask, segment_id, classification_labels=None):
        hidden = self.transformer(input_ids = input_ids, attention_mask = attention_mask, token_type_ids = segment_id)
        token_hidden = hidden[1]
        classification_logits = self.out(token_hidden)
        outputs = [classification_logits]
        if classification_labels is not None:
            loss_fct_classification = torch.nn.CrossEntropyLoss()
            loss_classification = loss_fct_classification(classification_logits.view(-1, 5), classification_labels.view(-1))

            outputs += [loss_classification]

        return outputs

def predict_type(model, test_input_ids, test_attention_masks, test_input_types):
    model.eval()
    with torch.no_grad():
        outputs = model(test_input_ids, test_attention_masks, test_input_types)
    classification_logits = outputs[0].detach().cpu().numpy()
    prediction = list(np.argmax(classification_logits,axis=1))
    scores = np.exp(classification_logits).tolist()
    
    return prediction.pop(), scores.pop()

def nli_prediction(content, evidence):
    model = Classifier(transformer_model)    
    model.load_state_dict(torch.load(os.path.join(path, 'best-nli-bert-model.pt'), map_location='cpu'), strict = False)
    
    test_input_ids, test_attention_masks, test_input_types = transformer_tokenizer(content, evidence, tokenizer)
    test_input_ids = test_input_ids.unsqueeze(1).reshape(1,MAX_LEN)
    test_attention_masks = test_attention_masks.unsqueeze(1).reshape(1,MAX_LEN)
    test_input_types = test_input_types.unsqueeze(1).reshape(1,MAX_LEN)

    pred, score = predict_type(model, test_input_ids, test_attention_masks, test_input_types)
    print(pred,score)
    return pred, max(score), np.mean(score)
