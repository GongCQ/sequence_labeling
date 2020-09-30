import os
import torch.nn as nn
import pytorch_pretrained_bert as torch_bert

class SeqEmbBert(nn.Module):
    def __init__(self, bert_model_path, label_num):
        super(SeqEmbBert, self).__init__()
        self.bert_model_path = bert_model_path
        self.label_num = label_num
        bert_model = torch_bert.BertModel.from_pretrained(bert_model_path)
        self.add_module('bert_model', bert_model)
        self.full_conn = nn.Linear(bert_model.config.hidden_size, label_num + 2)