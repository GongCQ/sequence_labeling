import os
import torch.nn as nn
import pytorch_pretrained_bert as torch_bert
from utils.config import USE_GPU

class EmbSeqBert(nn.Module):
    def __init__(self, bert_model_path, label_num, bert_trainable=True, bert_layer_index=-1,
                 bert_full_conn_dropout=0, bert_learning_rate=1e-5, bert_weight_decay=0,
                 bert_full_conn_learning_rate=0.003, bert_full_conn_weight_decay=0):
        '''
        :param bert_model_path: the path of the folder which contains bin file, not the path of the bin file.
        :param label_num: the number of labels, exclude the begin and end label.
        '''
        super(EmbSeqBert, self).__init__()
        self.bert_model_path = bert_model_path
        self.label_num = label_num
        self.bert_trainable = bert_trainable
        self.bert_layer_index = bert_layer_index
        self.bert_full_conn_dropout = bert_full_conn_dropout
        self.bert_learning_rate = bert_learning_rate
        self.bert_weight_decay = bert_weight_decay
        self.bert_full_conn_learning_rate = bert_full_conn_learning_rate
        self.bert_full_conn_weight_decay = bert_full_conn_weight_decay
        self.bert_model = torch_bert.BertModel.from_pretrained(bert_model_path)
        self.dropout = nn.Dropout(p=bert_full_conn_dropout)
        self.full_conn = nn.Linear(self.bert_model.config.hidden_size, label_num + 2)
        for param in self.bert_model.parameters():
            param.requires_grad_(bert_trainable)

        if USE_GPU:
            self.bert_model = nn.DataParallel(self.bert_model)
            self.bert_model = self.bert_model.cuda()
            self.dropout = self.dropout.cuda()
            self.full_conn = self.full_conn.cuda()

    def forward(self, seq_ids, mask):
        '''
        :param seq_ids: tensor, shape [batch_size * seq_len], with char id selected in [0, 1, 2, ..., vocab_size - 1]
        :param mask: tensor, shape [batch_size * seq_len], with mask-indicator selected in [0, 1]
        :return: tensor, shape [batch_size, seq_len, emb_size], while emb_size is the hidden_size of bert_model, 768.
        '''
        if self.bert_layer_index == -1:
            char_emb_seq, sen_emb = self.bert_model(input_ids=seq_ids, attention_mask=mask,
                                                    output_all_encoded_layers=False)
        else:
            char_emb_seq_list, sen_emb = self.bert_model(input_ids=seq_ids, attention_mask=mask,
                                                         output_all_encoded_layers=True)
            char_emb_seq = char_emb_seq_list[self.bert_layer_index]

        if self.bert_full_conn_dropout > 0:
            feature_seq = self.full_conn(self.dropout(char_emb_seq) * (1 - self.bert_full_conn_dropout))
        else:
            feature_seq = self.full_conn(char_emb_seq)
        # print('char_emb_seq norm     %s' % str(float(char_emb_seq.abs().mean())))
        # print('feature_seq norm      %s' % str(float(feature_seq.abs().mean())))
        # print('full_conn weight norm %s' % str(float(self.full_conn.weight.abs().mean())))
        # print('full_conn bias norm   %s' % str(float(self.full_conn.bias.abs().mean())))
        return feature_seq

    def get_params_config(self):
        params_config = [{'params': self.full_conn.parameters(),
                          'lr': self.bert_full_conn_learning_rate,
                          'weight_decay': self.bert_full_conn_weight_decay}]
        if self.bert_trainable:
            params_config += [{'params': self.bert_model.parameters(),
                               'lr': self.bert_learning_rate,
                               'weight_decay': self.bert_weight_decay}]
        return params_config