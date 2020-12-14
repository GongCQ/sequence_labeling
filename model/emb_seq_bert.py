import os
import torch.nn as nn
import pytorch_pretrained_bert as torch_bert
from utils.config import USE_GPU, BERT_TRAINABLE, BERT_LAYER_INDEX, BERT_LEARNING_RATE, BERT_WEIGHT_DECAY
from utils.config import BERT_FULL_CONN_DROP_OUT, BERT_FULL_CONN_LEARNING_RATE, BERT_FULL_CONN_WEIGHT_DECAY

class EmbSeqBert(nn.Module):
    def __init__(self, bert_model_path, label_num):
        '''
        :param bert_model_path: the path of the folder which contains bin file, not the path of the bin file.
        :param label_num: the number of labels, exclude the begin and end label.
        '''
        super(EmbSeqBert, self).__init__()
        self.bert_model_path = bert_model_path
        self.label_num = label_num
        self.bert_model = torch_bert.BertModel.from_pretrained(bert_model_path)
        self.dropout = nn.Dropout(p=BERT_FULL_CONN_DROP_OUT)
        self.full_conn = nn.Linear(self.bert_model.config.hidden_size, label_num + 2)
        for param in self.bert_model.parameters():
            param.requires_grad  = BERT_TRAINABLE

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
        if BERT_LAYER_INDEX == -1:
            char_emb_seq, sen_emb = self.bert_model(input_ids=seq_ids, attention_mask=mask,
                                                    output_all_encoded_layers=False)
        else:
            char_emb_seq_list, sen_emb = self.bert_model(input_ids=seq_ids, attention_mask=mask,
                                                         output_all_encoded_layers=True)
            char_emb_seq = char_emb_seq_list[BERT_LAYER_INDEX]

        if BERT_FULL_CONN_DROP_OUT > 0:
            feature_seq = self.full_conn(self.dropout(char_emb_seq) * (1 - BERT_FULL_CONN_DROP_OUT))
        else:
            feature_seq = self.full_conn(char_emb_seq)
        # print('char_emb_seq norm     %s' % str(float(char_emb_seq.abs().mean())))
        # print('feature_seq norm      %s' % str(float(feature_seq.abs().mean())))
        # print('full_conn weight norm %s' % str(float(self.full_conn.weight.abs().mean())))
        # print('full_conn bias norm   %s' % str(float(self.full_conn.bias.abs().mean())))
        return feature_seq

    def get_params_config(self):
        params_config = [{'params': self.full_conn.parameters(),
                          'lr': BERT_FULL_CONN_LEARNING_RATE,
                          'weight_decay': BERT_FULL_CONN_WEIGHT_DECAY}]
        if BERT_TRAINABLE:
            params_config += [{'params': self.bert_model.parameters(),
                               'lr': BERT_LEARNING_RATE,
                               'weight_decay': BERT_WEIGHT_DECAY}]
        return params_config