import os
import torch.nn as nn
import pytorch_pretrained_bert as torch_bert

class EmbSeqBert(nn.Module):
    def __init__(self, bert_model_path, label_num, trainable: bool = True, gpu: bool = False):
        '''
        :param bert_model_path: the path of the folder which contains bin file, not the path of the bin file.
        :param label_num: the number of labels, exclude the begin and end label.
        :param gpu: use gpu or not.
        '''
        super(EmbSeqBert, self).__init__()
        self.bert_model_path = bert_model_path
        self.label_num = label_num
        self.trainable = trainable
        self.gpu = gpu
        self.bert_model = torch_bert.BertModel.from_pretrained(bert_model_path)
        self.dropout = nn.Dropout(p=0.5)
        self.full_conn = nn.Linear(self.bert_model.config.hidden_size, label_num + 2)
        for param in self.bert_model.parameters():
            param.requires_grad  = trainable

    def forward(self, seq_ids, mask):
        '''
        :param seq_ids: tensor, shape [batch_size * seq_len], with char id selected in [0, 1, 2, ..., vocab_size - 1]
        :param mask: tensor, shape [batch_size * seq_len], with mask-indicator selected in [0, 1]
        :return: tensor, shape [batch_size, seq_len, emb_size], while emb_size is the hidden_size of bert_model, 768.
        '''
        char_emb_seq, sen_emb = self.bert_model(input_ids=seq_ids, attention_mask=mask,
                                                output_all_encoded_layers=False)
        feature_seq = self.full_conn(self.dropout(char_emb_seq))
        print('char_emb_seq norm     %s' % str(float(char_emb_seq.abs().mean())))
        print('feature_seq norm      %s' % str(float(feature_seq.abs().mean())))
        print('full_conn weight norm %s' % str(float(self.full_conn.weight.abs().mean())))
        print('full_conn bias norm   %s' % str(float(self.full_conn.bias.abs().mean())))
        return feature_seq