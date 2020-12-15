import os
import copy
import csv
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from utils.config import USE_GPU
from utils.data import PAD_INDEX
import model.lstm_ as lstm_

class EmbSeqLSTM(nn.Module):
    def __init__(self, emb_array, label_num,
                 emb_trainable=True, emb_max_norm=None, hidden_size=100, dropout=0, bidirectional=False,
                 emb_learning_rate=0.001, lstm_learning_rate=0.01, full_conn_learning_rate=0.01,
                 lstm_weight_decay=0, full_conn_weight_decay=0,
                 default_lstm=True):
        '''
        :param emb_array: a 2-D array with shape [vocab_size, emb_size], which is obtained from data.get_char_emb_array
        :param label_num: the number of labels, exclude the begin and end label.
        '''
        super(EmbSeqLSTM, self).__init__()
        self.emb_trainable = emb_trainable
        self.emb_max_norm = emb_max_norm
        self.label_num = label_num
        self.hidden_size = hidden_size
        self.emb_learning_rate = emb_learning_rate
        self.lstm_learning_rate = lstm_learning_rate
        self.full_conn_learning_rate = full_conn_learning_rate
        self.lstm_weight_decay = lstm_weight_decay
        self.full_conn_weight_decay = full_conn_weight_decay

        self.emb = nn.Embedding.from_pretrained(embeddings=torch.Tensor(emb_array),
                                                padding_idx=PAD_INDEX, max_norm=emb_max_norm)
        self.emb.requires_grad_(emb_trainable)
        self.char_emb_size = emb_array.shape[1]
        LSTM = nn.LSTM if default_lstm else lstm_.LSTM_
        self.lstm = LSTM(input_size=self.char_emb_size, hidden_size=hidden_size,
                            num_layers=1, bias=True, batch_first=True, dropout=dropout,
                            bidirectional=bidirectional)
        self.full_conn = nn.Linear(self.hidden_size * (2 if bidirectional else 1), label_num + 2)

        if USE_GPU:
            self.emb = self.emb.cuda()
            self.lstm = self.lstm.cuda()
            self.full_conn = self.full_conn.cuda()

        print('---- parameters of EmbSeqLSTM ----')
        print('type of LSTM            %s' % type(self.lstm))
        print('default_lstm            %s' % default_lstm)
        print('dropout                 %s' % dropout)
        print('bidirectional           %s' % bidirectional)
        print('emb_trainable           %s' % emb_trainable)
        print('emb_max_norm            %s' % emb_max_norm)
        print('hidden_size             %s' % hidden_size)
        print('emb_learning_rate       %s' % emb_learning_rate)
        print('lstm_learning_rate      %s' % lstm_learning_rate)
        print('full_conn_learning_rate %s' % full_conn_learning_rate)
        print('lstm_weight_decay       %s' % lstm_weight_decay)
        print('full_conn_weight_decay  %s' % full_conn_weight_decay)
        print('----------------------------------')

    def forward(self, seq_ids, mask):
        '''
        :param seq_ids: tensor, shape [batch_size * seq_len], with char id selected in [0, 1, 2, ..., vocab_size - 1]
        :param mask: tensor, shape [batch_size * seq_len], with mask-indicator selected in [0, 1]
        :return: tensor, shape [batch_size, seq_len, emb_size], while emb_size is the hidden_size of lstm.
        '''

        lengths = torch.sum(mask, dim=1, keepdim=False)
        emb_seq_batch = self.emb(seq_ids)
        # ---------------
        # packed_emb_seq_batch = nn.utils.rnn.pack_padded_sequence(emb_seq_batch, lengths,
        #                                                          batch_first=True,
        #                                                          enforce_sorted=False)
        # packed_state_seq, (final_hidden_state, final_cell_state) = self.lstm(packed_emb_seq_batch)
        # state_seq, l = nn.utils.rnn.pad_packed_sequence(packed_state_seq, batch_first=True, padding_value=0)
        # ...............
        state_seq, (final_hidden_state, final_cell_state) = self.lstm(emb_seq_batch)
        # ---------------
        feature_seq = self.full_conn(state_seq)
        return feature_seq

    def get_params_config(self):
        params_config = [{'params': self.full_conn.parameters(),
                          'lr': self.full_conn_learning_rate,
                          'weight_decay': self.full_conn_weight_decay},
                         {'params': self.lstm.parameters(),
                          'lr': self.lstm_learning_rate,
                          'weight_decay': self.lstm_weight_decay}]
        if self.emb_trainable:
            params_config.append({'params': self.emb.parameters(),
                                  'lr': self.emb_learning_rate})
        return params_config