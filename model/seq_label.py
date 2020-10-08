import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from model.crf import CRF
import utils
from model.emb_seq_bert import EmbSeqBert

class SeqLabel(nn.Module):
    def __init__(self, emb_seq_model, label_num, gpu: bool = False):
        super(SeqLabel, self).__init__()
        self.emb_seq_model = emb_seq_model
        self.label_num = label_num
        self.crf = CRF(label_num, gpu=gpu)

        # self.optimizer = optim.SGD(self.parameters(), lr=0.01, momentum=0,weight_decay=1e-8)
        # weight_decay越大，参数值越倾向于变小
        self.optimizer = optim.SGD(params=[
                                           {'params': self.emb_seq_model.bert_model.parameters(),
                                            'lr': 0.00001,
                                            'weight_decay': 0},
                                           {'params': self.emb_seq_model.full_conn.parameters(),
                                            'lr': 0.003,
                                            'weight_decay': 0.01},
                                           {'params': self.crf.parameters()}],
                                   lr=0.1, momentum=0,weight_decay=1e-8)

    def forward(self, seq_ids, mask):
        feature_seq = self.emb_seq_model(seq_ids=seq_ids, mask=mask)
        path_score, label_ids = self.crf._viterbi_decode(feats=feature_seq, mask=mask)
        label_ids.squeeze()
        return path_score, label_ids

    def loss(self, seq_ids, label_ids, mask):
        # neg_log_likelihood_loss(self, feats, mask, tags)
        feature_seq = self.emb_seq_model(seq_ids=seq_ids, mask=mask)
        score = self.crf.neg_log_likelihood_loss(feats=feature_seq, mask=mask, tags=label_ids)
        return score / seq_ids.shape[0]

    def train_batch(self, seq_ids, label_ids, mask):

        loss = self.loss(seq_ids, label_ids, mask)
        loss_value = float(loss)
        if loss_value > utils.MAX_LOSS:
            raise Exception('loss %s exceed the MAX_LOSS %s' % (loss, utils.MAX_LOSS))
        loss.backward()
        self.optimizer.step()
        self.zero_grad()
        return loss_value
