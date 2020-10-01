import os
import torch
import utils.data as data
import utils.metric as metric
from model.emb_seq_bert import EmbSeqBert
from model.seq_label import SeqLabel

format = 'bio'
bert_model_path = './bert_model/pytorch_pretrained_bert/bert-base-chinese/'
tok = data.Tokenizer(path=os.path.join(bert_model_path, 'vocab.txt'))
train_data_set = data.DataSet(path='./data/tiny_data/%s/train.txt' % format, tokenizer=tok, batch_size=20, shuffle=False)
met = metric.Metric(label_set=train_data_set.label_tokenizer.label_set,
                    tag_set=train_data_set.tag_set, format=format)
emb_seq_model = EmbSeqBert(bert_model_path=bert_model_path,
                           label_num=train_data_set.label_tokenizer.label_num,
                           trainable=True)
seq_label_model = SeqLabel(emb_seq_model=emb_seq_model,
                           label_num=train_data_set.label_tokenizer.label_num)

for i in range(50):
    c = 0
    for seq_ids_batch, label_ids_batch, seq_ids_mask_batch, label_ids_mask_batch in train_data_set:
        print('\n~~ %s %s' % (i, c))
        seq_ids_batch = torch.Tensor(seq_ids_batch).to(torch.int64)
        label_ids_batch = torch.Tensor(label_ids_batch).to(torch.int64)
        seq_ids_mask_batch = torch.Tensor(seq_ids_mask_batch).to(torch.bool)
        label_ids_mask_batch = torch.Tensor(label_ids_mask_batch).to(torch.bool)
        loss = seq_label_model.train_batch(seq_ids=seq_ids_batch, label_ids=label_ids_batch, mask=seq_ids_mask_batch)
        print(loss)
        c += 1



