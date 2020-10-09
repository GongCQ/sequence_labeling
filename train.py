import os
import datetime as dt
import torch
import utils.data as data
import utils.metric as metric
import utils.evaluate as evaluate
from model.emb_seq_bert import EmbSeqBert
from model.seq_label import SeqLabel
import utils.config as config

bert_model_path = './bert_model/pytorch_pretrained_bert/bert-base-chinese/'
tok = data.Tokenizer(path=os.path.join(bert_model_path, 'vocab.txt'))
dsm = data.DataSetManager(path='./data/%s/%s' % (config.DATA_SET, config.LABEL_FORMAT),
                          vocab_path=os.path.join(bert_model_path, 'vocab.txt'),
                          batch_size=config.BATCH_SIZE, shuffle=True)
train_data_set = dsm.train_data_set
met = metric.Metric(label_set=train_data_set.label_tokenizer.label_set,
                    tag_set=train_data_set.tag_set, format=config.LABEL_FORMAT)

emb_seq_model = EmbSeqBert(bert_model_path=bert_model_path,
                           label_num=train_data_set.label_tokenizer.label_num)
seq_label_model = SeqLabel(emb_seq_model=emb_seq_model,
                           label_num=train_data_set.label_tokenizer.label_num)

evaluator = evaluate.Evaluator(model=seq_label_model, data_set_manager=dsm, metric=met)

for i in range(config.EPOCH_NUM):
    c = 0
    for seq_ids_batch, label_ids_batch, seq_ids_mask_batch, label_ids_mask_batch in train_data_set:
        seq_ids_batch_t = torch.Tensor(seq_ids_batch).to(torch.int64)
        label_ids_batch_t = torch.Tensor(label_ids_batch).to(torch.int64)
        seq_ids_mask_batch_t = torch.Tensor(seq_ids_mask_batch).to(torch.bool)
        label_ids_mask_batch_t = torch.Tensor(label_ids_mask_batch).to(torch.bool)
        loss = seq_label_model.train_batch(seq_ids=seq_ids_batch_t,
                                           label_ids=label_ids_batch_t,
                                           mask=seq_ids_mask_batch_t)

        if c % config.EVAL_BATCH_INTERVAL == 0:
            print('\nepoch %s, batch %s， %s ----------------------------------------' % (i, c, dt.datetime.now()))
            print('loss %s' % str(loss))
            print()
            evaluator.random_eval(config.EVAL_SIZE, print_detail=False)

        # evaluator.eval(seq_ids_batch, label_ids_batch, seq_ids_mask_batch, label_ids_mask_batch, print_detail=False)

        c += 1

    print()
    evaluator.full_eval()
    dsm.on_epoch_end()



