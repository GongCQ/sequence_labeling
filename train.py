import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import sys
import datetime as dt
import torch
import utils.data as data
import utils.metric as metric
import utils.evaluate as evaluate
from model.emb_seq_bert import EmbSeqBert
from model.emb_seq_lstm import EmbSeqLSTM
from model.emb_seq_lattice import LatticeLSTM
from model.seq_label import SeqLabel
import utils.config as config

import codecs
sys.stdout = codecs.getwriter("utf-8")(sys.stdout.detach())

config.print_config()
entity_tags = None # {'LOC', 'PER', 'ORG'}
print('entity_tags %s' % entity_tags)
bert_model_path = './bert_model/pytorch_pretrained_bert/bert-base-chinese/'
tok = data.Tokenizer(path=os.path.join(bert_model_path, 'vocab.txt'))
dsm = data.DataSetManager(path='./data/%s/%s' % (config.DATA_SET, config.LABEL_FORMAT),
                          vocab_path=os.path.join(bert_model_path, 'vocab.txt'),
                          batch_size=config.BATCH_SIZE, shuffle=True, entity_tags=entity_tags)
train_data_set = dsm.train_data_set
met = metric.Metric(label_set=train_data_set.label_tokenizer.label_set,
                    tag_set=train_data_set.tag_set, format=config.LABEL_FORMAT)

MODEL = 'LSTM'
if MODEL == 'BERT':
    print('**** BERT MODEL ****')
    emb_seq_model = EmbSeqBert(bert_model_path=bert_model_path,
                               label_num=train_data_set.label_tokenizer.label_num)
elif MODEL == 'LSTM':
    print('**** LSTM MODEL ****')
    emb_array = data.get_char_emb_array(os.path.join('word_emb', 'char_emb_300d_nl2sql.txt'), tok)
    emb_seq_model = EmbSeqLSTM(emb_array=emb_array, label_num=train_data_set.label_tokenizer.label_num)
elif MODEL == 'LATTICE':
    print('**** LATTICE MODEL ****')
    print('%s begin construct tok and emb.' % dt.datetime.now())
    bidirectional = True
    cut_all = True
    ignore_invalid_word = False
    emb_trainable = True
    emb_max_norm = None
    word_emb_path = os.path.join('word_emb', 'word_emb_200d_tencent_ailab_top_10w.txt')
    char_emb_path = os.path.join('word_emb', 'char_emb_300d_nl2sql.txt')
    print('bidirectional: %s' % bidirectional)
    print('cut_all: %s' % cut_all)
    print('ignore_invalid_word: %s' % ignore_invalid_word)
    print('emb_trainable: %s' % emb_trainable)
    print('emb_max_norm: %s' % emb_max_norm)
    print('word_emb_path: %s' % word_emb_path)
    print('char_emb_path: %s' % char_emb_path)
    word_tok = data.WordTokenizer(word_emb_path=word_emb_path)
    char_emb_array = data.get_char_emb_array(char_emb_path, tokenizer=tok)
    word_emb_array = word_tok.emb_array
    print('%s end construct tok and emb.' % dt.datetime.now())
    emb_seq_model = LatticeLSTM(tokenizer=tok, word_tokenizer=word_tok,
                                cut_all=cut_all, ignore_invalid_word=ignore_invalid_word,
                                word_emb_array=word_emb_array, char_emb_array=char_emb_array,
                                label_num=train_data_set.label_tokenizer.label_num,
                                char_input_size=300, word_input_size=200, hidden_size=100,
                                emb_max_norm=emb_max_norm, emb_trainable=emb_trainable, bidirectional=bidirectional)
else:
    raise Exception('UNKNOWN MODEL %s.' % MODEL)
# -----------------------
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

        if config.USE_GPU:
            seq_ids_batch_t = seq_ids_batch_t.cuda()
            label_ids_batch_t = label_ids_batch_t.cuda()
            seq_ids_mask_batch_t = seq_ids_mask_batch_t.cuda()
            label_ids_mask_batch_t = label_ids_mask_batch_t.cuda()

        loss = seq_label_model.train_batch(seq_ids=seq_ids_batch_t,
                                           label_ids=label_ids_batch_t,
                                           mask=seq_ids_mask_batch_t)
        print('%s epoch %s, batch %s, loss %s.' % (dt.datetime.now(),i, c, loss))
        if c % config.EVAL_BATCH_INTERVAL == 0:
            print('\nepoch %s, batch %s， %s ----------------------------------------' % (i, c, dt.datetime.now()))
            print('loss %s' % str(loss))
            print()
            evaluator.random_eval(config.EVAL_SIZE, print_detail=False, print_eval_test=False)

        # evaluator.eval(seq_ids_batch, label_ids_batch, seq_ids_mask_batch, label_ids_mask_batch, print_detail=False)

        c += 1

    print()
    evaluator.full_eval()
    dsm.on_epoch_end()



