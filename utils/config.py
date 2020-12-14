import torch

USE_GPU = torch.cuda.is_available()
SEQ_MAX_LEN = 250
DATA_SET = 'msra_ner' # 'tiny_data' 'people_daily_1998'  'msra_ner'
LABEL_FORMAT = 'bio'

EPOCH_NUM = 50
BATCH_SIZE = 64
EVAL_BATCH_INTERVAL = 100
EVAL_SIZE = 100

LEARNING_RATE = 0.1
MOMENTUM = 0
WEIGHT_DECAY = 1e-8

BERT_TRAINABLE = True
BERT_LAYER_INDEX = 11 # -1, 0 ~ 11
BERT_FULL_CONN_DROP_OUT = 0
BERT_LEARNING_RATE = 1e-5
BERT_WEIGHT_DECAY = 0
BERT_FULL_CONN_LEARNING_RATE = 0.003
BERT_FULL_CONN_WEIGHT_DECAY = 0.01

def print_config():
    format_str = '%-28s: %s'
    print('==== PARAM ====')
    print(format_str % ('USE_GPU', USE_GPU))
    print(format_str % ('SEQ_MAX_LEN', SEQ_MAX_LEN))
    print(format_str % ('DATA_SET', DATA_SET))
    print(format_str % ('LABEL_FORMAT', LABEL_FORMAT))
    print()
    print(format_str % ('EPOCH_NUM', EPOCH_NUM))
    print(format_str % ('BATCH_SIZE', BATCH_SIZE))
    print(format_str % ('EVAL_BATCH_INTERVAL', EVAL_BATCH_INTERVAL))
    print(format_str % ('EVAL_SIZE', EVAL_SIZE))
    print()
    print(format_str % ('LEARNING_RATE', LEARNING_RATE))
    print(format_str % ('MOMENTUM', MOMENTUM))
    print(format_str % ('WEIGHT_DECAY', WEIGHT_DECAY))
    print()
    print(format_str % ('BERT_TRAINABLE', BERT_TRAINABLE))
    print(format_str % ('BERT_LAYER_INDEX', BERT_LAYER_INDEX))
    print(format_str % ('BERT_FULL_CONN_DROP_OUT', BERT_FULL_CONN_DROP_OUT))
    print(format_str % ('BERT_LEARNING_RATE', BERT_LEARNING_RATE))
    print(format_str % ('BERT_WEIGHT_DECAY', BERT_WEIGHT_DECAY))
    print(format_str % ('BERT_FULL_CONN_LEARNING_RATE', BERT_FULL_CONN_LEARNING_RATE))
    print(format_str % ('BERT_FULL_CONN_WEIGHT_DECAY', BERT_FULL_CONN_WEIGHT_DECAY))
    print('==== PARAM ====')