import os
import copy
import random
import csv
from collections import OrderedDict
import numpy as np
import pandas as pd
import pytorch_pretrained_bert as torch_bert
from utils.config import SEQ_MAX_LEN

# ----------------------------------------------------
OUT_LABEL = 'O' # immutable
LABEL_SEP = '-'
PAD_INDEX = 0 # immutable
PAD_TOKEN = '[PAD]' # immutable
UNKNOWN_TOKEN = '[UNK]' # immutable
BEGIN_LABEL = '[B]'
BEGIN_ID = -2
END_LABEL = '[E]'
END_ID = -1
UNKNOWN_LABEL = '[X]'
LABEL_NUM = None
TAG_NUM = None

BEGIN_TOKEN = '[unused1]'
END_TOKEN = '[unused2]'

UNKNOWN_TOKEN_DICT = {'“': '[unused3]',
                      '”': '[unused4]',
                      '‘': '[unused5]',
                      '’': '[unused6]',
                      '—': '[unused7]',
                      '…': '[unused8]'}
UNKNOWN_TOKEN_DICT_INV = {}
[UNKNOWN_TOKEN_DICT_INV.setdefault(v, k) for k, v in UNKNOWN_TOKEN_DICT.items()]
# ----------------------------------------------------

class Tokenizer:
    def __init__(self, path: str):
        '''
        :param path: vocab file path
        '''
        self.path = path
        self.tokenizer = torch_bert.tokenization.BertTokenizer(vocab_file=self.path, max_len=SEQ_MAX_LEN)
        self.pad_index = self.tokenizer.vocab[PAD_TOKEN]

    def encode(self, seq: list):
        for s in range(len(seq)):
            seq[s] = seq[s].lower() # forbidden uppercase
            if seq[s] in UNKNOWN_TOKEN_DICT.keys():
                seq[s] = UNKNOWN_TOKEN_DICT[seq[s]]
            if seq[s] not in self.tokenizer.vocab.keys():
                seq[s] = UNKNOWN_TOKEN
        ids = self.tokenizer.convert_tokens_to_ids(seq)
        # if len(ids) > SEQ_MAX_LEN:
        #     ids = ids[ : SEQ_MAX_LEN]
        # elif len(ids) < SEQ_MAX_LEN:
        #     ids = ids + [self.pad_index] * (SEQ_MAX_LEN - len(ids))
        return ids

    def decode(self, ids: list):
        seq = self.tokenizer.convert_ids_to_tokens(ids)
        for s in range(len(seq)):
            if seq[s] in UNKNOWN_TOKEN_DICT_INV.keys():
                seq[s] = UNKNOWN_TOKEN_DICT_INV[seq[s]]
        return seq

class LabelTokenizer:
    def __init__(self, label_set: set):
        self.label_set = copy.deepcopy(label_set)
        self.label_id_dict = {}
        self.id_label_dict = {}
        for label in self.label_set:
            self.label_id_dict[label] = len(self.label_id_dict) + 1
            self.id_label_dict[len(self.id_label_dict) + 1] = label

        self.label_id_dict[BEGIN_LABEL] = len(label_set) + 2 + BEGIN_ID + 1
        self.label_id_dict[END_LABEL] = len(label_set) + 2 + END_ID + 1
        self.begin_id = self.label_id_dict[BEGIN_LABEL]
        self.end_id = self.label_id_dict[END_LABEL]
        self.id_label_dict[self.begin_id] = BEGIN_LABEL
        self.id_label_dict[self.end_id] = END_LABEL
        self.unknown_id = 0
        self.label_id_dict[UNKNOWN_LABEL] = self.unknown_id
        self.id_label_dict[self.unknown_id] = UNKNOWN_LABEL
        self.label_num = len(self.label_id_dict) - 2

    def encode(self, seq: list):
        ids = [None] * len(seq)
        for i, label in enumerate(seq):
            ids[i] = self.label_id_dict[label]
        # if SEQ_MAX_LEN is not None:
        #     if len(ids) > SEQ_MAX_LEN:
        #         ids = ids[ : SEQ_MAX_LEN]
        #     elif len(ids) < SEQ_MAX_LEN:
        #         ids = ids + [END_ID] * (SEQ_MAX_LEN - len(ids))
        return ids

    def decode(self, ids: list):
        seq = [None] * len(ids)
        for i, id in enumerate(ids):
            seq[i] = self.id_label_dict[id]
        return seq

class DataSet:
    def __init__(self, path: str,
                 tokenizer: Tokenizer, label_tokenizer: LabelTokenizer or None,
                 batch_size: int, shuffle: bool = True):
        '''
        :param path: data file path
        :param label_tokenizer: LabelTokenizer, may be None.
        :param tokenizer: Tokenizer
        '''
        self.path = path
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.shuffle = shuffle

        self.file = open(path, encoding='utf-8')
        self.seq_label_list, self.max_len_dict, self.label_count_dict = self._resolve_seq_label_file(self.file)
        self.tag_set = set([label.split(LABEL_SEP)[1]
                            for label in self.label_count_dict.keys()
                            if label != OUT_LABEL])
        self.label_set = set(self.label_count_dict.keys())
        if label_tokenizer is None:
            self.label_tokenizer = LabelTokenizer(self.label_set)
        else:
            self.label_tokenizer = label_tokenizer

        self._regenerate_samples()

    def _regenerate_samples(self):
        if self.shuffle:
            random.shuffle(self.seq_label_list)
        self.seq_ids_list = []
        self.label_ids_list = []
        for seq_label in self.seq_label_list:
            seq = []
            label = []
            for sl in seq_label:
                seq.append(sl[0])
                label.append(sl[1])
            seq_ids = self.tokenizer.encode(seq)
            label_ids = self.label_tokenizer.encode(label)
            self.seq_ids_list.append(seq_ids)
            self.label_ids_list.append(label_ids)

    def __len__(self):
        return int(np.ceil(len(self.seq_label_list) / self.batch_size))

    def __getitem__(self, batch_index):
        if batch_index >= len(self):
            raise IndexError('index %s out of range' % batch_index)
        seq_ids_batch = self.seq_ids_list[batch_index * self.batch_size :
                                          min(len(self.seq_ids_list), (batch_index + 1) * self.batch_size)]
        label_ids_batch = self.label_ids_list[batch_index * self.batch_size :
                                              min(len(self.seq_ids_list), (batch_index + 1) * self.batch_size)]
        padded_seq_ids_batch, padded_label_ids_batch, seq_ids_mask_batch, label_ids_mask_batch = \
            self._pad_batch(seq_ids_batch=seq_ids_batch, label_ids_batch=label_ids_batch)
        return padded_seq_ids_batch, padded_label_ids_batch, seq_ids_mask_batch, label_ids_mask_batch

    def get_random_batch(self, size):
        size = min(size, len(self.seq_label_list))
        length = len(self.seq_label_list)
        index_list = list(range(length))
        random.shuffle(index_list)
        index_list = index_list[ : size]

        seq_ids_batch = []
        label_ids_batch = []
        for index in index_list:
            seq_ids_batch.append(self.seq_ids_list[index])
            label_ids_batch.append(self.label_ids_list[index])

        padded_seq_ids_batch, padded_label_ids_batch, seq_ids_mask_batch, label_ids_mask_batch = \
            self._pad_batch(seq_ids_batch=seq_ids_batch, label_ids_batch=label_ids_batch)
        return padded_seq_ids_batch, padded_label_ids_batch, seq_ids_mask_batch, label_ids_mask_batch

    def decode_batch(self, padded_seq_ids_batch, padded_label_ids_batch, seq_ids_mask_batch, label_ids_mask_batch):
        # seq_batch = []
        # label_batch = []
        # for padded_seq_ids, padded_label_ids, seq_ids_mask, label_ids_mask in \
        #     zip(padded_seq_ids_batch, padded_label_ids_batch, seq_ids_mask_batch, label_ids_mask_batch):
        #     seq = self.tokenizer.decode(list(np.array(padded_seq_ids)[np.array(seq_ids_mask) == 1]))
        #     label = self.label_tokenizer.decode(list(np.array(padded_label_ids)[np.array(label_ids_mask) == 1]))
        #     seq_batch.append(seq)
        #     label_batch.append(label)

        seq_batch = self.decode_seq_batch(padded_seq_ids_batch, seq_ids_mask_batch)
        label_batch = self.decode_label_batch(padded_label_ids_batch, label_ids_mask_batch)
        return seq_batch, label_batch

    def decode_seq_batch(self, padded_seq_ids_batch, seq_ids_mask_batch):
        seq_batch = []
        for padded_seq_ids, seq_ids_mask in zip(padded_seq_ids_batch, seq_ids_mask_batch):
            seq = self.tokenizer.decode(list(np.array(padded_seq_ids)[np.array(seq_ids_mask) == 1]))
            seq_batch.append(seq)
        return seq_batch

    def decode_label_batch(self, padded_label_ids_batch, label_ids_mask_batch):
        label_batch = []
        for padded_label_ids, label_ids_mask in \
                zip(padded_label_ids_batch, label_ids_mask_batch):
            label = self.label_tokenizer.decode(list(np.array(padded_label_ids)[np.array(label_ids_mask) == 1]))
            label_batch.append(label)
        return label_batch

    def print_batch(self, seq_batch, label_batch):
        for seq, label in zip(seq_batch, label_batch):
            for s, l in zip(seq, label):
                print(s + ' ' + l)
            print('')

    def info(self):
        label_count_str = '\n                  '.join(label + ' ' + str(count)
                                    for label, count
                                    in sorted(self.label_count_dict.items(), key=lambda x: x[0], reverse=False))
        info = 'sequence number : %s' % len(self.seq_label_list) + \
               '\nmax length      : %s' % self.max_len_dict['max_len'] + \
               '\nmax length 97.5 : %s' % self.max_len_dict['max_len_975'] + \
               '\nmax length 99.0 : %s' % self.max_len_dict['max_len_990'] + \
               '\nmax length 99.9 : %s' % self.max_len_dict['max_len_999'] + \
               '\ntag set         : %s' % ' '.join(self.tag_set) + \
               '\nlabel set       : %s' % label_count_str + '\n'
        return info

    def _format_seq(self, seq):
        # --------------------------------------
        # if len(seq) > SEQ_MAX_LEN - 2:
        #     seq = seq[ : SEQ_MAX_LEN - 2]
        # for char_label in seq:
        #     char_label[0] = char_label[0].lower()
        # seq = [[BEGIN_TOKEN, BEGIN_LABEL]] + seq + [[END_TOKEN, END_LABEL]]
        # ......................................
        if len(seq) > SEQ_MAX_LEN:
            seq = seq[ : SEQ_MAX_LEN]
        for char_label in seq:
            char_label[0] = char_label[0].lower()
        # --------------------------------------
        return seq

    def _resolve_seq_label_file(self, file):
        seq_list = []
        seq_len_list = []
        seq = []
        label_count_dict = {}
        for line in file:
            line = line.replace('\n', '')
            if len(line) <= 1:
                seq_len = len(seq)
                seq_len_list.append(seq_len)
                if len(seq) > 0:
                    seq = self._format_seq(seq)
                    seq_list.append(seq)
                seq = []
            else:
                line_split = line.split(' ')
                assert len(line_split) == 2, 'line must be separated by spacing.'
                char = line_split[0]
                label = line_split[1]
                label_count_dict.setdefault(label, 0)
                label_count_dict[label] += 1
                seq.append([char, label])
        sorted_seq_len_list = sorted(seq_len_list, reverse=False)
        loc_990 = int(len(seq_len_list) * 0.99 ) - 1
        loc_975 = int(len(seq_len_list) * 0.975) - 1
        loc_999 = int(len(seq_len_list) * 0.999 ) - 1
        max_len = sorted_seq_len_list[-1]
        max_len_990 = sorted_seq_len_list[loc_990]
        max_len_975 = sorted_seq_len_list[loc_975]
        max_len_999 = sorted_seq_len_list[loc_999]
        max_len_dict = {'max_len'    : max_len,     'max_len_990': max_len_990,
                        'max_len_975': max_len_975, 'max_len_999': max_len_999}
        # if self.shuffle:
        #     random.shuffle(seq_list)
        return seq_list, max_len_dict, label_count_dict

    def _pad(self, ids, pad_index, max_len):
        if len(ids) > max_len:
            mask = [1] * max_len
            ids = ids[ : max_len]
        elif len(ids) < max_len:
            mask = [1] * len(ids) + [0] * (max_len - len(ids))
            ids = ids + [pad_index] * (max_len - len(ids))
        else:
            mask = [1] * len(ids)
        return ids, mask

    def _pad_seq(self, seq_ids, max_len):
        return self._pad(seq_ids, pad_index=self.tokenizer.pad_index, max_len=max_len)

    def _pad_label(self, label_ids, max_len):
        return self._pad(label_ids, pad_index=self.label_tokenizer.unknown_id, max_len=max_len)

    def _pad_batch(self, seq_ids_batch, label_ids_batch):
        max_len = max([len(seq_ids) for seq_ids in seq_ids_batch])
        padded_seq_ids_batch = []
        padded_label_ids_batch = []
        seq_ids_mask_batch = []
        label_ids_mask_batch = []
        for i in range(len(seq_ids_batch)):
            padded_seq_ids, seq_ids_mask = self._pad_seq(seq_ids=seq_ids_batch[i], max_len=max_len)
            padded_label_ids, label_ids_mask = self._pad_label(label_ids=label_ids_batch[i], max_len=max_len)
            padded_seq_ids_batch.append(padded_seq_ids)
            padded_label_ids_batch.append(padded_label_ids)
            seq_ids_mask_batch.append(seq_ids_mask)
            label_ids_mask_batch.append(label_ids_mask)
        return padded_seq_ids_batch, padded_label_ids_batch, seq_ids_mask_batch, label_ids_mask_batch

    def _test_tokenizer(self):
        index = random.randint(0, len(self.seq_label_list) - 1)
        seq_ids = self.seq_ids_list[index]
        label_ids = self.label_ids_list[index]
        seq = self.tokenizer.decode(seq_ids)
        label = self.label_tokenizer.decode(label_ids)
        for i in range(len(self.seq_label_list[index])):
            if self.seq_label_list[index][i][0] != seq[i] or self.seq_label_list[index][i][1] != label[i]:
                print(str(self.seq_label_list[index][i]) + ' ' + seq[i] + ' ' + label[i])

class DataSetManager:
    def __init__(self, path: str, vocab_path: str, batch_size: int, shuffle: bool):
        '''
        :param path: a folder contains three files train.txt, test.txt, valid.txt
        :param vocab_path: a file named vocab.txt
        :param batch_size:
        :param shuffle:
        '''
        self.path = path
        self.vocab_path = vocab_path
        self.batch_size = batch_size
        self.shuffle = shuffle
        train_file_path = os.path.join(path, 'train.txt')
        test_file_path = os.path.join(path, 'test.txt')
        valid_file_path = os.path.join(path, 'valid.txt')
        self.tokenizer = Tokenizer(path=vocab_path)
        self.train_data_set = DataSet(path=train_file_path,
                                      tokenizer=self.tokenizer, label_tokenizer=None,
                                      batch_size=batch_size, shuffle=shuffle)
        self.test_data_set = DataSet(path=test_file_path,
                                     tokenizer=self.tokenizer, label_tokenizer=self.train_data_set.label_tokenizer,
                                     batch_size=batch_size, shuffle=shuffle)
        self.valid_data_set = DataSet(path=valid_file_path,
                                     tokenizer=self.tokenizer, label_tokenizer=self.train_data_set.label_tokenizer,
                                     batch_size=batch_size, shuffle=shuffle)
        print(self.train_data_set.info())

    def on_epoch_end(self):
        self.train_data_set._regenerate_samples()

def get_char_emb_array(char_emb_path, tokenizer: Tokenizer, emb_norm=None):
    char_emb_df = pd.read_csv(char_emb_path, index_col=0, header=None, sep=' ',
                              error_bad_lines=False, quoting=csv.QUOTE_NONE, encoding='utf-8')
    vocab_size = len(tokenizer.tokenizer.vocab)
    emb_size = char_emb_df.shape[1]
    emb_array = np.zeros([vocab_size, emb_size], dtype=float)
    for char, id in tokenizer.tokenizer.vocab.items():
        if char in char_emb_df.index:
            emb = char_emb_df.loc[char, :].values
            if emb_norm is not None:
                norm = np.linalg.norm(emb)
                emb = (emb_norm * emb / norm) if norm > 0 else emb
            emb_array[id, :] = emb
    return emb_array

class WordTokenizer:
    def __init__(self, word_vocab_path, emb_norm=1):
        self.word_vocab_path = word_vocab_path
        self.word_id_dict = {}
        self.pad_token = '\000'
        self.unknown_token = '\001'
        self.word_id_dict['[PAD]'] = 0
        self.word_id_dict['[UNK]'] = 1
        file = open(word_vocab_path, encoding='utf-8')
        for line in file:
            line = line.strip()
            if len(line) >= 1:
                word = line.split(' ')[0]
                self.word_id_dict[word] = len(self.word_id_dict)
        self.word_id_pair_list = list(self.word_id_dict.items())

    def get_id(self, word):
        return self.word_id_dict.get(word, 1)

    def get_word(self, id):
        return self.word_id_pair_list[id][0] if id < len(self.word_id_pair_list) else self.unknown_token

    def vocab_size(self):
        return len(self.word_id_pair_list)

def get_word_emb_array(word_emb_path, tokenizer: WordTokenizer, emb_norm=None):
    word_emb_df = pd.read_csv(word_emb_path, index_col=0, header=None, sep=' ',
                              error_bad_lines=False, quoting=csv.QUOTE_NONE, encoding='utf-8')
    vocab_size = tokenizer.vocab_size()
    emb_size = word_emb_df.shape[1]
    emb_array = np.zeros([vocab_size, emb_size], dtype=float)
    for word, id in tokenizer.word_id_dict.items():
        if word in word_emb_df.index:
            emb = word_emb_df.loc[word, :].values
            if emb_norm is not None:
                norm = np.linalg.norm(emb)
                emb = (emb_norm * emb / norm) if norm > 0 else emb
            emb_array[id, :] = emb
    return emb_array


