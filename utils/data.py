import os
import copy
import random
import utils
import pytorch_pretrained_bert as torch_bert

OUT_LABEL = 'O'
LABEL_SEP = '-'
SEQ_MAX_LEN = 50
PAD_TOKEN = '[PAD]' # 不可修改
UNKNOWN_TOKEN = '[UNK]' # 不可修改
BEGIN_LABEL = '[B]'
BEGIN_ID = -2
END_LABEL = '[E]'
END_ID = -1

UNKNOWN_TOKEN_DICT = {'“': '[unused1]',
                      '”': '[unused2]',
                      '‘': '[unused3]',
                      '’': '[unused4]',
                      '—': '[unused5]',
                      '…': '[unused6]'}
UNKNOWN_TOKEN_DICT_INV = {}
[UNKNOWN_TOKEN_DICT_INV.setdefault(v, k) for k, v in UNKNOWN_TOKEN_DICT.items()]

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
        if len(ids) > SEQ_MAX_LEN:
            ids = ids[ : SEQ_MAX_LEN]
        elif len(ids) < SEQ_MAX_LEN:
            ids = ids + [self.pad_index] * (SEQ_MAX_LEN - len(ids))
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
            self.label_id_dict[label] = len(self.label_id_dict)
            self.id_label_dict[len(self.id_label_dict)] = label
        self.label_id_dict[BEGIN_LABEL] = len(label_set) + 2 + BEGIN_ID
        self.id_label_dict[BEGIN_ID] = BEGIN_LABEL
        self.label_id_dict[END_LABEL] = len(label_set) + 2 + END_ID
        self.id_label_dict[END_ID] = END_LABEL

    def encode(self, seq: list):
        ids = [None] * len(seq)
        for i, label in enumerate(seq):
            ids[i] = self.label_id_dict[label]
        if SEQ_MAX_LEN is not None:
            if len(ids) > SEQ_MAX_LEN:
                ids = ids[ : SEQ_MAX_LEN]
            elif len(ids) < SEQ_MAX_LEN:
                ids = ids + [END_ID] * (SEQ_MAX_LEN - len(ids))
        return ids

    def decode(self, ids: list):
        seq = [None] * len(ids)
        for i, id in enumerate(ids):
            seq[i] = self.id_label_dict[id]
        return seq

class DataSet:
    def __init__(self, path: str, tokenizer: Tokenizer, batch_size: int):
        '''
        :param path: data file path
        :param tokenizer: Tokenizer
        '''
        self.path = path
        self.file = open(path, encoding='utf-8')
        self.seq_label_list, self.max_len_dict, self.label_count_dict = self._resolve_seq_label_file(self.file)
        self.tag_set = set([label.split(LABEL_SEP)[1]
                            for label in self.label_count_dict.keys()
                            if label != OUT_LABEL])
        self.tokenizer = tokenizer
        self.label_tokenizer = LabelTokenizer(set(self.label_count_dict.keys()))
        self.batch_size = batch_size

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

    def info(self):
        label_count_str = '; '.join(label + ' ' + str(count)
                                    for label, count
                                    in sorted(self.label_count_dict.items(), key=lambda x: x[0], reverse=False))
        info = 'sequence number   : %s' % len(self.seq_label_list) + \
               '\nmax length      : %s' % self.max_len_dict['max_len'] + \
               '\nmax length 99.0 : %s' % self.max_len_dict['max_len_990'] + \
               '\nmax length 97.5 : %s' % self.max_len_dict['max_len_975'] + \
               '\nmax length 95.0 : %s' % self.max_len_dict['max_len_950'] + \
               '\ntag set         : %s' % ' '.join(self.tag_set) + \
               '\nlabel set       : %s' % label_count_str + '\n'
        return info

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
                    if len(seq) > SEQ_MAX_LEN:  # limit sequence length while reading data.
                        seq = seq[ : SEQ_MAX_LEN]
                    seq_list.append(seq)
                seq = []
            else:
                line_split = line.split(' ')
                assert len(line_split) == 2, 'line must be separated by spacing.'
                char = line_split[0].lower() # limit to lowercase while reading data.
                label = line_split[1]
                label_count_dict.setdefault(label, 0)
                label_count_dict[label] += 1
                seq.append([char, label])
        sorted_seq_len_list = sorted(seq_len_list, reverse=False)
        loc_990 = int(len(seq_len_list) * 0.99 ) - 1
        loc_975 = int(len(seq_len_list) * 0.975) - 1
        loc_950 = int(len(seq_len_list) * 0.95 ) - 1
        max_len = sorted_seq_len_list[-1]
        max_len_990 = sorted_seq_len_list[loc_990]
        max_len_975 = sorted_seq_len_list[loc_975]
        max_len_950 = sorted_seq_len_list[loc_950]
        max_len_dict = {'max_len'    : max_len,     'max_len_990': max_len_990,
                        'max_len_975': max_len_975, 'max_len_950': max_len_950}
        return seq_list, max_len_dict, label_count_dict

    def _test_tokenizer(self):
        index = random.randint(0, len(self.seq_label_list) - 1)
        seq_ids = self.seq_ids_list[index]
        label_ids = self.label_ids_list[index]
        seq = self.tokenizer.decode(seq_ids)
        label = self.label_tokenizer.decode(label_ids)
        for i in range(len(self.seq_label_list[index])):
            if self.seq_label_list[index][i][0] != seq[i] or self.seq_label_list[index][i][1] != label[i]:
                print(str(self.seq_label_list[index][i]) + ' ' + seq[i] + ' ' + label[i])