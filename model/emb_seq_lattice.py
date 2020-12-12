import os
import warnings
import datetime as dt
import pandas as pd
import jieba
import torch
import torch.nn as nn
from utils.config import *
from utils.data import Tokenizer, WordTokenizer, get_char_emb_array

pd.set_option('display.max_rows', 1000)
pd.set_option('display.max_columns', 1000)
# pd.set_option('display.max_colwidth',50)
pd.set_option('expand_frame_repr', False)
pd.set_option('display.unicode.ambiguous_as_wide', True)
pd.set_option('display.unicode.east_asian_width', True)

# [s for s in jieba.cut('工信处女干事每月经过下属科室都要亲口交代24口交换机等技术性器件的安装工作',cut_all=True)]
# ['工信处', '处女', '女干事', '干事', '每月', '月经', '经过', '下属', '科室', '都', '要', '亲口', '口交', '交代', '24', '口交', '交换', '交换机', '换机', '等', '技术', '技术性', '性器', '器件', '的', '安装', '安装工', '装工', '工作']


class Lattice:
    def __init__(self, tokenizer: Tokenizer, word_tokenizer: WordTokenizer, cut_all=True, ignore_invalid_word=False):
        self.tokenizer = tokenizer
        self.word_tokenizer = word_tokenizer
        self.cut_all = cut_all
        self.ignore_invalid_word = ignore_invalid_word
        self.search_forward_len = 4

    def to_lattice(self, seq_ids, mask, print_result=False):
        '''
        the input is as same as emb_seq_model.
        :param seq_ids: tensor, shape [batch_size * seq_len], with char id selected in [0, 1, 2, ..., vocab_size - 1]
        :param mask: tensor, shape [batch_size * seq_len], with mask-indicator selected in [0, 1]
        :return: a list in which the element is some triple-tuples such as (word_begin, word_len, word_id)
        '''
        fraction_list_batch = []
        for s, m in zip(seq_ids, mask):
            seq_len = m.sum().item()
            seq_id = s[ : seq_len]
            char_list = self.tokenizer.decode((seq_id.detach().cpu() if USE_GPU else seq_id.detach()).numpy())
            for c, char in enumerate(char_list):
                if len(char) != 1:
                    char_list[c] = self.word_tokenizer.unknown_token
            text = ''.join(char_list)
            seg = [s for s in jieba.cut(text, cut_all=self.cut_all)]
            fraction_list = [(seg[0], 0)] # [word, begin_index]
            cut_all_fail = False
            if self.cut_all:
                for current_word in seg[1 : ]:
                    last_fraction = fraction_list[-1]
                    last_word = last_fraction[0]
                    last_begin = last_fraction[1]
                    last_len = len(last_word)
                    current_max_begin = last_begin + last_len
                    current_len = len(current_word)
                    if current_len == 0:
                        continue
                    current_matched_begin = current_max_begin
                    for i in range(1, max(current_len, last_len) + 1):  # search backward
                        current_try_begin = current_max_begin - i
                        last_end = min(last_len, last_len - i + current_len)
                        current_end = min(current_len, i)
                        if text[current_try_begin : current_try_begin + current_len] == current_word and \
                            last_word[last_len - i : last_end] == current_word[ : current_end] and \
                            (last_len > i or last_len == i and current_len > last_len):
                            current_matched_begin = current_try_begin
                    if text[current_matched_begin : current_matched_begin + current_len] != current_word:
                        for i in range(1, self.search_forward_len):
                            current_matched_begin += 1
                            if text[current_matched_begin: current_matched_begin + current_len] == current_word:
                                break
                    fraction_list.append((current_word, current_matched_begin))
                for fraction in fraction_list:
                    if fraction[0] != text[fraction[1]: fraction[1] + len(fraction[0])]:
                        print('check format error.\n%s\n%s\n%s' % (text, seg, fraction_list))
                        cut_all_fail = True
                        break

            if not self.cut_all or cut_all_fail:
                seg = [s for s in jieba.cut(text, cut_all=False)]
                fraction_list = [(seg[0], 0)]
                accum_len = len(fraction_list[-1][0])
                for current_word in seg[1 : ]:
                    fraction_list.append((current_word, accum_len))
                    accum_len += len(current_word)
                if cut_all_fail:
                    print('cut_all_fail and turn to cut_all=False  %s' % fraction_list)
            fraction_list_batch.append(fraction_list)

            if print_result:
                align_df = pd.DataFrame(data=[[''] * len(text)] * (1 + len(fraction_list)), columns=list(range(len(text))))
                align_df.iloc[0, :] = char_list
                for i in range(len(fraction_list)):
                    fraction = fraction_list[i]
                    for j in range(len(fraction[0])):
                        align_df.iloc[i + 1, fraction[1] + j] = fraction[0][j]
                for col in align_df.columns:
                    col_chars = align_df[col].values
                    if ''.join(col_chars[col_chars != align_df[col].iloc[0]]) != '':
                        warnings.warn('invalid element in align_df.\n%s\n%s' % (text, fraction_list))
                align_df.to_csv('temp/' + text[ : min(50, len(text))] + '.csv')

        lattice_seq_batch = []
        for fraction_list in fraction_list_batch:
            lattice_seq = [(fraction[1], len(fraction[0]), self.word_tokenizer.get_id(fraction[0]))
                           for fraction in fraction_list
                           if (self.word_tokenizer.is_valid_word(fraction[0]) or not self.ignore_invalid_word)]
            lattice_seq_batch.append(lattice_seq)

        return lattice_seq_batch

    def reverse(self, lattice_seq_batch, seq_ids):
        seq_len = seq_ids.shape[1]
        reverse_lattice_seq_batch = []
        for lattice_seq in lattice_seq_batch:
            reverse_lattice_seq = []
            for i, lattice in enumerate(lattice_seq):
                word_begin, word_len, word_id = lattice
                reverse_word_begin = seq_len - word_begin - word_len
                reverse_lattice = (reverse_word_begin, word_len, word_id)
                reverse_lattice_seq.append(reverse_lattice)
            reverse_lattice_seq_batch.append(reverse_lattice_seq)
        return reverse_lattice_seq_batch


class LatticeWordCell(nn.Module):
    def __init__(self, input_size, hidden_size, bias=True):
        '''
        equation(13)
        :param input_size:
        :param hidden_size:
        :param bias:
        '''
        super(LatticeWordCell, self).__init__()
        self.linear_i_i = nn.Linear(input_size, hidden_size, bias=bias)
        self.linear_h_i = nn.Linear(hidden_size, hidden_size, bias=bias)
        self.linear_i_f = nn.Linear(input_size, hidden_size, bias=bias)
        self.linear_h_f = nn.Linear(hidden_size, hidden_size, bias=bias)
        self.linear_i_g = nn.Linear(input_size, hidden_size, bias=bias)
        self.linear_h_g = nn.Linear(hidden_size, hidden_size, bias=bias)
        if USE_GPU:
            self.linear_i_i = self.linear_i_i.cuda()
            self.linear_h_i = self.linear_h_i.cuda()
            self.linear_i_f = self.linear_i_f.cuda()
            self.linear_h_f = self.linear_h_f.cuda()
            self.linear_i_g = self.linear_i_g.cuda()
            self.linear_h_g = self.linear_h_g.cuda()

    def forward(self, x_b_e_w, h_b_c, c_b_c):
        i_b_e_w = torch.sigmoid(self.linear_i_i(x_b_e_w) + self.linear_h_i(h_b_c))
        f_b_e_w = torch.sigmoid(self.linear_i_f(x_b_e_w) + self.linear_h_f(h_b_c))
        g_b_e_w = torch.tanh(self.linear_i_g(x_b_e_w) + self.linear_h_g(h_b_c))
        c_b_e_w = f_b_e_w * c_b_c + i_b_e_w * g_b_e_w
        return c_b_e_w


class LatticeCharCell(nn.Module):
    def __init__(self, input_size, hidden_size, bias=True):
        super(LatticeCharCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.linear_i_i = nn.Linear(input_size, hidden_size, bias=bias)
        self.linear_h_i = nn.Linear(hidden_size, hidden_size, bias=bias)
        self.linear_i_f = nn.Linear(input_size, hidden_size, bias=bias)
        self.linear_h_f = nn.Linear(hidden_size, hidden_size, bias=bias)
        self.linear_i_g = nn.Linear(input_size, hidden_size, bias=bias)
        self.linear_h_g = nn.Linear(hidden_size, hidden_size, bias=bias)
        self.linear_i_o = nn.Linear(input_size, hidden_size, bias=bias)
        self.linear_h_o = nn.Linear(hidden_size, hidden_size, bias=bias)

        # linear_l_x and linear_l_c is the parameters in equation(14)
        self.linear_l_x = nn.Linear(input_size, hidden_size, bias=bias)
        self.linear_l_c = nn.Linear(hidden_size, hidden_size, bias=bias)

        if USE_GPU:
            self.linear_i_i = self.linear_i_i.cuda()
            self.linear_h_i = self.linear_h_i.cuda()
            self.linear_i_f = self.linear_i_f.cuda()
            self.linear_h_f = self.linear_h_f.cuda()
            self.linear_i_g = self.linear_i_g.cuda()
            self.linear_h_g = self.linear_h_g.cuda()
            self.linear_i_o = self.linear_i_o.cuda()
            self.linear_h_o = self.linear_h_o.cuda()

            self.linear_l_x = self.linear_l_x.cuda()
            self.linear_l_c = self.linear_l_c.cuda()

    def forward(self, input, h_c_0, all_c_b_e_w, all_num_word):
        '''
        :param input: current char embedding, also known as x_e_c in equation(14)
        :param h_c_0: tuple, previous hidden state and cell state.
        :param all_c_b_e_w: [batch_size, max_num_word, hidden_size],
                            in which max_num_word is the number of the words those ending at e.
        :param all_num_word: list, [batch_size], indicate how many words is ending at e.
        :return:
        '''
        x_e_c = input  # x_e_c in equation(14)
        h_0, c_0 = h_c_0
        i = torch.sigmoid(self.linear_i_i(input) + self.linear_h_i(h_0))
        f = torch.sigmoid(self.linear_i_f(input) + self.linear_h_f(h_0))
        g = torch.tanh(self.linear_i_g(input) + self.linear_h_g(h_0))
        o = torch.sigmoid(self.linear_i_o(input) + self.linear_h_o(h_0))
        # -----------------
        # c_1 = f * c_0 + i * g
        # .................
        batch_c_j_c = torch.zeros(all_c_b_e_w.shape[0], self.hidden_size)
        if USE_GPU:
            batch_c_j_c = batch_c_j_c.cuda()
        for batch, num_word in enumerate(all_num_word):
            if num_word > 0:
                c_b_e_w = all_c_b_e_w[batch, : num_word, :] # [num_word, hidden_size]
                i_b_e_c = torch.sigmoid(self.linear_l_x(x_e_c[batch, :]) + self.linear_l_c(c_b_e_w)) # [num_word, hidden_size], equation(14)
                exp_i_b_e_c = torch.exp(i_b_e_c) # [num_word, hidden_size]
                sum_exp_i_b_j_c = torch.sum(exp_i_b_e_c, dim=0) # [hidden_size]
                exp_i_j_c = torch.exp(i[batch, :]) # [hidden_size]
                denominator_in_equation_16 = exp_i_j_c + sum_exp_i_b_j_c # [hidden_size]
                alpha_b_j_c = exp_i_b_e_c / denominator_in_equation_16  # [num_word, hidden_size], the first line in equation(16)
                alpha_j_c = exp_i_j_c / denominator_in_equation_16 # [hidden_size], the second line in equation(16)
                c_j_c = torch.sum(alpha_b_j_c * c_b_e_w, dim=0) + alpha_j_c * g[batch, :] # [hidden_size], equation(15)
            else:
                c_j_c = g[batch, :] # try "c_j_c=i[batch,:]*g[batch,:]" or "c_j_c=f*c_0+i[batch,:]*g[batch,:]" here?
            batch_c_j_c[batch, :] = c_j_c
        # -----------------
        h_1 = o * torch.tanh(batch_c_j_c)

        return (h_1, batch_c_j_c)

class LatticeLSTM(nn.Module):
    def __init__(self, tokenizer: Tokenizer, word_tokenizer: WordTokenizer, cut_all, ignore_invalid_word,
                 word_emb_array, char_emb_array, label_num,
                 char_input_size, word_input_size, hidden_size, char_bias=True, word_bias=True,
                 dropout=0, bidirectional=False, emb_max_norm=None, emb_trainable=True,
                 emb_learning_rate=0.001, lstm_learning_rate=0.01, full_conn_learning_rate=0.01):
        super(LatticeLSTM, self).__init__()
        self.lattice = Lattice(tokenizer=tokenizer, word_tokenizer=word_tokenizer,
                               cut_all=cut_all, ignore_invalid_word=ignore_invalid_word)
        self.char_input_size = char_input_size
        self.word_input_size = word_input_size
        self.hidden_size = hidden_size
        self.lattice_word_cell = LatticeWordCell(input_size=word_input_size, hidden_size=hidden_size, bias=word_bias)
        self.lattice_char_cell = LatticeCharCell(input_size=char_input_size, hidden_size=hidden_size, bias=char_bias)
        if bidirectional:
            self.reverse_lattice_word_cell = LatticeWordCell(input_size=word_input_size, hidden_size=hidden_size,
                                                             bias=word_bias)
            self.reverse_lattice_char_cell = LatticeCharCell(input_size=char_input_size, hidden_size=hidden_size,
                                                             bias=char_bias)

        self.dropout_prob = dropout
        self.dropout = nn.Dropout(p=dropout)
        self.bidirectional = bidirectional
        self.emb_trainable = emb_trainable
        self.emb_learning_rate = emb_learning_rate
        self.lstm_learning_rate = lstm_learning_rate
        self.full_conn_learning_rate = full_conn_learning_rate

        self.word_emb = nn.Embedding.from_pretrained(embeddings=torch.Tensor(word_emb_array),
                                                padding_idx=word_tokenizer.pad_id, max_norm=emb_max_norm)

        self.char_emb = nn.Embedding.from_pretrained(embeddings=torch.Tensor(char_emb_array),
                                                padding_idx=tokenizer.pad_index, max_norm=emb_max_norm)

        self.word_emb.requires_grad_(emb_trainable)
        self.char_emb.requires_grad_(emb_trainable)
        self.full_conn = nn.Linear(self.hidden_size * (2 if bidirectional else 1), label_num + 2)

        if USE_GPU:
            self.word_emb = self.word_emb.cuda()
            self.char_emb = self.char_emb.cuda()
            self.lattice_word_cell = self.lattice_word_cell.cuda()
            self.lattice_char_cell = self.lattice_char_cell.cuda()
            self.full_conn = self.full_conn.cuda()
            if bidirectional:
                self.reverse_lattice_word_cell = self.reverse_lattice_word_cell.cuda()
                self.reverse_lattice_char_cell = self.reverse_lattice_char_cell.cuda()

    def _lattice_flow(self, word_cell, char_cell, lattice_seq_batch, seq_ids, mask, char_emb_seq = None):
        batch_size = seq_ids.shape[0]
        seq_len = seq_ids.shape[1]
        end_lattice_dict_seq = [[[] for b in range(batch_size)] for s in range(seq_len)]
        for batch, lattice_seq in enumerate(lattice_seq_batch):
            for word_begin, word_len, word_id in lattice_seq:
                word_end = word_begin + word_len - 1
                if word_len > 1:  # important!!!
                    end_lattice_dict_seq[word_end][batch].append((word_begin, word_len, word_id, batch))

        init_hidden_cell_state = (torch.zeros(batch_size, self.hidden_size), torch.zeros(batch_size, self.hidden_size))
        if USE_GPU:
            init_hidden_cell_state = (init_hidden_cell_state[0].cuda(), init_hidden_cell_state[1].cuda())
        # char_hidden_state = torch.zeros(batch_size, seq_len, self.hidden_size)
        # char_cell_state = torch.zeros(batch_size, seq_len, self.hidden_size)
        char_hidden_list = []
        char_cell_list = []
        for s in range(seq_len):  # sequence direction
            # word lstm cell ...
            end_lattice_batch = end_lattice_dict_seq[s]
            all_num_word = [len(end_lattice_list) for end_lattice_list in end_lattice_batch]

            flatten_end_lattice_batch = sum(end_lattice_batch, [])
            if len(flatten_end_lattice_batch) > 0:
                flatten_batch_batch = [batch for _, _, _, batch in flatten_end_lattice_batch]
                flatten_begin_batch = [word_begin for word_begin, _, _, _ in flatten_end_lattice_batch]
                flatten_word_id_batch = [word_id for _, _, word_id, _ in flatten_end_lattice_batch]
                flatten_word_id_batch = torch.Tensor(flatten_word_id_batch).long()

                flatten_hidden_batch = []
                flatten_cell_batch = []
                for batch, begin in zip(flatten_batch_batch, flatten_begin_batch):
                    flatten_hidden_batch.append(char_hidden_list[begin][batch])
                    flatten_cell_batch.append(char_cell_list[begin][batch])
                flatten_hidden_batch = torch.stack(flatten_hidden_batch, dim=0)
                flatten_cell_batch = torch.stack(flatten_cell_batch, dim=0)
                if USE_GPU:
                    flatten_word_id_batch = flatten_word_id_batch.cuda()
                    flatten_hidden_batch = flatten_hidden_batch.cuda()
                    flatten_cell_batch = flatten_cell_batch.cuda()

                flatten_word_emb_batch = self.word_emb(flatten_word_id_batch)
                flatten_c_b_e_w_batch = word_cell(x_b_e_w=flatten_word_emb_batch,
                                                  h_b_c=flatten_hidden_batch,
                                                  c_b_c=flatten_cell_batch)
            else:
                flatten_c_b_e_w_batch = torch.zeros(0, self.hidden_size).float()
            if USE_GPU:
                flatten_c_b_e_w_batch = flatten_c_b_e_w_batch.cuda()

            # char lstm cell ...
            max_num_word = max(all_num_word)
            all_c_b_e_w = torch.Tensor(batch_size, max_num_word, self.hidden_size)
            if USE_GPU:
                all_c_b_e_w = all_c_b_e_w.cuda()
            begin_in_flatten_batch = 0
            for batch, num_word in enumerate(all_num_word):
                if num_word > 0:
                    all_c_b_e_w[batch, : num_word, :] = \
                        flatten_c_b_e_w_batch[begin_in_flatten_batch: begin_in_flatten_batch + num_word, :]
                begin_in_flatten_batch += num_word
            h_c_0 = (char_hidden_list[s - 1], char_cell_list[s - 1]) if s > 0 else init_hidden_cell_state
            input = self.char_emb(seq_ids[:, s]) if char_emb_seq is None else char_emb_seq[:, s, :]
            h_1, c_j_c = char_cell(input=input, h_c_0=h_c_0, all_c_b_e_w=all_c_b_e_w, all_num_word=all_num_word)
            char_hidden_list.append(h_1)
            char_cell_list.append(c_j_c)

        char_hidden_state = torch.stack(char_hidden_list, dim=1)
        if self.dropout_prob > 0:
            char_hidden_state = self.dropout(char_hidden_state)
        return char_hidden_state

    def forward(self, seq_ids, mask, char_emb_seq = None):
        '''
        :param seq_ids: tensor, [batch_size, seq_len]
        :param mask: tensor, [batch_size, seq_len]
        :param char_emb_seq: tensor, [batch_size, seq_len, char_emb_size]
        :return:
        '''
        # list of list of triple-tuple, [batch_size, num_word_in_seq, (word_begin, word_len, word_id)]
        lattice_seq_batch = self.lattice.to_lattice(seq_ids=seq_ids, mask=mask)
        char_hidden_state = self._lattice_flow(word_cell=self.lattice_word_cell,
                                               char_cell=self.lattice_char_cell,
                                               lattice_seq_batch=lattice_seq_batch, seq_ids=seq_ids, mask=mask,
                                               char_emb_seq=char_emb_seq)

        if self.bidirectional:
            seq_len = seq_ids.shape[1]
            reverse_index = list(range(seq_len))
            reverse_index.reverse()
            reverse_seq_ids = seq_ids[:, reverse_index]
            reverse_mask = mask[:, reverse_index]
            reverse_char_emb_seq = char_emb_seq[:, reverse_index, :] if char_emb_seq is not None else None
            reverse_lattice_seq_batch = self.lattice.reverse(lattice_seq_batch=lattice_seq_batch, seq_ids=seq_ids)
            reverse_char_hidden_state = self._lattice_flow(word_cell=self.reverse_lattice_word_cell,
                                                           char_cell=self.reverse_lattice_char_cell,
                                                           lattice_seq_batch=reverse_lattice_seq_batch,
                                                           seq_ids=reverse_seq_ids, mask=reverse_mask,
                                                           char_emb_seq=reverse_char_emb_seq)
            char_hidden_state = torch.cat([char_hidden_state, reverse_char_hidden_state], dim=2)

        feature_seq = self.full_conn(char_hidden_state)
        return feature_seq

    def get_params_config(self):
        param_config = [{'params': self.word_emb.parameters(),
                         'lr': self.emb_learning_rate},
                        {'params': self.char_emb.parameters(),
                         'lr': self.emb_learning_rate},
                        {'params': self.lattice_word_cell.parameters(),
                         'lr': self.lstm_learning_rate},
                        {'params': self.lattice_char_cell.parameters(),
                         'lr': self.lstm_learning_rate},
                        {'params': self.full_conn.parameters(),
                         'lr': self.full_conn_learning_rate}]
        return param_config

if __name__ == '__main__':
    test_file = open('temp/lattice_test.txt')
    bert_model_path = './bert_model/pytorch_pretrained_bert/bert-base-chinese/'
    tok = Tokenizer(path=os.path.join(bert_model_path, 'vocab.txt'))
    wtok = WordTokenizer('word_emb/word_emb_200d_tencent_ailab_top_50w.txt')
    giw = True
    clt = Lattice(tok, wtok, cut_all=True, ignore_invalid_word=True)
    clf = Lattice(tok, wtok, cut_all=True, ignore_invalid_word=False)

    count = 0
    dt1 = dt.datetime.now()
    for text in test_file:
        ids = tok.encode(list(text))
        ids_batch = torch.Tensor([ids]).long()
        mask_batch = torch.Tensor([[1] * len(ids)]).long()
        lllt = clt.to_lattice(ids_batch, mask_batch, print_result=False)
        lllf = clf.to_lattice(ids_batch, mask_batch, print_result=False)
        print('%s  %s  %s' % (count, len(lllt[0]), len(lllf[0])))
        if len(lllt[0]) > len(lllf[0]):
            raise Exception('aaaaaaa')
        count += 1
    dt2 = dt.datetime.now()
    print((dt2 - dt1).total_seconds())
    print((dt2 - dt1).total_seconds() / count)
    print(count)


