import os
import warnings
import datetime as dt
import pandas as pd
import jieba
import torch
import torch.nn as nn
from utils.data import Tokenizer, WordTokenizer

pd.set_option('display.max_rows', 1000)
pd.set_option('display.max_columns', 1000)
# pd.set_option('display.max_colwidth',50)
pd.set_option('expand_frame_repr', False)
pd.set_option('display.unicode.ambiguous_as_wide', True)
pd.set_option('display.unicode.east_asian_width', True)

# [s for s in jieba.cut('工信处女干事每月经过下属科室都要亲口交代24口交换机等技术性器件的安装工作',cut_all=True)]
# ['工信处', '处女', '女干事', '干事', '每月', '月经', '经过', '下属', '科室', '都', '要', '亲口', '口交', '交代', '24', '口交', '交换', '交换机', '换机', '等', '技术', '技术性', '性器', '器件', '的', '安装', '安装工', '装工', '工作']


class CharLattice:
    def __init__(self, tokenizer: Tokenizer, word_tokenizer: WordTokenizer, cut_all=True):
        self.tokenizer = tokenizer
        self.word_tokenizer = word_tokenizer
        self.cut_all = cut_all
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
            char_list = self.tokenizer.decode(seq_id.detach().numpy())
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
                        warnings.warn('check format error.\n%s\n%s' % (text, fraction_list))
                        cut_all_fail = True
                        break

            if not self.cut_all or cut_all_fail:
                seg = [s for s in jieba.cut(text, cut_all=False)]
                fraction_list = [(seg[0], 0)]
                accum_len = len(fraction_list[-1][0])
                for current_word in seg[1 : ]:
                    fraction_list.append((current_word, accum_len))
                    accum_len += len(current_word)
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
                           for fraction in fraction_list]
            lattice_seq_batch.append(lattice_seq)

        return lattice_seq_batch


if __name__ == '__main__':
    test_file = open('temp/lattice_test.txt')
    bert_model_path = './bert_model/pytorch_pretrained_bert/bert-base-chinese/'
    tok = Tokenizer(path=os.path.join(bert_model_path, 'vocab.txt'))
    wtok = WordTokenizer('word_emb/word_emb_200d_tencent_ailab_top_10w.txt')
    cl = CharLattice(tok, wtok, cut_all=True)

    count = 0
    dt1 = dt.datetime.now()
    for text in test_file:
        ids = tok.encode(list(text))
        ids_batch = torch.Tensor([ids]).long()
        mask_batch = torch.Tensor([[1] * len(ids)]).long()
        cl.to_lattice(ids_batch, mask_batch, print_result=False)
        count += 1
    dt2 = dt.datetime.now()
    print((dt2 - dt1).total_seconds())
    print((dt2 - dt1).total_seconds() / count)
    print(count)


