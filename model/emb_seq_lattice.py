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
                    # -------------------------
                    # for i in range(1, min(current_len, last_len) + 1):
                    #     current_try_begin = current_max_begin - i
                    #     if text[current_try_begin : current_try_begin + current_len] == current_word and \
                    #         last_word[last_len - i : ] == current_word[ : i]:
                    #         current_matched_begin = current_try_begin
                    # .........................
                    for i in range(1, max(current_len, last_len) + 1):
                        current_try_begin = current_max_begin - i
                        if text[current_try_begin : current_try_begin + current_len] == current_word and \
                            last_word[last_len - i : min(last_len, last_len - i + current_len)] == current_word[ : min(current_len, i)]:
                            current_matched_begin = current_try_begin

                    # -------------------------
                    fraction_list.append((current_word, current_matched_begin))
                if len(fraction_list[-1][0]) + fraction_list[-1][1] != len(text):
                    warnings.warn('generate an invalid fraction_list when cut_all is True.\n%s\n%s' %
                                  (text, fraction_list))
                    cut_all_fail = True

            if not self.cut_all or cut_all_fail:
                seg = [s for s in jieba.cut(text, cut_all=False)]
                fraction_list = [(seg[0], 0)]
                accum_len = len(fraction_list[-1][0])
                for current_word in seg[1 : ]:
                    fraction_list.append((current_word, accum_len))
                    accum_len += len(current_word)
            fraction_list_batch.append(fraction_list)

            if print_result:
                align_df = pd.DataFrame(data=[[''] * len(text)] * len(fraction_list), columns=list(text))
                for i in range(len(fraction_list)):
                    fraction = fraction_list[i]
                    for j in range(len(fraction[0])):
                        align_df.iloc[i, fraction[1] + j] = fraction[0][j]
                # print('~align~: ' + str(fraction_list))
                # print(align_df)
                align_df.to_csv('temp/' + text[ : min(50, len(text))] + '.csv')

        return fraction_list_batch


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


