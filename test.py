import utils.data as data
import pytorch_pretrained_bert as torch_bert
import datetime as dt
import numpy as np
import torch
import os
import utils.metric
import copy
import random
import torch

t1 = torch.Tensor([[1,2,3], [3,4,5]])
t2 = torch.Tensor([1,2, 9])
t = t1 / t2
print(t.detach().numpy())
exit(0)

t = torch.Tensor([[1,2,3], [3,4,5]])
t2 = torch.Tensor([[5,6], [4,5]])
for tt1, tt2 in zip(t, t2):
    print(tt1)
    print(tt2)
    print(int(tt1.sum()))
    print()
print(len(t))
exit(2)



# -----------------------------------------------------------
format = 'bioes'
model_path = './bert_model/pytorch_pretrained_bert/bert-base-chinese/'
tok = data.Tokenizer(path=os.path.join(model_path, 'vocab.txt'))
ds = data.DataSet(path='./data/tiny_data/%s/train.txt' % format, tokenizer=tok, batch_size=10, shuffle=False)
met = utils.metric.Metric(label_set=ds.label_tokenizer.label_set, tag_set=ds.tag_set, format=format)
# for padded_seq_ids_batch, padded_label_ids_batch, seq_ids_mask_batch, label_ids_mask_batch in ds:
#     seq_batch, label_batch = \
#         ds.decode_batch(padded_seq_ids_batch, padded_label_ids_batch, seq_ids_mask_batch, label_ids_mask_batch)
#     ds.print_batch(seq_batch, label_batch)
#     # ll = []
#     # for label in label_batch:
#     #     ll += label
#     # pll = copy.deepcopy(ll)
#     # for p in range(len(ll)):
#     #     if random.uniform(0, 1) <= 0.2:
#     #         el = random.choice(list(ds.label_tokenizer.label_set))
#     #         pll[p] = el
#     # total_accurate, total_recall, accurate_dict, recall_dict = met.entity_wise_metric(true_label_list=ll, predict_label_list=pll)
#     # total_accurate_, total_recall_, accurate_dict_, recall_dict_ = met.elem_wise_metric(true_label_list=ll, predict_label_list=pll)
#     # ect = met.check_format(ll)
#     # ecp = met.check_format(pll)
#     # print(str(ect) + ' ' + str(ecp))
#     ddd = 0


# -----------------------------------------------------------
# model_path = './bert_model/pytorch_pretrained_bert/bert-base-chinese/'
# # model_path = './bert_model/chinese_bert_wwm/chinese_wwm_ext_pytorch/'
# tok = data.Tokenizer(path=os.path.join(model_path, 'vocab.txt'))
# bert_model = torch_bert.BertModel.from_pretrained(model_path)
# ds = data.DataSet(path='./data/tiny_data/bioes/train.txt', tokenizer=tok, batch_size=32, shuffle=False)
# print(ds.info())
#
# # bert_model = bert_model.cuda() # 显存又他喵的不够啊，这破GPU有跟没有一样！
#
# for padded_seq_ids_batch, padded_label_ids_batch, seq_ids_mask_batch, label_ids_mask_batch in ds:
#     seq_batch, label_batch = \
#         ds.decode_batch(padded_seq_ids_batch, padded_label_ids_batch, seq_ids_mask_batch, label_ids_mask_batch)
#     padded_seq_ids_batch_tensor = torch.Tensor(padded_seq_ids_batch).to(torch.int64) # .cuda()
#     seq_ids_mask_batch_tensor = torch.Tensor(seq_ids_mask_batch).to(torch.int64) # .cuda()
#     padded_seq_ids_batch_tensor.requires_grad = False
#     seq_ids_mask_batch_tensor.requires_grad = False
#     char_emb_seq, sen_emb = bert_model(input_ids=padded_seq_ids_batch_tensor,
#                                        attention_mask=seq_ids_mask_batch_tensor,
#                                        output_all_encoded_layers=False)
#     ds.print_batch(seq_batch, label_batch)
#     print(dt.datetime.now())
#     ddd = 0

#
# rs = ['从', '前', '有', '座', '山', '，', '山', '上', '有', '座', '庙', '。']
# ids = tok.encode(rs)
# seq = tok.decode(ids)

print('begin test %s' % dt.datetime.now())
for i in range(100000):
    ds._test_tokenizer()
print('end test %s' % dt.datetime.now())
#
# lc = 0
# for padded_seq_ids_batch, padded_label_ids_batch, seq_ids_mask_batch, label_ids_mask_batch in ds:
#     seq_batch, label_batch = \
#         ds.decode_batch(padded_seq_ids_batch, padded_label_ids_batch, seq_ids_mask_batch, label_ids_mask_batch)
#     ds.print_batch(seq_batch, label_batch)
#     lc += len(seq_batch)
#     ddd = 0
# print(lc)
# ddd = 0

# -----------------------------------------------------------
# tok = torch_bert.tokenization.BertTokenizer.from_pretrained('./bert_model/pytorch_pretrained_bert/bert-base-chinese')
# mm = torch_bert.BertModel.from_pretrained('./bert_model/pytorch_pretrained_bert/bert-base-chinese')

# tok = torch_bert.tokenization.BertTokenizer.from_pretrained('./bert_model/chinese_bert_wwm/chinese_wwm_ext_pytorch')
# mm = torch_bert.BertModel.from_pretrained('./bert_model/chinese_bert_wwm/chinese_wwm_ext_pytorch')

# ids = tok.convert_tokens_to_ids(rs)
# seq = tok.convert_ids_to_tokens(ids)

# -----------------------------------------------------------
ddd = 0