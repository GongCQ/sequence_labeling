import utils.data as data
import pytorch_pretrained_bert as torch_bert
import datetime as dt

tok = data.Tokenizer(path='./bert_model/pytorch_pretrained_bert/bert-base-chinese/vocab.txt')
ds = data.DataSet(path='./data/tiny_data/bioes/train.txt', tokenizer=tok, batch_size=32)

rs = ['从', '前', '有', '座', '山', '，', '山', '上', '有', '座', '庙', '。']
ids = tok.encode(rs)
seq = tok.decode(ids)

print('begin test %s' % dt.datetime.now())
for i in range(100000):
    ds._test_tokenizer()
print('end test %s' % dt.datetime.now())

# tok = torch_bert.tokenization.BertTokenizer.from_pretrained('./bert_model/pytorch_pretrained_bert/bert-base-chinese')
# mm = torch_bert.BertModel.from_pretrained('./bert_model/pytorch_pretrained_bert/bert-base-chinese')

# tok = torch_bert.tokenization.BertTokenizer.from_pretrained('./bert_model/chinese_bert_wwm/chinese_wwm_ext_pytorch')
# mm = torch_bert.BertModel.from_pretrained('./bert_model/chinese_bert_wwm/chinese_wwm_ext_pytorch')

# ids = tok.convert_tokens_to_ids(rs)
# seq = tok.convert_ids_to_tokens(ids)
ddd = 0