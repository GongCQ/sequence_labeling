import os
import sys
import datetime as dt
import warnings
import torch
from utils.data import DataSetManager
from utils.metric import Metric
from utils.config import BATCH_SIZE, USE_GPU

class Evaluator:
    def __init__(self, model, data_set_manager: DataSetManager, metric: Metric, test_file_path=None):
        self.model = model
        self.data_set_manager = data_set_manager
        self.metric = metric
        self.test_text_list = []
        self.test_seq_ids_batch = []
        self.test_mask_batch = []
        if test_file_path is None:
            test_file_path = os.path.join('data', 'test_file.txt')
        try:
            test_file = open(test_file_path, encoding='utf-8')
            for line in test_file:
                text = line.strip()
                self.test_text_list.append(text)
            self.test_seq_ids_batch, self.test_mask_batch = \
                self.data_set_manager.train_data_set.encode_text_batch(self.test_text_list)
        except Exception as e:
            warnings.warn('fail to load test file.')

    def _eval_test_text(self):
        if not(len(self.test_text_list) == len(self.test_seq_ids_batch) and
            len(self.test_text_list) == len(self.test_mask_batch) and len(self.test_text_list) > 0):
            warnings.warn('invalid test texts.')
            return

        seq_ids = torch.Tensor(self.test_seq_ids_batch).to(torch.int64)
        mask = torch.Tensor(self.test_mask_batch).to(torch.bool)
        if USE_GPU:
            seq_ids = seq_ids.cuda()
            mask = mask.cuda()
        path_score, predict_label_ids_batch = self.model(seq_ids=seq_ids, mask=mask)
        predict_label_ids_batch = predict_label_ids_batch.cpu().numpy()
        mask = mask.cpu().numpy()
        predict_label_batch = self.data_set_manager.train_data_set.decode_label_batch(predict_label_ids_batch, mask)
        print('eval_test_text --------')
        for text, predict_label in zip(self.test_text_list, predict_label_batch):
            print(''.join([(' %s/%s ' % (char, label))
                           if label != 'O'
                           else char
                           for char, label in zip(text, predict_label)]))
        print('-----------------------')


    def eval(self, seq_ids_batch, label_ids_batch, seq_ids_mask_batch, label_ids_mask_batch,
             print_detail=False, label=''):
        label_batch = []
        predict_label_batch = []
        for i in range(0, len(seq_ids_batch), BATCH_SIZE):
            seq_ids_slice = seq_ids_batch[i : min(len(seq_ids_batch), i + BATCH_SIZE)]
            label_ids_slice = label_ids_batch[i : min(len(label_ids_batch), i + BATCH_SIZE)]
            seq_ids_mask_slice = seq_ids_mask_batch[i : min(len(seq_ids_mask_batch), i + BATCH_SIZE)]
            label_ids_mask_slice = label_ids_mask_batch[i : min(len(label_ids_mask_batch), i + BATCH_SIZE)]

            seq_ids_slice_t = torch.Tensor(seq_ids_slice).to(torch.int64)
            seq_ids_mask_slice_t = torch.Tensor(seq_ids_mask_slice).to(torch.bool)
            if USE_GPU:
                seq_ids_slice_t = seq_ids_slice_t.cuda()
                seq_ids_mask_slice_t = seq_ids_mask_slice_t.cuda()

            path_score, predict_label_ids_batch = self.model(seq_ids=seq_ids_slice_t, mask=seq_ids_mask_slice_t)
            predict_label_ids_batch = predict_label_ids_batch.cpu().numpy()
            seq_slice = \
                self.data_set_manager.train_data_set.decode_seq_batch(padded_seq_ids_batch=seq_ids_slice,
                                                                      seq_ids_mask_batch=seq_ids_mask_slice)
            label_slice = \
                self.data_set_manager.train_data_set.decode_label_batch(padded_label_ids_batch=label_ids_slice,
                                                                        label_ids_mask_batch=label_ids_mask_slice)
            predict_label_slice = \
                self.data_set_manager.train_data_set.decode_label_batch(padded_label_ids_batch=predict_label_ids_batch,
                                                                        label_ids_mask_batch=label_ids_mask_slice)

            label_batch += label_slice
            predict_label_batch += predict_label_slice

        total_f_score, total_precision, total_recall, precision_dict, recall_dict = \
            self.metric.label_wise_metric_batch(true_label_list_batch=label_batch,
                                                predict_label_list_batch=predict_label_batch)
        total_f_score_2, total_precision_2, total_recall_2, precision_dict_2, recall_dict_2 = \
            self.metric.entity_wise_metric_batch(true_label_list_batch=label_batch,
                                                 predict_label_list_batch=predict_label_batch)

        print('[%-8s] label-wise : precision %.6f, recall %.6f, f_score %s' %
              (label, total_precision, total_recall, total_f_score))
        print('[%-8s] entity-wise: precision %.6f, recall %.6f, f_score %s' %
              (label, total_precision_2, total_recall_2, total_f_score_2))

        if print_detail:
            print('label-wise detail : ')
            label_list = sorted(list(precision_dict.keys() & recall_dict.keys()))
            for label in label_list:
                print('%6s  precision %.6f  recall %.6f' % (label, precision_dict[label], recall_dict[label]))
            print('entity-wise detail: ')
            entity_list = sorted(list(precision_dict_2.keys() & recall_dict_2.keys()))
            for entity in entity_list:
                print('%6s  precision %.6f  recall %.6f' % (entity, precision_dict_2[entity], recall_dict_2[entity]))


    def random_eval(self, size, print_detail=False, print_eval_test=False):
        if size == 0 or size is None:
            size = sys.maxsize
        line = 'random_eval --------' if size < sys.maxsize else 'FULL_EVAL ****************'
        print('%s %s size %s' % (line, dt.datetime.now(), size))
        for name, data_set in zip(['train', 'test', 'valid'],
                                  [self.data_set_manager.train_data_set,
                                   self.data_set_manager.test_data_set,
                                   self.data_set_manager.valid_data_set]):
            if size == sys.maxsize and name == 'train':
                continue
            seq_ids_batch, label_ids_batch, seq_ids_mask_batch, label_ids_mask_batch = data_set.get_random_batch(size)
            label = ('f_' if size == sys.maxsize else 'r_') + name
            print()
            print('.... %6s set ....' % name)
            self.eval(seq_ids_batch, label_ids_batch, seq_ids_mask_batch, label_ids_mask_batch, print_detail, label)
        print('%s %s size %s' % (line, dt.datetime.now(), size))

        if print_eval_test:
            try:
                self._eval_test_text()
            except Exception as e:
                print('fail to eval_test_text, %s' % e)

    def full_eval(self, print_detail=True, print_eval_test=True):
        self.random_eval(sys.maxsize, print_detail, print_eval_test)
