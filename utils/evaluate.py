import sys
import datetime as dt
import torch
from utils.data import DataSetManager
from utils.metric import Metric
from utils.config import BATCH_SIZE, USE_GPU

class Evaluator:
    def __init__(self, model, data_set_manager: DataSetManager, metric: Metric):
        self.model = model
        self.data_set_manager = data_set_manager
        self.metric = metric

    def eval(self, seq_ids_batch, label_ids_batch, seq_ids_mask_batch, label_ids_mask_batch, print_detail=False):
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

        print('label-wise : precision %.6f, recall %.6f, f_score %s' % (total_precision, total_recall, total_f_score))
        print('entity-wise: precision %.6f, recall %.6f, f_score %s' % (total_precision_2, total_recall_2, total_f_score_2))

        if print_detail:
            print('label-wise detail : ')
            label_list = sorted(list(precision_dict.keys() & recall_dict.keys()))
            for label in label_list:
                print('%6s  precision %.6f  recall %.6f' % (label, precision_dict[label], recall_dict[label]))
            print('entity-wise detail: ')
            entity_list = sorted(list(precision_dict_2.keys() & recall_dict_2.keys()))
            for entity in entity_list:
                print('%6s  precision %.6f  recall %.6f' % (entity, precision_dict_2[entity], recall_dict_2[entity]))

    def random_eval(self, size, print_detail=False):
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
            print()
            print('.... %6s set ....' % name)
            self.eval(seq_ids_batch, label_ids_batch, seq_ids_mask_batch, label_ids_mask_batch, print_detail)
        print('%s %s size %s' % (line, dt.datetime.now(), size))

    def full_eval(self, print_detail=True):
        self.random_eval(sys.maxsize, print_detail)
