import torch
from utils.data import DataSetManager
from utils.metric import Metric

class Evaluator:
    def __init__(self, model, data_set_manager: DataSetManager, metric: Metric):
        self.model = model
        self.data_set_manager = data_set_manager
        self.metric = metric

    def eval(self, seq_ids_batch, label_ids_batch, seq_ids_mask_batch, label_ids_mask_batch, print_detail=True):
        seq_ids_batch_t = torch.Tensor(seq_ids_batch).to(torch.int64)
        seq_ids_mask_batch_t = torch.Tensor(seq_ids_mask_batch).to(torch.bool)

        path_score, predict_label_ids_batch = self.model(seq_ids=seq_ids_batch_t, mask=seq_ids_mask_batch_t)
        # seq_batch = \
        #     self.data_set_manager.train_data_set.decode_seq_batch(padded_seq_ids_batch=seq_ids_batch,
        #                                                           seq_ids_mask_batch=seq_ids_mask_batch)
        label_batch = \
            self.data_set_manager.train_data_set.decode_label_batch(padded_label_ids_batch=label_ids_batch,
                                                                    label_ids_mask_batch=label_ids_mask_batch)
        predict_label_batch = \
            self.data_set_manager.train_data_set.decode_label_batch(padded_label_ids_batch=predict_label_ids_batch,
                                                                    label_ids_mask_batch=label_ids_mask_batch)

        total_accurate, total_recall, accurate_dict, recall_dict = \
            self.metric.label_wise_metric_batch(true_label_list_batch=label_batch,
                                                predict_label_list_batch=predict_label_batch)
        total_accurate_2, total_recall_2, accurate_dict_2, recall_dict_2 = \
            self.metric.entity_wise_metric_batch(true_label_list_batch=label_batch,
                                                 predict_label_list_batch=predict_label_batch)

        print('\n~~ label  wise: prc %.6f, rec %.6f' % (total_accurate, total_recall))
        print('~~ entity wise: prc %.6f, rec %.6f' % (total_accurate_2, total_recall_2))

        if print_detail:
            print('\n~~ label  wise: \nprc_detail %s \nrec_detail %s' % (accurate_dict, recall_dict))
            print('~~ entity wise: \nprc_detail %s \nrec_detail %s' % (accurate_dict_2, recall_dict_2))

    def random_eval(self, size):
        pass

    def full_eval(self):
        pass
