import os
import copy
import numpy as np
from utils.data import OUT_LABEL, LABEL_SEP

class Metric:
    def __init__(self, label_set: set, tag_set: set, format: str, decimal: int = 6):
        '''
        all labels must be str, not int id.
        :param label_set: a set such as {'B-ORG', 'I-ORG', 'B-PER', 'I-PER', 'O'}
        :param tag_set: a set such as {'ORG', 'LOC', 'PER'}
        :param format: bio / bioes
        :param decimal: the number of decimal place
        '''
        assert format == 'bio' or format == 'bioes', 'unknown format %s, format must either be bio or bioes.' % format
        self.label_set = copy.deepcopy(label_set)
        self.tag_set = copy.deepcopy(tag_set)
        self.format = format
        self.decimal = decimal

    def _check_format_bioes(self, label_list):
        error_count = 0
        for i, label in enumerate(label_list):
            next_label = label_list[i + 1] if i < len(label_list) - 1 else OUT_LABEL
            loc, tag = tuple(label.split(LABEL_SEP)) if label.find(LABEL_SEP) > 0 else (OUT_LABEL, OUT_LABEL)
            next_loc, next_tag = tuple(next_label.split(LABEL_SEP)) if next_label.find(LABEL_SEP) > 0 else (OUT_LABEL, OUT_LABEL)

            if loc == 'B':
                if not ((next_loc == 'I' or next_loc == 'E') and tag == next_tag):
                    error_count += 1
            elif loc == 'I':
                if not ((next_loc == 'I' or next_loc == 'E') and tag == next_tag):
                    error_count += 1
            elif loc == 'O':
                if next_loc == 'I' or next_loc == 'E':
                    error_count += 1
            elif loc == 'E':
                if next_loc == 'I' or next_loc == 'E':
                    error_count += 1
            elif loc == 'S':
                if next_loc == 'I' or next_loc == 'E':
                    error_count += 1

        return error_count

    def _check_format_bio(self, label_list):
        error_count = 0
        for i, label in enumerate(label_list):
            next_label = label_list[i + 1] if i < len(label_list) - 1 else OUT_LABEL
            loc, tag = tuple(label.split(LABEL_SEP)) if label.find(LABEL_SEP) > 0 else (OUT_LABEL, OUT_LABEL)
            next_loc, next_tag = tuple(next_label.split(LABEL_SEP)) if next_label.find(LABEL_SEP) > 0 else (OUT_LABEL, OUT_LABEL)

            if loc == 'B':
                if not ((next_loc == 'I' and tag == next_tag) or next_loc == 'O' or next_loc == 'B'):
                    error_count += 1
            elif loc == 'I':
                if not ((next_loc == 'I' and tag == next_tag) or next_loc == 'O' or next_loc == 'B'):
                    error_count += 1
            elif loc == 'O':
                if next_loc == 'I':
                    error_count += 1
        return error_count

    def check_format(self, label_list):
        return self._check_format_bioes(label_list) if self.format == 'bioes' else self._check_format_bio(label_list)

    def label_wise_metric(self, true_label_list, predict_label_list):
        assert len(true_label_list) == len(predict_label_list), 'the lengths of truth and prediction have to be equal.'

        true_pos_predict_pos_count = 0
        true_pos_predict_neg_count = 0
        true_neg_predict_pos_count = 0
        true_pos_predict_pos_count_dict = dict([(label, 0) for label in self.label_set])
        true_pos_predict_neg_count_dict = dict([(label, 0) for label in self.label_set])
        true_neg_predict_pos_count_dict = dict([(label, 0) for label in self.label_set])
        for true_label, predict_label in zip(true_label_list, predict_label_list):
            is_correct = true_label == predict_label
            if true_label.find(LABEL_SEP) > 0:
                if is_correct:
                    true_pos_predict_pos_count += 1
                    true_pos_predict_pos_count_dict[true_label] += 1
                else:
                    true_pos_predict_neg_count += 1
                    true_pos_predict_neg_count_dict[true_label] += 1

            if predict_label.find(LABEL_SEP) > 0:
                if not is_correct:
                    true_neg_predict_pos_count += 1
                    true_neg_predict_pos_count_dict[predict_label] += 1

        predict_pos_count = true_pos_predict_pos_count + true_neg_predict_pos_count
        true_pos_count = true_pos_predict_pos_count + true_pos_predict_neg_count
        total_precision = true_pos_predict_pos_count / predict_pos_count if predict_pos_count > 0 else np.nan
        total_recall = true_pos_predict_pos_count / true_pos_count if true_pos_count > 0 else np.nan
        total_f_score = 2 * total_precision * total_recall / (total_precision + total_recall) \
                        if total_precision + total_recall > 0 else np.nan

        precision_dict = {}
        recall_dict = {}
        for label in self.label_set:
            predict_pos_count = true_pos_predict_pos_count_dict[label] + true_neg_predict_pos_count_dict[label]
            true_pos_count = true_pos_predict_pos_count_dict[label] + true_pos_predict_neg_count_dict[label]
            precision = true_pos_predict_pos_count_dict[label] / predict_pos_count if predict_pos_count > 0 else np.nan
            recall = true_pos_predict_pos_count_dict[label] / true_pos_count if true_pos_count > 0 else np.nan
            precision_dict[label] = round(precision, self.decimal)
            recall_dict[label] = round(recall, self.decimal)

        return round(total_f_score, self.decimal), round(total_precision, self.decimal), \
               round(total_recall, self.decimal), precision_dict, recall_dict

    def entity_wise_metric(self, true_label_list, predict_label_list):
        assert len(true_label_list) == len(predict_label_list), 'the lengths of truth and prediction have to be equal.'

        true_entity_set = set() # 'tag_begin_end'
        predict_entity_set = set()
        for entity_set, label_list in [(true_entity_set, true_label_list), (predict_entity_set, predict_label_list)]:
            current_entity_begin = None
            current_entity_tag = None
            for i, label in enumerate(label_list):
                in_entity = label.find(LABEL_SEP) > 0
                at_begin = in_entity and (label[0] == 'B' or (self.format == 'bioes' and label[0] == 'S'))
                tag_changed = current_entity_tag is not None and label.find(LABEL_SEP + current_entity_tag) < 0
                end_last_entity = current_entity_tag is not None and (at_begin or tag_changed)
                # a new entity beginning or stepping out last entity, it means the ending of the last entity.
                if end_last_entity:
                    stamp = '%s %s %s' % (current_entity_tag, current_entity_begin, i)
                    entity_set.add(stamp)
                    current_entity_begin = None
                    current_entity_tag = None
                # a new entity beginning.
                if at_begin:
                    label_split = label.split(LABEL_SEP)
                    loc = label_split[0]
                    tag = label_split[1]
                    current_entity_begin = i
                    current_entity_tag = tag

        true_pos_predict_pos_count = 0
        true_pos_predict_neg_count = 0
        true_neg_predict_pos_count = 0
        true_pos_predict_pos_count_dict = dict([(tag, 0) for tag in self.tag_set])
        true_pos_predict_neg_count_dict = dict([(tag, 0) for tag in self.tag_set])
        true_neg_predict_pos_count_dict = dict([(tag, 0) for tag in self.tag_set])
        for stamp in true_entity_set:
            tag = stamp.split(' ')[0]
            if stamp in predict_entity_set:
                true_pos_predict_pos_count += 1
                true_pos_predict_pos_count_dict[tag] += 1
            else:
                true_pos_predict_neg_count += 1
                true_pos_predict_neg_count_dict[tag] += 1
        for stamp in predict_entity_set:
            tag = stamp.split(' ')[0]
            if stamp not in true_entity_set:
                true_neg_predict_pos_count += 1
                true_neg_predict_pos_count_dict[tag] += 1

        predict_pos_count = true_pos_predict_pos_count + true_neg_predict_pos_count
        true_pos_count = true_pos_predict_pos_count + true_pos_predict_neg_count
        total_precision = true_pos_predict_pos_count / predict_pos_count if predict_pos_count > 0 else np.nan
        total_recall = true_pos_predict_pos_count / true_pos_count if true_pos_count > 0 else np.nan
        total_f_score = 2 * total_precision * total_recall / (total_precision + total_recall) \
                        if total_precision + total_recall > 0 else np.nan

        precision_dict = {}
        recall_dict = {}
        for tag in self.tag_set:
            predict_pos_count = true_pos_predict_pos_count_dict[tag] + true_neg_predict_pos_count_dict[tag]
            true_pos_count = true_pos_predict_pos_count_dict[tag] + true_pos_predict_neg_count_dict[tag]
            precision = true_pos_predict_pos_count_dict[tag] / predict_pos_count if predict_pos_count > 0 else np.nan
            recall = true_pos_predict_pos_count_dict[tag] / true_pos_count if true_pos_count > 0 else np.nan
            precision_dict[tag] = round(precision, self.decimal)
            recall_dict[tag] = round(recall, self.decimal)

        return round(total_f_score, self.decimal), round(total_precision, self.decimal), \
               round(total_recall, self.decimal), precision_dict, recall_dict

    def _flatten_batch(self, true_label_list_batch, predict_label_list_batch):
        all_true_label_list = []
        all_predict_label_list = []
        for true_label_list, predict_label_list in zip(true_label_list_batch, predict_label_list_batch):
            all_true_label_list += true_label_list
            all_predict_label_list += predict_label_list
        return all_true_label_list, all_predict_label_list

    def label_wise_metric_batch(self, true_label_list_batch, predict_label_list_batch):
        all_true_label_list, all_predict_label_list = self._flatten_batch(true_label_list_batch, predict_label_list_batch)
        return self.label_wise_metric(true_label_list=all_true_label_list, predict_label_list=all_predict_label_list)

    def entity_wise_metric_batch(self, true_label_list_batch, predict_label_list_batch):
        all_true_label_list, all_predict_label_list = self._flatten_batch(true_label_list_batch, predict_label_list_batch)
        return self.entity_wise_metric(true_label_list=all_true_label_list, predict_label_list=all_predict_label_list)
