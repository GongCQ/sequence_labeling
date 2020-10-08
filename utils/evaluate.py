from utils.data import DataSetManager
from utils.metric import Metric

class Evaluator:
    def __init__(self, model, data_set_manager: DataSetManager, metric: Metric):
        self.model = model
        self.data_set_manager = data_set_manager
        self.metric = metric

    def random_eval(self, size):
        pass

    def full_eval(self):
        pass
