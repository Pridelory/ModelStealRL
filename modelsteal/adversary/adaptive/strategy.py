import numpy as np
import torch
from tqdm import tqdm
from modelsteal.utils import utils


class DecisionBoundStrategy(object):
    def __init__(self, substitude_model, queryset, batch_size, budget):
        self.substitude_model = substitude_model
        self.queryset = queryset
        self.batch_size = batch_size
        self.budget = budget
        self.list = list()
        self.idx_set = set(range(len(self.queryset)))
        self._restart()

    def _restart(self):
        self.idx_set = (range(len(self.queryset)))

    def get_db_result(self):
        start_B = 0
        # end_B = len(self.queryset.data)
        end_B = 128
        db_indices = []
        with tqdm(total=end_B) as pbar:
            for t, B in enumerate(range(start_B, end_B, self.batch_size)):
                idx_set = range(B, B + self.batch_size, 1)
                x_t = torch.stack([self.queryset[i][0] for i in idx_set])
                y_t = self.substitude_model(x_t).cpu()
                # get the decision bound
                db_indice = utils.count_db(y_t)
                db_indices.extend(db_indice)
                pbar.update(x_t.size(0))
            return db_indices
