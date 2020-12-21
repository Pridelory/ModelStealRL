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
        budget = self.budget
        start_B = 0
        end_B = budget
        db_indices = []
        with tqdm(total=budget) as pbar:
            # for t, B in enumerate(range(start_B, end_B, self.batch_size)):
            for i in range(10):
                # idxs = np.random.choice(list(self.idx_set), replace=False,size=8)
                # self.idx_set = self.idx_set - set(idxs)
                idx_set = set(range(80))
                x_t = torch.stack([self.queryset[i][0] for i in idx_set])
                # temp_image = tensor_to_PIL(x_t[1])
                # temp_image.show()
                y_t = self.substitude_model(x_t).cpu()
                db_indice = utils.count_db(y_t)
                db_indices.extend(db_indice)
            return db_indices

