import torch
from tqdm import tqdm
from modelsteal.utils import utils

class Strategy(object):
    def get_result(self, k):
        pass


class DecisionBoundStrategy(Strategy):
    """
    decision bound strategy
    """
    def __init__(self, substitude_model, queryset, batch_size, budget):
        self.substitude_model = substitude_model
        self.queryset = queryset
        self.batch_size = batch_size
        self.budget = budget
        self.list = list()
        self.idx_set = set(range(len(self.queryset)))

    def get_result(self, k):
        start_B = 0
        # end_B = len(self.queryset.data)
        # end_B = len(self.queryset)
        end_B = 1024
        db_indices = []
        whole_indices = [[] for i in range(45)]
        with tqdm(total=end_B) as pbar:
            for t, B in enumerate(range(start_B, end_B, self.batch_size)):
                idx_set = range(B, B + self.batch_size, 1)
                x_t = torch.stack([self.queryset[i][0] for i in idx_set])
                y_t = self.substitude_model(x_t).cpu()
                # get the decision bound
                db_indice = utils.count_db(y_t)
                for i, j in zip(db_indice, idx_set):
                    whole_indices[i].append(j)
                pbar.update(x_t.size(0))

            len_of_array = [len(temp) for temp in whole_indices]
            i = len_of_array.index(max(len_of_array))
            return whole_indices[i]

class KmaxStrategy(Strategy):
    def __init__(self, substitude_model, queryset, batch_size, budget):
        self.substitude_model = substitude_model
        self.queryset = queryset
        self.batch_size = batch_size
        self.budget = budget

    def get_result(self, k):
        start_B = 0
        end_B = 1024
        whole_indices = []
        with tqdm(total=end_B) as pbar:
            for t, B in enumerate(range(start_B, end_B, self.batch_size)):
                idx_set = range(B, B + self.batch_size, 1)
                x_t = torch.stack([self.queryset[i][0] for i in idx_set])
                y_t = self.substitude_model(x_t).cpu()
                # get the dict
                for i, j in zip(idx_set, range(self.batch_size)):
                    temp_tuplelist = [(i, max(y_t[j]).item())]
                    whole_indices.extend(temp_tuplelist)
                pbar.update(x_t.size(0))
            # sort the whole_indices by the second column
            sorted_indices = sorted(whole_indices, key= lambda t:t[1], reverse=True)
            # get top k indices
            return [sorted_indices[i][0] for i in range(100)]
















