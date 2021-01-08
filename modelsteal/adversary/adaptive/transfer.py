import torch.nn as nn
import numpy as np
import torch


class Transfer(object):
    def __init__(self, index_list, queryset, victim_model, substitude_model):
        self.local_choosen_samples = None
        self.victim_choosen_samples = None
        self.victim_model = victim_model
        self.substitude_model = substitude_model
        self.index_list = index_list
        self.queryset = queryset
        self.substitude_result = None
        self.victim_result = None
        self.transferset = []
        self.refresh()

    def refresh(self):
        # generate the queryset for the victim model
        self.victim_choosen_samples = torch.stack([self.queryset[i][0] for i in self.index_list])
        # generate the queryset for the substitude model
        self.local_choosen_samples = torch.stack([self.queryset[i][0] for i in self.index_list])

        # query the victim model
        self.victim_result = self.victim_model(self.victim_choosen_samples)
        # query the substitude model
        self.substitude_result = self.substitude_model(self.local_choosen_samples)

    def get_reward(self):
        """
        count reward by kullback-Leibler divergence(relative entropy)
        :return:
        """
        dim = self.substitude_result.shape[0]
        loss_f = nn.KLDivLoss(reduction='mean')
        return np.mean([loss_f(self.substitude_result[i], self.victim_result[i]).item() for i in range(dim)])

    def get_update_set(self):
        # generate the original set that is used to update the substitude model (non-transformed)
        img_t = [self.queryset.data[i] for i in self.index_list]
        if isinstance(self.queryset.data[0], torch.Tensor):
            img_t = [x.numpy() for x in img_t]
        for i in range(self.victim_choosen_samples.size(0)):
            img_t_i = img_t[i].squeeze() if isinstance(img_t[i], np.ndarray) else img_t[i]
            self.transferset.append((img_t_i, self.victim_result[i].cpu().squeeze()))

        return self.transferset
