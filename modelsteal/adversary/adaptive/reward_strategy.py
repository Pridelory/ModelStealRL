import torch.nn as nn
from numpy import *
import torch

class RewardStrategy(object):
    def __init__(self, local_choosen_samples, victim_choosen_samples,victim_model, substitude_model):
        self.local_choosen_samples = local_choosen_samples
        self.victim_choosen_samples = victim_choosen_samples
        self.victim_model = victim_model
        self.substitude_model = substitude_model
        self.substitude_result = None
        self.victim_result = None

    def count_by_kld(self):
        """
        count reward by kullback-Leibler divergence(relative entropy)
        :return:
        """
        self.substitude_result = self.substitude_model(self.local_choosen_samples)
        self.victim_result = self.victim_model(self.victim_choosen_samples)
        dim = self.substitude_result.shape[0]
        loss_f = nn.KLDivLoss(reduction='mean')
        return mean([loss_f(self.substitude_result[i], self.victim_result[i]).item() for i in range(dim)])

    def get_victim_trained_info(self):
        # sample_num = self.victim_result.shape[0]
        # victim_result = torch.stack([self.victim_result[i][0] for i in range(sample_num)])
        # return victim_result
        return self.victim_result