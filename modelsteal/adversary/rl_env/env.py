import torchvision
import torch.nn as nn
from modelsteal.adversary.adaptive.strategy import DecisionBoundStrategy
from modelsteal.utils import utils
from modelsteal.adversary.adaptive.reward_strategy import RewardStrategy
import modelsteal.utils.customized_model as model_utils
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
import torch.nn.functional as F

class TrainSet(object):
    def __init__(self, train_data, train_label):
        self.train_data = train_data
        self.train_label = train_label

    def __len__(self):
        return 2


class Reset(nn.Module):
    def __init__(self, pretrained, output_classes):
        super(Reset, self).__init__()
        # self.conv = nn.Conv2d(1, 3, kernel_size=1)
        self.reset = torchvision.models.resnet18(pretrained)
        self.fc = nn.Linear(1000, output_classes)

    def forward(self, x):
        # x = self.conv(x)
        x = self.reset(x)
        x = self.fc(x)
        x = F.softmax(x)
        return x


class Step:
    def __init__(self, action, state, victim_model, victim_queryset, queryset, batch_size, budget, device):
        super(Step, self).__init__()
        self.action = action
        self.state = state
        self.victim_model = victim_model
        self.victim_queryset = victim_queryset
        self.queryset = queryset
        self.batch_size = batch_size
        self.budget = budget
        self.device = device

    def step(self, j):
        """
        1.数据集根据state网络分成不同决策边界的cluster，
        2.action去选择某个特定的cluster i的前k个样本
        3.用选出的样本去查victim model，把结果与当前结果算出reward；
        4.用此cluster更新state为new_state
        """
        # 根据当前state网络逻辑划分数据集
        strategy = DecisionBoundStrategy(self.state, self.queryset, self.batch_size, self.budget)
        db_indice_list = strategy.get_db_result(j)

        print(len(set(db_indice_list)))
        # action去选择某个特定的cluster i的前k个样本
        # 获取victim的query set下标
        index_list = [i for i in range(len(db_indice_list)) if db_indice_list[i] == self.action]

        # if j == 0:
        #     index_list = [38]

        # 根据index_list选样本
        local_choosen_samples = utils.get_trainingdata_by_index(self.queryset, index_list)
        victim_choosen_samples = utils.get_trainingdata_by_index(self.victim_queryset, index_list)

        # temp_image = tensor_to_pil(victim_choosen_samples[1])
        # temp_image.show()

        # 用choosen_samples分别查victim model 和 substitude model算reward
        reward_strategy = RewardStrategy(local_choosen_samples, victim_choosen_samples, victim_model=self.victim_model,
                                         substitude_model=self.state)
        reward = reward_strategy.count_by_kld()

        # 用此cluster更新state为new_state
        victim_trained_info = reward_strategy.get_victim_trained_info()
        trainset = TensorDataset(local_choosen_samples, victim_trained_info)
        new_state = model_utils.train_model(self.state, trainset, device=self.device)
        return self.state, reward
