import torchvision
import torch.nn as nn
from modelsteal.adversary.adaptive.strategy import KmaxStrategy, DecisionBoundStrategy, Strategy
from modelsteal.adversary.adaptive.transfer import Transfer
import modelsteal.adversary.adaptive.prepare as prepare
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

class LeNet(nn.Module):
    """A simple MNIST network

    Source: https://github.com/pytorch/examples/blob/master/mnist/main.py
    """
    def __init__(self, num_classes=10, **kwargs):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4*4*50, 500)
        self.fc2 = nn.Linear(500, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4*4*50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class Step:
    def __init__(self, action, state, victim_model, victim_queryset, victim_testset, queryset, batch_size, budget, device):
        super(Step, self).__init__()
        self.action = action
        self.state = state
        self.victim_model = victim_model
        self.victim_queryset = victim_queryset
        self.victim_testset = victim_testset
        self.queryset = queryset
        self.batch_size = batch_size
        self.budget = budget
        self.device = device

    def step(self, j):
        """
        1.根据action选策略
        2.用选出的样本去查victim model，把结果与当前结果算出reward；
        3.用此cluster更新state为new_state
        """
        strategy = Strategy()
        if self.action == 0:
            strategy = DecisionBoundStrategy(self.state, self.queryset, self.batch_size, self.budget)
        elif self.action == 1 :
            strategy = KmaxStrategy(self.state, self.queryset, self.batch_size, self.budget)

        # 根据action选定的策略获取数据集下标
        db_indice_list = strategy.get_result(100)

        # 用choosen_samples分别查victim model 和 substitude model算reward
        transfer = Transfer(db_indice_list, self.queryset, victim_model=self.victim_model,
                                         substitude_model=self.state)
        reward = transfer.get_reward()

        # 用此cluster更新state为new_state
        updateset = transfer.get_update_set()
        # update the substitude model
        new_state = prepare.train(self.state, updateset, self.victim_testset, device=self.device)
        return new_state, reward






