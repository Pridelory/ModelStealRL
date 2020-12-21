import torchvision
import torch.nn as nn
from modelsteal.adversary.adaptive.strategy import DecisionBoundStrategy

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
        return x


class Step:
    def __init__(self, action, state, queryset, batch_size, budget):
        super(Step, self).__init__()
        self.action = action
        self.state = state
        self.queryset = queryset
        self.batch_size = batch_size
        self.budget = budget

    def step(self):
        """
        1.数据集根据state网络分成不同决策边界的cluster，
        2.action去选择某个特定的cluster i的前k个样本
        3.用选出的样本去查victim model，把结果与当前结果算出reward；
        4.用此cluster更新state为new_state
        """
        # 根据当前state网络逻辑划分数据集
        strategy = DecisionBoundStrategy(self.state, self.queryset, self.batch_size, self.budget)
        db_indice_list = strategy.get_db_result()

        # action去选择某个特定的cluster i的前k个样本


        return 1,2,3,4