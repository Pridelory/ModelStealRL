from modelsteal.adversary.rl_env import env
import argparse
import tensorlayer as tl
import tensorflow as tf
from modelsteal.utils import utils
import numpy as np
import random
import torch
import modelsteal.config as cfg
import os
from modelsteal import datasets
from scipy.special import comb


def get_model(inputs_shape):
    # 第一部分
    input = tl.layers.Input(inputs_shape)
    # h1 = tl.layers.Dense(16, tf.nn.relu, W_init=tf.initializers.GlorotUniform())(input)
    # h2 = tl.layers.Dense(16, tf.nn.relu, W_init=tf.initializers.GlorotUniform())(h1)
    # # 第二部分
    # svalue = tl.layers.Dense(2, )(h2)
    # # 第三部分
    # avalue = tl.layers.Dense(2, )(h2)  # 计算avalue
    # mean = tl.layers.Lambda(lambda x: tf.reduce_mean(x, axis=1, keepdims=True))(avalue)  # 用Lambda层，计算avg(a)
    # advantage = tl.layers.ElementwiseLambda(lambda x, y: x - y)([avalue, mean])  # a - avg(a)

    # output = tl.layers.ElementwiseLambda(lambda x, y: x + y)([svalue, avalue])
    output = tl.layers.Dense(4, act=None, W_init=tf.random_uniform_initializer(0, 0.01), b_init=None, name='q_a_s')(
        input)
    return tl.models.Model(inputs=input, outputs=output)


def main():
    parser = argparse.ArgumentParser(description='Train a model')
    # Required arguments
    # parser.add_argument('model_dir', metavar='DIR', type=str, help='Directory containing transferset.pickle')
    # parser.add_argument('model_arch', metavar='MODEL_ARCH', type=str, help='Model name')
    # parser.add_argument('testdataset', metavar='DS_NAME', type=str, help='Name of test')
    # parser.add_argument('--budgets', metavar='B', type=str,
    #                     help='Comma separated values of budgets. Knockoffs will be trained for each budget.')
    # # Optional arguments
    # parser.add_argument('-d', '--device_id', metavar='D', type=int, help='Device id. -1 for CPU.', default=0)
    # parser.add_argument('-b', '--batch-size', type=int, default=64, metavar='N',
    #                     help='input batch size for training (default: 64)')
    # parser.add_argument('-e', '--epochs', type=int, default=100, metavar='N',
    #                     help='number of epochs to train (default: 100)')
    parser.add_argument('--queryset', metavar='TYPE', type=str, help='Adversary\'s dataset (P_A(X))', required=True)
    parser.add_argument('--modelfamily', metavar='TYPE', type=str, help='Model family', default=None)
    parser.add_argument('--batch_size', metavar='TYPE', type=int, help='Batch size of queries', default=8)
    parser.add_argument('-d', '--device_id', metavar='D', type=int, help='Device id. -1 for CPU.', default=-1)
    parser.add_argument('--budget', metavar='N', type=int, help='Size of transfer set to construct',
                        required=True)
    # parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
    #                     help='learning rate (default: 0.01)')
    # parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
    #                     help='SGD momentum (default: 0.5)')
    # parser.add_argument('--log-interval', type=int, default=50, metavar='N',
    #                     help='how many batches to wait before logging training status')
    # parser.add_argument('--resume', default=None, type=str, metavar='PATH',
    #                     help='path to latest checkpoint (default: none)')
    # parser.add_argument('--lr-step', type=int, default=60, metavar='N',
    #                     help='Step sizes for LR')
    # parser.add_argument('--lr-gamma', type=float, default=0.1, metavar='N',
    #                     help='LR Decay Rate')
    # parser.add_argument('-w', '--num_workers', metavar='N', type=int, help='# Worker threads to load data', default=10)
    # parser.add_argument('--pretrained', type=str, help='Use pretrained network', default=None)
    # parser.add_argument('--weighted-loss', action='store_true', help='Use a weighted loss', default=False)
    # # Attacker's defense
    # parser.add_argument('--argmaxed', action='store_true', help='Only consider argmax labels', default=False)
    # parser.add_argument('--optimizer_choice', type=str, help='Optimizer', default='sgdm',
    #                     choices=('sgd', 'sgdm', 'adam', 'adagrad'))
    args = parser.parse_args()
    params = vars(args)

    torch.manual_seed(cfg.DEFAULT_SEED)
    if params['device_id'] >= 0:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(params['device_id'])
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    # ----------- Set up queryset -----------
    queryset_name = params['queryset']
    valid_datasets = datasets.__dict__.keys()
    if queryset_name not in valid_datasets:
        raise ValueError('Dataset not found. Valid arguments = {}'.format(valid_datasets))
    modelfamily = datasets.dataset_to_modelfamily[queryset_name] if params['modelfamily'] is None else params['modelfamily']
    transform = datasets.modelfamily_to_transforms[modelfamily]['toResnet']
    if queryset_name == 'ImageFolder':
        assert params['root'] is not None, 'argument "--root ROOT" required for ImageFolder'
        queryset = datasets.__dict__[queryset_name](root=params['root'], transform=transform)
    else:
        queryset = datasets.__dict__[queryset_name](train=True, transform=transform)

    # classes of dataset
    class_size = len(queryset.classes)
    # comb_num is the output dimension of q_network
    comb_num = int(comb(class_size, 2))

    # 按episode循环
    for i in range(1):

        # 重置环境初始状态
        # 初始化一个substitude model
        state = env.Reset(pretrained=True, output_classes=class_size)
        state.to(device)
        # for batch_idx, (data, target) in enumerate(queryset):

        # episode
        for j in range(99):
            # 从state网络中提取权重listm(s)
            weight_list = utils.extract_parameter(state)

            # 初始化Q network
            q_network = get_model([None, len(weight_list)])
            q_network.train()

            # 把当前状态（权重矩阵）输入到Q网络 得到Action数组
            allQ = q_network(np.asarray([weight_list], dtype=np.float32)).numpy()

            # 找一个Q值最大的action
            # 加噪声
            a = np.max(allQ, 1)

            step = env.Step(action=a, state=state, queryset=queryset, batch_size=params['batch_size'], budget=params['budget'])
            # 输入到环境，获得下一步的state，reward，done
            s1, r, d, _ = step.step()

            # 把new-state 再放入Q，得到
            # newQ = qnetwork(np.asarray([to_one_hot(s1, 16)], dtype=np.float32)).numpy()

            # 找一个最合适的action
            # a = np.XXX(newQ, 1)
            # targetQ = r + l * a
            print()


if __name__ == '__main__':
    main()
