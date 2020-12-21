import os
import os.path as osp
import torch
import numpy as np
from scipy.special import comb

def create_dir(dir_path):
    """
    create dirs if not exist
    :param dir_path:
    :return:
    """
    if not osp.exists(dir_path):
        print('Path {} does not exist. Creating it...'.format(dir_path))
        os.makedirs(dir_path)

## 把分类的数字表示，变成onehot表示。
# 例如有4类，那么第三类变为：[0,0,1,0]的表示。
def to_one_hot(i, n_classes=None):
    """

    :param i:
    :param n_classes:
    :return:
    """
    a = np.zeros(n_classes, 'uint8')  # 这里先按照分类数量构建一个全0向量
    a[i] = 1  # 然后点亮需要onehot的位数。
    return a

def extract_parameter(model):
    """
    count the weight list of a neural network
    :param model: model to be extracted
    :return: list containing all parameters of the neural network
    """
    result_list = []
    for name, parameters in model.named_parameters():
        temp_list = torch.flatten(parameters).detach().numpy().tolist()
        result_list.extend(temp_list)
    return result_list

def count_db_detail(indices, comb_num):
    """
    count the decision bound between two indices
    :param indices:
    :param comb_num:
    :return:
    """
    sorted, _ = torch.sort(indices)
    x = sorted[0].item()
    y = sorted[1].item()
    result = (-0.5) * x * x + 8.5 * x + y - 1
    return int(result)

def count_db(y_t):
    """
    count the decision bound of the batch y_t
    :param y_t:
    :return:
    """
    class_size = len(y_t)
    comb_num = int(comb(class_size, 2))
    db_indices = []
    for i, data in enumerate(y_t):
        indices  = torch.topk(data, 2).indices
        db_indice = count_db_detail(indices, comb_num)
        db_indices.append(db_indice)
    return db_indices
