"""
@Author: zhkun
@Time:  16:30
@File: utils
@Description:
@Something to attention
"""
import os
import time
import numpy as np
import math
import torch
from torch import nn
from torch.nn import functional as F

VERY_BIG_NUMBER = 1e30
VERY_SMALL_NUMBER = 1e-30
VERY_POSITIVE_NUMBER = VERY_BIG_NUMBER
VERY_NEGATIVE_NUMBER = -VERY_BIG_NUMBER


def mask(val, mask):
    device = 'cuda' if mask.is_cuda else 'cpu'
    return torch.mul(val, torch.tensor(mask, dtype=torch.float, device=device))


def exp_mask(val, mask):
    device = 'cuda' if mask.is_cuda else 'cpu'
    return torch.add(val, (1 - torch.tensor(mask, dtype=torch.float, device=device)) * VERY_NEGATIVE_NUMBER)


def softmax(logits, dim, mask=None, scale=1):
    if mask is not None:
        logits = exp_mask(logits, mask)

    out = F.softmax(logits * scale, dim=dim)
    return out


def softmax_utils(logits, axis, mask=None, scale=1, shift=0, temperature=1):

    if mask is not None:
        logits = exp_mask(logits, mask)

    if shift != 0:
        out = F.softmax(logits * scale * temperature, dim=axis)
    else:
        max_value = np.max(logits, axis)
        out = F.softmax((logits - max_value) * scale * temperature, dim=axis)

    return out

# better normalization for activation function
# https://arxiv.org/pdf/2006.12169.pdf
def activation(x, active='tanh'):
    if active == 'tanh':
        return 1.4674 * nn.Tanh() + 0.3885
    elif active == 'relu':
        return 1.4142 * nn.ReLU()
    elif active == 'leakyrelu':
        return 1.4141 * nn.LeakyReLU()
    elif active == 'elu':
        return 1.2234 * nn.ELU() + 0.0742
    elif active == 'selu':
        return 0.9660 * nn.SELU() + 0.2585
    elif active == 'gelu':
        return 1.4915 * nn.GELU() - 0.9097
    else:
        raise ValueError(f'wrong activation function {active}, will added in the future')


class MLP(nn.Module):
    def __init__(self, input_sizes, output_sizes, activation='relu'):
        super(MLP, self).__init__()
        if activation == 'relu':
            self.active = nn.ReLU()
        elif activation == 'tanh':
            self.active = nn.Tanh()
        elif activation == 'gelu':
            self.active = nn.GELU()
        elif activation == 'sigmoid':
            self.active = nn.Sigmoid()
        else:
            raise ValueError('wrong activation, please select relu, tanh, gelu, sigmoid')

        if not isinstance(input_sizes, list):
            self.linear = nn.Linear(input_sizes, output_sizes)
        else:
            self.linear = []
            for idx, sizes in enumerate(zip(input_sizes, output_sizes)):
                self.linear.append(nn.Linear(sizes[0], sizes[1]))

    def forward(self, x):
        if not isinstance(self.linear, list):
            result = self.linear(x)
        else:
            result = self.linear[0](x)

            for layer in self.linear[1:]:
                result = self.active(result)
                result = layer(result)

        return result


class CosineDecay(object):

    def __init__(self,
                 max_value,
                 min_value,
                 num_loops):
        self._max_value = max_value
        self._min_value = min_value
        self._num_loops = num_loops

    def get_value(self, i):
        if i < 0:
            i = 0
        if i >= self._num_loops:
            i = self._num_loops - 1
        value = (math.cos(i * math.pi / self._num_loops) + 1.0) * 0.5
        value = value * (self._max_value - self._min_value) + self._min_value
        return value


class CosineWarmUp(object):

    def __init__(self,
                 max_value,
                 min_value,
                 num_loops):
        self._max_value = max_value
        self._min_value = min_value
        self._num_loops = num_loops

    def get_value(self, i):
        if i < 0:
            i = 0
        if i >= self._num_loops:
            i = self._num_loops - 1
        i = i - self._num_loops + 1
        value = (math.cos(i * math.pi / self._num_loops) + 1.0) * 0.5
        value = value * (self._max_value - self._min_value) + self._min_value
        return value


class CosineWarmUpDecay(object):

    def __init__(self,
                 max_value,
                 min_value,
                 num_loops,
                 warm_up=0.05):
        self._max_value = max_value
        self._min_value = min_value
        self._num_loops = num_loops
        self._warm_up_p = warm_up
        self._warm_up = CosineWarmUp(max_value, min_value, int(num_loops * warm_up))
        self._decay = CosineDecay(max_value, min_value, int(num_loops * (1.0 - warm_up)))

    def get_value(self, i):
        if i < self._num_loops * self._warm_up_p:
            return self._warm_up.get_value(i)
        else:
            return self._decay.get_value(i)


class ExponentialDecay(object):

    def __init__(self,
                 init_value,
                 min_value,
                 num_loops):
        self._init_value = init_value
        self._min_value = min_value
        self._num_loops = num_loops

        self._value = init_value
        self._step = math.pow(min_value / init_value, 1.0 / num_loops)

    @property
    def value(self):
        return self._value

    def next(self):
        if self._value > self._min_value:
            self._value = self._value * self._step
            if self._value < self._min_value:
                self._value = self._min_value
        return self._value

    def reset(self):
        self._value = self._init_value


def prepar_data():
    if not os.path.exists('data'):
        os.mkdir('data')

    if not (os.path.exists('data/snli_1.0_train.txt')
            and os.path.exists('data/snli_1.0_dev.txt')
            and os.path.exists('data/snli_1.0_test.txt')):
        if not os.path.exists('data/snli_1.0.zip'):
            print('Downloading SNLI....')
            os.system('wget -P data https://nlp.stanford.edu/projects/snli/snli_1.0.zip')
        print('Unzipping SNLI....')
        os.system('unzip -d data/ data/snli_1.0.zip')
        os.system('mv -f data/snli_1.0/snli_*.txt data/')
    else:
        print('Found')


def get_current_time():
    return str(time.strftime('%Y%m%d%H%M%S', time.localtime(time.time())))[:-2]


def calc_eplased_time_since(start_time):
    curret_time = time.time()
    seconds = int(curret_time - start_time)
    hours = seconds // 3600
    seconds = seconds % 3600
    minutes = seconds // 60
    seconds = seconds % 60
    time_str = '{:0>2d}h{:0>2d}min{:0>2d}s'.format(hours, minutes, seconds)
    return time_str


def eval_map_mrr(qids, aids, preds, labels):
    # 衡量map指标和mrr指标
    dic = dict()
    pre_dic = dict()
    for qid, aid, pred, label in zip(qids, aids, preds, labels):
        pre_dic.setdefault(qid, [])
        pre_dic[qid].append([aid, pred, label])
    for qid in pre_dic:
        dic[qid] = sorted(pre_dic[qid], key=lambda k: k[1], reverse=True)
        aid2rank = {aid: [label, rank] for (rank, (aid, pred, label)) in enumerate(dic[qid])}
        dic[qid] = aid2rank

    MAP = 0.0
    MRR = 0.0
    useful_q_len = 0
    for q_id in dic:
        sort_rank = sorted(dic[q_id].items(), key=lambda k: k[1][1], reverse=False)
        correct = 0
        total = 0
        AP = 0.0
        mrr_mark = False
        for i in range(len(sort_rank)):
            if sort_rank[i][1][0] == 1:
                correct += 1
        if correct == 0:
            continue
        useful_q_len += 1
        correct = 0
        for i in range(len(sort_rank)):
            # compute MRR
            if sort_rank[i][1][0] == 1 and mrr_mark == False:
                MRR += 1.0 / float(i + 1)
                mrr_mark = True
            # compute MAP
            total += 1
            if sort_rank[i][1][0] == 1:
                correct += 1
                AP += float(correct) / float(total)

        AP /= float(correct)
        MAP += AP

    MAP /= useful_q_len
    MRR /= useful_q_len
    return MAP, MRR


def write_file(file_name, content):
    with open(file_name, 'a', encoding='utf8') as w:
        for item in content:
            w.write(item + '\n')


class BiClassCalculator(object):
    def __init__(self):
        self._tp = 0
        self._tn = 0
        self._fp = 0
        self._fn = 0

    def update(self, label_predict, label_true):
        hit = np.equal(label_predict, label_true)
        hit = np.float32(hit)
        miss = 1.0 - hit

        pos = np.float32(label_predict)
        neg = 1.0 - pos

        self._tp += np.sum(hit * pos, keepdims=False)
        self._tn += np.sum(hit * neg, keepdims=False)
        self._fp += np.sum(miss * pos, keepdims=False)
        self._fn += np.sum(miss * neg, keepdims=False)

    @property
    def precision(self):
        num_pos_pred = self._tp + self._fp
        return self._tp / num_pos_pred if num_pos_pred > 0 else math.nan

    @property
    def recall(self):
        num_pos_true = self._tp + self._fn
        return self._tp / num_pos_true if num_pos_true > 0 else math.nan

    @property
    def f1(self):
        pre = self.precision
        rec = self.recall
        return 2 * (pre * rec) / (pre + rec)

    @property
    def accuracy(self):
        num_hit = self._tp + self._tn
        num_all = self._tp + self._tn + self._fp + self._fn
        return num_hit / num_all if num_all > 0 else math.nan

