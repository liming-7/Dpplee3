'''
copyright@uestcliming
functions of parameter
'''
from __future__ import absolute_import

import numpy as np
import collections
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

def compute_updates(p1, p2):
    '''
    compute updates or the changes of two state_dict.
    '''
    res = collections.OrderedDict()
    for k,v in p1.items():
        res[k]=p1[k]-p2[k]
    return res



def add_params(p1, p2):
    '''
    Add two lists of parameters
    '''
    res = collections.OrderedDict()
    for k,v in p1.items():
        res[k]=p1[k]+p2[k]
    return res


def get_neutral(array):
    '''
    Get list of zero-valued numpy arrays for
    specified list of numpy arrays
    '''
    res = []
    for x in array:
        res.append(np.zeros_like(x))
    return res

'''感觉有点问题这里'''
def divide_by(array_list, num_workers):
    '''
    Divide a list of parameters by an integer num_workers.
    '''
    for i, x in enumerate(array_list):
        array_list[i] /= num_workers
    return array_list

def get_loss(loss_function, output, label, use_gpu):
    '''
    get objective loss of model and backprograte to compute gradients
    some loss function not impelement
    '''
    if not isinstance(loss_function, str):
        raise TypeError('loss_function should be str object')
    label =np.asarray(label)

    if loss_function == 'binary_cross_entropy':
        loss = F.binary_cross_entropy(output, label)
    elif loss_function == 'poisson_nll_loss':
        loss = F.poisson_nll_loss(output, label)
    elif loss_function == 'cross_entropy':
        loss = F.cross_entropy(output, label)
    elif loss_function == 'hinge_embedding_loss':
        loss = F.hinge_embedding_loss(output, label)
    elif loss_function == 'margin_ranking_loss':
        loss = F.margin_ranking_loss(output, label)
    elif loss_function == 'multilabel_soft_margin_loss':
        loss = F.multilabel_soft_margin_loss(output, label)
    elif loss_function == 'multi_margin_loss':
        loss = F.multi_margin_loss(output, label)
    elif loss_function == 'nll_loss':
        if use_gpu:
            label = Variable(torch.LongTensor(label).cuda())
        label = Variable(torch.LongTensor(label))
        loss = F.nll_loss(output, label)
    elif loss_function == 'binary_cross_entropy_with_logits':
        loss = F.binary_cross_entropy_with_logits(output, label)

    return loss

def get_optimizer(optimizer, optimizer_config, params):
    '''
    get the optimizer of worker model
    '''
    if optimizer == 'SGD':
        p = ['lr', 'momentum', 'dampening', 'weight_decay', 'nesterov']
        keys = list(optimizer_config.keys())
        unde = list(set(p)^set(keys))
        for i in unde:
            if i == 'nesterov':
                optimizer_config[i] = False
            else:
                optimizer_config[i] = 0

        opti = optim.SGD(params, lr=optimizer_config['lr'], momentum=optimizer_config['momentum'],
                dampening=optimizer_config['dampening'], weight_decay=optimizer_config['weight_decay'],
                nesterov=optimizer_config['nesterov'])
        return opti

    elif optimizer=='Rprop':
        p = ['lr', 'etas', 'step_sizes']
        keys = list(optimizer_config.keys())
        unde = list(set(p)^set(keys))
        for i in unde:
            if i == 'lr':
                optimizer_config[i] = 1e-2
            elif i == 'etas':
                optimizer_config[i] = (0.5, 1.2)
            elif i =='step_sizes':
                optimizer_config[i] = (1e-6, 50)

        opti = optim.Rprop(params, lr=optimizer_config['lr'], etas=optimizer_config['etas'], step_sizes=optimizer_config['step_sizes'])
        return opti

    elif optimizer=='RMSprop':
        p = ['lr', 'alpha', 'eps', 'weight_decay', 'momentum', 'centered']
        keys = list(optimizer_config.keys())
        unde = list(set(p)^set(keys))
        for i in unde:
            if i == 'lr':
                optimizer_config[i] = 1e-2
            elif i == 'alpha':
                optimizer_config[i] = 0.99
            elif i == 'eps':
                optimizer_config[i] = 1e-8
            elif i == 'weight_decay':
                optimizer_config[i] = 0
            elif i == 'momentum':
                optimizer_config[i] = 0
            elif i == 'centered':
                optimizer_config[i] = False

        opti = optim.SGD(params, lr=optimizer_config['lr'], alpha=optimizer_config['alpha'],
                eps=optimizer_config['eps'], weight_decay=optimizer_config['weight_decay'],
                momentum=optimizer_config['momentum'], centered = optimizer_config['centered'])
        return opti

    elif optimizer=='LBFGS':
        p = ['lr', 'max_iter', 'max_eval', 'tolerance_grad', 'tolerance_change', 'history_size', 'line_search_fn']
        keys = list(optimizer_config.keys())
        unde = list(set(p)^set(keys))
        for i in unde:
            if i == 'lr':
                optimizer_config[i] = 1
            elif i == 'max_iter':
                optimizer_config[i] = 20
            elif i == 'max_eval':
                optimizer_config[i] = None
            elif i == 'tolerance_grad':
                optimizer_config[i] = 1e-5
            elif i == 'tolerance_change':
                optimizer_config[i] = 1e-9
            elif i == 'history_size':
                optimizer_config[i] = 100
            elif i == 'line_search_fn':
                optimizer_config[i] = None

        opti = optim.SGD(params, lr=optimizer_config['lr'], max_iter=optimizer_config['max_iter'],
                tolerance_grad=optimizer_config['tolerance_grad'], tolerance_change=optimizer_config['tolerance_change'],
                history_size=optimizer_config['history_size'], line_search_fn = optimizer_config['line_search_fn'])
        return opti

    elif optimizer=='ASGD':
        p = ['lr', 'lambd', 'alpha', 't0', 'weight_decay']
        keys = list(optimizer_config.keys())
        unde = list(set(p)^set(keys))
        for i in unde:
            if i == 'lr':
                optimizer_config[i] = 1e-2
            elif i == 'lambd':
                optimizer_config[i] = 1e-4
            elif i == 'alpha':
                optimizer_config[i] = 0.75
            elif i == 't0':
                optimizer_config[i] = 1e-6
            elif i == 'weight_decay':
                optimizer_config[i] = 0

        opti = optim.SGD(params, lr=optimizer_config['lr'], lambd=optimizer_config['lambd'],
                alpha=optimizer_config['alpha'], t0=optimizer_config['t0'],
                weight_decay=optimizer_config['weight_decay'])
        return opti

    elif optimizer=='Adamax':
        p = ['lr', 'betas', 'eps', 'weight_decay']
        keys = list(optimizer_config.keys())
        unde = list(set(p)^set(keys))
        for i in unde:
            if i == 'lr':
                optimizer_config[i] = 0.002
            elif i == 'betas':
                optimizer_config[i] = (0.9, 0.999)
            elif i == 'eps':
                optimizer_config[i] = 1e-08
            elif i == 'weight_decay':
                optimizer_config[i] = 0


        opti = optim.SGD(params, lr=optimizer_config['lr'], betas=optimizer_config['betas'],
                eps=optimizer_config['eps'], weight_decay=optimizer_config['weight_decay'])
        return opti

    elif optimizer=='SparseAdam':
        p = ['lr', 'betas', 'eps']
        keys = list(optimizer_config.keys())
        unde = list(set(p)^set(keys))
        for i in unde:
            if i == 'lr':
                optimizer_config[i] = 0.001
            elif i == 'betas':
                optimizer_config[i] = (0.9, 0.999)
            elif i == 'eps':
                optimizer_config[i] = 1e-08

        opti = optim.SGD(params, lr=optimizer_config['lr'], betas=optimizer_config['betas'],
                eps=optimizer_config['eps'])
        return opti

    elif optimizer=='Adam':
        p = ['lr', 'betas', 'eps', 'weight_decay']
        keys = list(optimizer_config.keys())
        unde = list(set(p)^set(keys))
        for i in unde:
            if i == 'lr':
                optimizer_config[i] = 0.001
            elif i == 'betas':
                optimizer_config[i] = (0.9, 0.999)
            elif i == 'eps':
                optimizer_config[i] = 1e-08
            elif i == 'weight_decay':
                optimizer_config[i] = 0

        opti = optim.SGD(params, lr=optimizer_config['lr'], betas=optimizer_config['betas'],
                eps=optimizer_config['eps'], weight_decay=optimizer_config['weight_decay'])
        return opti

    elif optimizer=='Adagrad':
        p = ['lr', 'lr_decay', 'weight_decay']
        keys = list(optimizer_config.keys())
        unde = list(set(p)^set(keys))
        for i in unde:
            if i == 'lr':
                optimizer_config[i] = 0.01
            elif i == 'lr_decay':
                optimizer_config[i] = 0
            elif i == 'weight_decay':
                optimizer_config[i] = 0

        opti = optim.SGD(params, lr=optimizer_config['lr'], lr_decay=optimizer_config['lr_decay'],
                weight_decay=optimizer_config['weight_decay'])
        return opti

    elif optimizer=='Adadelta':
        p = ['lr', 'rho', 'eps', 'weight_decay']
        keys = list(optimizer_config.keys())
        unde = list(set(p)^set(keys))
        for i in unde:
            if i == 'lr':
                optimizer_config[i] = 1.0
            elif i == 'rho':
                optimizer_config[i] = 0.9
            elif i == 'eps':
                optimizer_config[i] = 1e-06
            elif i == 'weight_decay':
                optimizer_config[i] = 0

        opti = optim.SGD(params, lr=optimizer_config['lr'], rho=optimizer_config['rho'],
                eps=optimizer_config['eps'], weight_decay=optimizer_config['weight_decay'])
        return opti

    else:
        raise ValueError('the optimizer is exactly the same as the original pytorch, please check again!')
