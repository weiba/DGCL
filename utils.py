#获取结构网络和特征网络

import os

import math
import pickle
from sklearn import metrics
import torch

import torch.nn as nn
import torch.nn.functional as F

from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module


# from torch_geometric.utils import dropout_adj, negative_sampling, remove_self_loops,add_self_loops,to_dense_adj
import random
import numpy as np
import scipy.sparse as sp
import time
from torch.nn import Linear

def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

def ProcessNet(data):
    e = data.AugEdge
    e = e.cpu().numpy().T
    e = e.astype(np.int32)
    edges = e
    sedges = np.array(list(edges), dtype=np.int32).reshape(edges.shape)
    sadj = sp.coo_matrix((np.ones(sedges.shape[0]), (sedges[:, 0], sedges[:, 1])), shape=(data.x.shape[0], data.x.shape[0]), dtype=np.float32)
    sadj = sadj + sadj.T.multiply(sadj.T > sadj) - sadj.multiply(sadj.T > sadj)
    sadj = sparse_mx_to_torch_sparse_tensor(sadj)
    return sadj

