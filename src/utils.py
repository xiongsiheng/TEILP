import numpy as np
import math
import sys
import os
import time
import pickle

from collections import Counter
import copy
import random
import json
from datetime import datetime
from joblib import Parallel, delayed


import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.stats import expon
from scipy.stats import shapiro, normaltest

import pandas as pd

import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import kstest, norm
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.metrics import mean_absolute_error

import itertools




class Option(object):
    def __init__(self, d):
        self.__dict__ = d
    def save(self):
        with open(os.path.join(self.this_expsdir, "option.txt"), "w") as f:
            for key, value in sorted(self.__dict__.items(), key=lambda x: x[0]):
                f.write("%s, %s\n" % (key, str(value)))



def weighted_mse_loss(output, target, scale, threshold, weight_high, weight_low):
    assert output.shape == target.shape, "Input and target tensors must have the same shape"
    device = output.device
    scale = torch.tensor(scale, device=device)
    # threshold = torch.tensor(threshold, device=device)
    weights = torch.where(torch.abs(target/scale - torch.tensor(0.5, device=device)) > threshold, 
                            torch.tensor(weight_high, device=device), torch.tensor(weight_low, device=device))
    # print(target.flatten())
    # print(weights.flatten())
    squared_diffs = (output - target) ** 2
    weighted_squared_diffs = squared_diffs * weights
    return torch.mean(weighted_squared_diffs)




def obtain_feature_maps(targ_rel, dataset, edges, num_train, num_rel, maxNum_used_for_training = None, 
                            mode = 'total', idx_range = None, allRel_padding = False, 
                            with_prior=False, prior = None, collect_all_data=False, only_collect_time=True):
    flag_enhancement = 0
    flag_use_shifting_data_for_training = 0
    outDict1 = []
    rel_collector = {}

    keys_ls = ['hh', 'ht', 'th', 'tt']
    num1_ls = [0, 0, 2, 2]
    num2_ls = [0, 2, 0, 2]

    for k in keys_ls:
        rel_collector[k] = []

    train_edges = edges[:num_train]
    train_idx = np.arange(num_train)

    num_split = [0, 0]
    if mode == 'total':
        if not idx_range:
            idx_range = range(len(edges))

    for j in idx_range:
        e = edges[j]
        if e[1] not in targ_rel:
            continue

        masked_train_edges = train_edges[~np.all(train_edges==e, axis=1)]

        if with_prior:
            if prior is not None:
                masked_train_edges = masked_train_edges[masked_train_edges[:,3] < prior[idx_range.index(j)]]
            else:
                masked_train_edges = masked_train_edges[masked_train_edges[:,3] < e[3]]

        if j < num_train:
            if maxNum_used_for_training:
                if num_split[0] >= maxNum_used_for_training:
                    continue
            num_split[0] += 1
        else:
            num_split[1] += 1


        if dataset in ['gdelt', 'icews14', 'icews05-15']:
            if only_collect_time:
                outDict = {'Target':{int(e[1]): e[3]}}
            else:
                outDict = {'Target':{int(e[1]): [e[0], e[1], e[2], e[3]]}}
        else:
            if only_collect_time:
                outDict = {'Target':{int(e[1]): [e[3], e[4]]}}
            else:
                outDict = {'Target':{int(e[1]): [e[0], e[1], e[2], e[3], e[4]]}}


        for i in range(4):
            k = keys_ls[i]
            num1 = num1_ls[i]
            num2 = num2_ls[i]
            outDict[k] = {}
            for rel in range(num_rel):
                x = masked_train_edges[(masked_train_edges[:,num2] == e[num1]) & (masked_train_edges[:,1] == rel)]
                if flag_use_shifting_data_for_training:
                    x = masked_train_edges[(masked_train_edges[:,num2] == e[num1]) & \
                                           (masked_train_edges[:,1] == rel) & \
                                           (masked_train_edges[:,3] <= e[3] - t_shift_for_training)]
                if len(x) > 0:
                    rel_collector[k].append(rel)
                    if collect_all_data:
                        if only_collect_time:
                            outDict[k][rel] = x[:, 3:]
                        else:
                            outDict[k][rel] = x
                    else:
                        x1 = my_shuffle(x)
                        if only_collect_time:
                            outDict[k][rel] = x1[0,3:5]
                            if flag_enhancement:
                                outDict[k][rel] = [np.min(x1[:,3]), np.max(x1[:,3])]

        outDict1.append(outDict)

    if allRel_padding:
        for k in keys_ls:
            rel_collector[k] = list(range(num_rel))

    if mode == 'split':
        return rel_collector, outDict1, num_split

    X, y = input_padding(targ_rel, dataset, keys_ls, rel_collector, outDict1)

    return X, y, num_split






def obtain_feature_maps_v2(targ_rel, dataset, edges, num_train, maxNum_used_for_training = None, 
                            mode = 'total', idx_range = None, allRel_padding = False, edge_range = None,
                            prior = None, with_prior=False, collect_all_data=False):
    outDict1 = []
    rel_collector = {}

    keys_ls = ['hhtt', 'htth']
    num1_ls = [0, 0]
    num2_ls = [0, 2]
    num3_ls = [2, 2]
    num4_ls = [2, 0]

    for k in keys_ls:
        rel_collector[k] = []

    train_edges = edges[:num_train]
    train_idx = np.arange(num_train)

    num_split = [0, 0]
    if mode == 'total':
        if not idx_range:
            idx_range = range(len(edges))

    if (idx_range is None) and (edge_range is not None):
        idx_range = edge_range

    for (j, idx) in enumerate(idx_range):
        if edge_range is None:
            e = edges[idx]
        else:
            e = copy.copy(idx)

        if e[1] not in targ_rel:
            continue

        masked_train_edges = train_edges[~np.all(train_edges==e, axis=1)]

        if with_prior:
            if prior is not None:
                masked_train_edges = masked_train_edges[masked_train_edges[:,3] < prior[j]]
            else:
                masked_train_edges = masked_train_edges[masked_train_edges[:,3] < e[3]]


        if edge_range is None:
            if idx < num_train:
                if maxNum_used_for_training is not None:
                    if num_split[0] >= maxNum_used_for_training:
                        continue
                num_split[0] += 1
            else:
                num_split[1] += 1


        if dataset in ['gdelt', 'icews14', 'icews05-15']:
            outDict = {'Target':{e[1]: e[3]}}
        else:
            outDict = {'Target':{e[1]: [e[3], e[4]]}}

        for i in range(len(keys_ls)):
            k = keys_ls[i]
            num1 = num1_ls[i]
            num2 = num2_ls[i]
            num3 = num3_ls[i]
            num4 = num4_ls[i]
            outDict[k] = {}
            for rel in range(num_rel):
                x = masked_train_edges[(masked_train_edges[:,num2] == e[num1]) & (masked_train_edges[:,1] == rel) \
                                        & (masked_train_edges[:,num4] == e[num3])]
                if len(x) > 0:
                    rel_collector[k].append(rel)
                    if collect_all_data:
                        outDict[k][rel] = x
                    else:
                        x1 = my_shuffle(x)
                        outDict[k][rel] = x1[0,3:5]

        outDict1.append(outDict)


    if allRel_padding:
        for k in keys_ls:
            rel_collector[k] = list(range(num_rel))

    if mode == 'split':
        return rel_collector, outDict1, num_split

    X, y = input_padding(targ_rel, dataset, keys_ls, rel_collector, outDict1)

    return X, y, num_split




def obtain_feature_maps_v3(targ_rel, dataset, edges, num_train, maxNum_used_for_training = None, 
                            mode = 'total', idx_range = None, allRel_padding = False, edge_range = None):
    outDict1 = []
    rel_collector = {}

    keys_ls = ['hhtt', 'htth']
    num1_ls = [0, 0]
    num2_ls = [0, 2]
    num3_ls = [2, 2]
    num4_ls = [2, 0]

    for k in keys_ls:
        rel_collector[k] = []

    train_edges = edges[:num_train]
    train_idx = np.arange(num_train)

    num_split = [0, 0]
    if mode == 'total':
        if not idx_range:
            idx_range = range(len(edges))

    if (idx_range is None) and (edge_range is not None):
        idx_range = edge_range

    for j in idx_range:
        if edge_range is None:
            e = edges[j]
        else:
            e = copy.copy(j)

        if e[1] not in targ_rel:
            continue

        masked_train_edges = train_edges[~np.all(train_edges==e, axis=1)]

        if edge_range is None:
            if j < num_train:
                if maxNum_used_for_training is not None:
                    if num_split[0] >= maxNum_used_for_training:
                        continue
                num_split[0] += 1
            else:
                num_split[1] += 1


        if dataset in ['gdelt', 'icews14', 'icews05-15']:
            outDict = {'Target':{e[1]: e[3]}}
        else:
            outDict = {'Target':{e[1]: [e[3], e[4]]}}

        for i in range(len(keys_ls)):
            k = keys_ls[i]
            num1 = num1_ls[i]
            num2 = num2_ls[i]
            num3 = num3_ls[i]
            num4 = num4_ls[i]
            outDict[k] = {}
            for rel in range(num_rel):
                x = masked_train_edges[(masked_train_edges[:,num2] == e[num1]) & (masked_train_edges[:,1] == rel) \
                                        & (masked_train_edges[:,num4] == e[num3])]
                if len(x) > 0:
                    rel_collector[k].append(rel)
                    x1 = my_shuffle(x)
                    outDict[k][rel] = x1[:,3]

        outDict1.append(outDict)

    if allRel_padding:
        for k in keys_ls:
            rel_collector[k] = list(range(num_rel))

    if mode == 'split':
        return rel_collector, outDict1, num_split

    X, y = input_padding_v2(targ_rel, dataset, keys_ls, rel_collector, outDict1)

    return X, y, num_split





def input_padding(targ_rel, dataset, keys_ls, rel_collector, outDict1):
    for i in range(len(keys_ls)):
        k = keys_ls[i]
        rel_collector[k] = list(set(rel_collector[k]))
        rel_collector[k].sort()

    X = []
    y = []
    for j in range(len(outDict1)):
        input_ls = []
        for i in range(len(keys_ls)):
            k = keys_ls[i]
            input_ls1 = []

            for rel in rel_collector[k]:
                if rel in outDict1[j][k]:
                    input_ls1 += outDict1[j][k][rel].tolist()
                else:
                    if dataset in ['gdelt', 'icews14', 'icews05-15']:
                        input_ls1 += [9999]
                    else:
                        input_ls1 += [9999, 9999]

            input_ls += input_ls1

        if len(input_ls) == 0:
            if dataset in ['gdelt', 'icews14', 'icews05-15']:
                input_ls = [9999]
            else:
                input_ls = [9999, 9999]

        X.append(input_ls)
        y.append(list(outDict1[j]['Target'].values())[0])

    X = np.array(X)
    y = np.array(y)

    return X, y




def input_padding_v2(targ_rel, dataset, keys_ls, rel_collector, outDict1):
    for i in range(len(keys_ls)):
        k = keys_ls[i]
        rel_collector[k] = list(set(rel_collector[k]))
        rel_collector[k].sort()

    X = []
    y = []
    for j in range(len(outDict1)):
        input_ls = []
        for i in range(len(keys_ls)):
            k = keys_ls[i]
            input_ls1 = []

            for rel in rel_collector[k]:
                if rel in outDict1[j][k]:
                    input_ls1.append(outDict1[j][k][rel].tolist())
                else:
                    if dataset in ['gdelt', 'icews14', 'icews05-15']:
                        input_ls1.append([9999])
                    else:
                        input_ls1.append([9999, 9999])

            input_ls += input_ls1

        if len(input_ls) == 0:
            if dataset in ['gdelt', 'icews14', 'icews05-15']:
                input_ls = [[9999]]
            else:
                input_ls = [[9999, 9999]]

        input_ls = join_vectors_with_different_lengths(input_ls)

        X.append(input_ls)
        y.append(list(outDict1[j]['Target'].values())[0])

    X = join_arrays_with_different_lengths(X)
    y = np.array(y)

    return X, y




def join_vectors_with_different_lengths(list_of_vectors):
    # Find the maximum length among the vectors
    max_length = max(len(vector) for vector in list_of_vectors)

    # Pad the shorter vectors with a specific value (e.g., 0) and store them as NumPy arrays
    padded_vectors = [np.pad(vector, (0, max_length - len(vector)), constant_values=9999) for vector in list_of_vectors]
    # print(list_of_vectors)
    # print('--------------------')

    # Combine the padded vectors into a single 2D NumPy array
    combined_array = np.vstack(padded_vectors)

    # print(combined_array)
    return combined_array


def join_arrays_with_different_lengths(list_of_arrays):
    # for x in list_of_arrays:
    #     print(x.shape)
    # Find the maximum size along the first dimension among the 2D arrays
    max_size = max(array.shape[1] for array in list_of_arrays)

    # Pad the smaller 2D arrays with a specific value (e.g., 0) along the first dimension
    padded_arrays = [np.pad(array, ((0, 0), (0, max_size - array.shape[1])), constant_values=9999) for array in list_of_arrays]

    # Combine the padded 2D arrays into a single 3D array
    combined_array = np.stack(padded_arrays, axis=0)

    # print(combined_array)
    return combined_array





def my_shuffle(data):
    # np.random.seed(12)
    idx = list(range(len(data)))
    random.shuffle(idx)
    new_data = data[idx]
    return new_data




def data_loader(edges, num_train, rel):
    idxs = np.arange(len(edges))
    train_idx = [idx for idx in idxs[edges[:,1] == rel] if idx < num_train]
    test_idx = [idx for idx in idxs[edges[:,1] == rel] if idx >= num_train]

    successful = 0
    if (len(train_idx) == 0) or (len(test_idx) == 0):
        return [], successful, []

    random.shuffle(train_idx)
    train_idx = train_idx[:maxTotalNum_used_for_training_single_rel]
    random.shuffle(test_idx)
    test_idx = test_idx[:maxTotalNum_used_for_test_single_rel]


    inputs0, targets, _ = obtain_feature_maps_v2([rel], dataset, edges, num_train, mode='split2', 
                                idx_range = train_idx + test_idx,
                                allRel_padding = False)


    inputs0 = inputs0 // resolution
    # inputs = inputs0.copy()
    # for (idx1, t1) in enumerate(t_ls):
    #     inputs[inputs0 == t1] = idx1

    train_inputs = inputs0[:len(train_idx),:]
    test_inputs = inputs0[len(train_idx):,:]
    train_targets = targets[:len(train_idx)]
    test_targets = targets[len(train_idx):]

    successful = 1

    return [train_inputs, train_targets, test_inputs, test_targets], successful, [train_idx, test_idx]



def data_loader_dist(edges, num_train, rel, with_prior=False, num_workers = 2):
    idxs = np.arange(len(edges))
    train_idx = [idx for idx in idxs[edges[:,1] == rel] if idx < num_train]
    test_idx = [idx for idx in idxs[edges[:,1] == rel] if idx >= num_train]

    successful = 0
    if (len(train_idx) == 0) or (len(test_idx) == 0):
        return [], successful, []

    random.shuffle(train_idx)
    train_idx = train_idx[:maxTotalNum_used_for_training_single_rel]
    # random.shuffle(test_idx)
    test_idx = test_idx[:maxTotalNum_used_for_test_single_rel]


    index_pieces = split_list_into_pieces(train_idx + test_idx, num_workers)
    results = Parallel(n_jobs=num_workers)(delayed(obtain_feature_maps)([rel], dataset, edges, num_train, mode = 'split', 
                                            idx_range = piece, allRel_padding = True, with_prior = with_prior) for piece in index_pieces)

    rel_collector, outDict, _ = results[0]
    for result in results[1:]:
        for k in rel_collector:
            rel_collector[k] += result[0][k]
        outDict += result[1]

    keys_ls = ['hh', 'ht', 'th', 'tt']
    # keys_ls = ['hhtt', 'htth']

    inputs, targets = input_padding([rel], dataset, keys_ls, rel_collector, outDict)
    inputs = inputs // resolution

    train_inputs = inputs[:len(train_idx),:]
    test_inputs = inputs[len(train_idx):,:]
    train_targets = targets[:len(train_idx),:]
    test_targets = targets[len(train_idx):,:]

    successful = 1

    return [train_inputs, train_targets, test_inputs, test_targets], successful, [train_idx, test_idx]





def data_loader_with_prior(edges, num_train, rel, data_idx, xgb_preds):
    train_idx, test_idx = data_idx

    inputs0, targets, _ = obtain_feature_maps_v2([rel], dataset, edges, num_train, mode='split2', 
                                idx_range = train_idx + test_idx,
                                allRel_padding = False, prior=xgb_preds)

    inputs0 = inputs0 // resolution

    train_inputs = inputs0[:len(train_idx),:]
    test_inputs = inputs0[len(train_idx):,:]
    train_targets = targets[:len(train_idx)]
    test_targets = targets[len(train_idx):]

    return [train_inputs, train_targets, test_inputs, test_targets]



def data_loader_with_prior_dist(edges, num_train, rel, data_idx, prior, num_workers = 2):
    _, test_idx = data_idx

    index_pieces = split_list_into_pieces(test_idx, num_workers)
    results = Parallel(n_jobs=num_workers)(delayed(obtain_feature_maps)([rel], dataset, edges, num_train, mode='split', 
                                            idx_range = piece, allRel_padding = True, prior=prior,
                                            with_prior = True) for piece in index_pieces)

    rel_collector, outDict, _ = results[0]
    for result in results[1:]:
        for k in rel_collector:
            rel_collector[k] += result[0][k]
        outDict += result[1]


    keys_ls = ['hh', 'ht', 'th', 'tt']
    # keys_ls = ['hhtt', 'htth']

    inputs, targets = input_padding([rel], dataset, keys_ls, rel_collector, outDict)
    inputs = inputs // resolution

    # train_inputs = inputs[:len(train_idx),:]
    test_inputs = inputs
    # train_targets = targets[:len(train_idx)]
    test_targets = targets

    return [_, _, test_inputs, test_targets]





def split_list_into_batches(lst, batch_size):
    """
    Splits a list into batches of a given size.
    """
    return [lst[i:i + batch_size] for i in range(0, len(lst), batch_size)]


def split_list_into_pieces(all_indices, num_pieces):
    indices_per_piece = len(all_indices) // num_pieces
    index_pieces = [all_indices[i:i + indices_per_piece] for i in range(0, len(all_indices), indices_per_piece)]
    if len(index_pieces)>num_pieces:
        output = index_pieces[:num_pieces-1] + [merge_list(index_pieces[num_pieces-1:])]
    else:
        output = index_pieces
    return output


def merge_list(ls_ls):
    output = []
    for ls in ls_ls:
        output += ls
    return output



def interval_intersection(interval1, interval2):
    a, b = interval1
    c, d = interval2
    if b < c or d < a:
        return 0
    else:
        return min(b, d)-max(a, c)

def interval_union(interval1, interval2):
    a, b = interval1
    c, d = interval2
    if b < c or d < a:
        return b-a + d-c
    else:
        return max(b, d) - min(a, c)

def interval_convex(interval1, interval2):
    a, b = interval1
    c, d = interval2
    return round(max(b, d) - min(a, c))



def obtain_aeIoU(preds, ys):
    aeIoU = []
    for i in range(len(preds)):
        pred = [min(preds[i]), max(preds[i])]
        y = ys[i]
        if interval_convex(pred, y) == 0:
            aeIoU.append(1.)
        else:
            x = max(1, interval_intersection(pred, y))*1./interval_convex(pred, y)
            aeIoU.append(x)
    return aeIoU


def obtain_TAC(preds, ys):
    TAC = []
    for i in range(len(preds)):
        pred = [min(preds[i]), max(preds[i])]
        y = ys[i]
        TAC.append(0.5*(1./(1+abs(pred[0] - y[0])) + 1./(1+abs(pred[1] - y[1]))))
    return TAC


def obtain_ranking(probs, ys, timestamp_range):
    rank = []
    for i in range(len(probs)):
        y = ys[i]
        y = timestamp_range.index(y)
        rank.append(len(probs[i][probs[i]>probs[i][y]]) + 1)
    return rank


def obtain_MSE(preds, ys):
    preds = preds.reshape((-1,))
    ys = ys.reshape((-1,))
    mse = (preds - ys)**2
    return mse.tolist()


def read_dataset_txt(path):
    edges = []
    with open(path, 'r') as f:
        lines=f.readlines()
        for l in lines:
            a=l.strip().split()
            a=[int(x) for x in a]
            b=copy.copy(a)
            if len(b)>4:
                b[3] = min(a[3], a[4])
                b[4] = max(a[3], a[4])
            edges.append(b)
    return edges



def read_dataset_txt_v2(path, dataset):
    edges = []
    if dataset in ['gdelt']:
        start_date_str = '2015-01-01'
    elif dataset in ['icews14']:
        start_date_str = '2014-01-01'
        # if 'test' in path:
            # start_date_str = '2014/1/1'
    elif dataset in ['icews05-15']:
        start_date_str = '2005-01-01'

    with open(path, 'r') as f:
        lines=f.readlines()
        for l in lines:
            a=l.strip().split()
            a=[int(x) for x in a[:3]] + [days_between_dates(start_date_str, a[3])]
            edges.append(a)

    return edges



def days_between_dates(start_date_str, end_date_str):
    if '-' in start_date_str:
        date_format = "%Y-%m-%d"
    else:
        date_format = "%Y/%m/%d"
    start_date = datetime.strptime(start_date_str, date_format)
    end_date = datetime.strptime(end_date_str, date_format)
    delta = end_date - start_date
    return delta.days



# def obtain_all_data(dataset, shuffle_train_set=True, flag_use_valid=1):
#     # flag_use_valid = 1

#     if dataset in ['icews14', 'icews05-15']:
#         edges1 = read_dataset_txt_v2('../data/'+ dataset +'/train.txt', dataset)
#     else:
#         edges1 = read_dataset_txt('../data/'+ dataset +'/train.txt')
#     edges1 = np.array(edges1)

#     if shuffle_train_set:
#         edges1 = my_shuffle(edges1)

#     len_edges2 = 0
#     edges2 = None
#     if flag_use_valid:
#         if dataset in ['icews14', 'icews05-15']:
#             edges2 = read_dataset_txt_v2('../data/'+ dataset +'/valid.txt', dataset)
#         else:
#             edges2 = read_dataset_txt('../data/'+ dataset +'/valid.txt')
#         edges2 = np.array(edges2)
#         len_edges2 = len(edges2)
#         # print(len(edges1), len_edges2)


#     if dataset in ['icews14', 'icews05-15']:
#         edges3 = read_dataset_txt_v2('../data/'+ dataset +'/test.txt', dataset)
#     else:
#         edges3 = read_dataset_txt('../data/'+ dataset +'/test.txt')
#     edges3 = np.array(edges3)


#     if flag_use_valid:
#         edges = np.vstack([edges1, edges2, edges3])
#     else:
#         edges = np.vstack([edges1, edges3])

#     num_train = len(edges1) + len_edges2

#     return edges1, edges2, edges3


def obtain_all_data(dataset, shuffle_train_set=True, flag_use_valid=1):
    edges1 = read_dataset_txt('../data/'+ dataset +'/train.txt')
    edges1 = np.array(edges1)

    if shuffle_train_set:
        edges1 = my_shuffle(edges1)

    len_edges2 = 0
    edges2 = None
    if flag_use_valid:
        edges2 = read_dataset_txt('../data/'+ dataset +'/valid.txt')
        edges2 = np.array(edges2)
        len_edges2 = len(edges2)
        # print(len(edges1), len_edges2)

    edges3 = read_dataset_txt('../data/'+ dataset +'/test.txt')
    edges3 = np.array(edges3)


    if flag_use_valid:
        edges = np.vstack([edges1, edges2, edges3])
    else:
        edges = np.vstack([edges1, edges3])

    num_train = len(edges1) + len_edges2

    return edges1, edges2, edges3



def prune_dataset(dataset, preserved_num_entity=100):
    for mode in ['train', 'valid', 'test']:
        if dataset in ['gdelt', 'icews14', 'icews05-15']:
            edges = read_dataset_txt_v2('../data/'+ dataset +'/'+ mode +'.txt', dataset)
        else:
            edges = read_dataset_txt('../data/'+ dataset +'/'+ mode +'.txt')
        edges = np.array(edges)

        edges = edges[np.isin(edges[:,0], range(preserved_num_entity)) & np.isin(edges[:,2], range(preserved_num_entity))]

        lines = []
        for edge in edges:
            lines.append('\t'.join([str(x) for x in edge]) + '\n')

        with open('../data/'+ dataset +'/'+ mode +'_pruned_'+ str(preserved_num_entity) +'.txt', 'w') as f:
            f.writelines(lines)


def rm_repetition_dataset(dataset):
    total_edges = []
    for mode in ['train', 'valid', 'test']:
        if dataset in ['gdelt', 'icews14', 'icews05-15']:
            edges = read_dataset_txt('../data/'+ dataset +'/'+ mode +'_pruned_100.txt')
        else:
            edges = read_dataset_txt('../data/'+ dataset +'/'+ mode +'.txt')

        total_edges += edges

    total_edges = np.array(total_edges)
    total_edges = np.unique(total_edges, axis = 0)
    total_edges = np.random.permutation(total_edges)

    new_edges_ls = [total_edges[:int(0.8*len(total_edges))],
                    total_edges[int(0.8*len(total_edges)):int(0.9*len(total_edges))],
                    total_edges[int(0.9*len(total_edges)):]]

    for (i, mode) in enumerate(['train', 'valid', 'test']):
        lines = []
        for edge in new_edges_ls[i]:
            lines.append('\t'.join([str(x) for x in edge]) + '\n')

        with open('../data/'+ dataset +'/'+ mode +'_unique.txt', 'w') as f:
            f.writelines(lines)





def do_temporal_shift_to_dataset(dataset, time_shift_mode=1, flag_save_idx=False, flag_use_valid_for_training=0, flag_save_data=False):
    if dataset in ['icews14', 'icews05-15']:
        edges1 = read_dataset_txt_v2('../data/'+ dataset +'/train.txt', dataset)
        edges2 = read_dataset_txt_v2('../data/'+ dataset +'/valid.txt', dataset)
        edges3 = read_dataset_txt_v2('../data/'+ dataset +'/test.txt', dataset)
    else:
        edges1 = read_dataset_txt('../data/' + dataset + '/train.txt')
        edges2 = read_dataset_txt('../data/' + dataset + '/valid.txt')
        edges3 = read_dataset_txt('../data/' + dataset + '/test.txt')

    edges = np.vstack((edges1, edges2, edges3))
    idx_ls = np.array(range(len(edges))).reshape((-1,1))
    # print(len(edges))
    # print(len(edges1), len(edges2), len(edges3))
    # print(idx_ls)
    # sys.exit()

    edges = np.hstack((edges, idx_ls))

    if time_shift_mode in [0,1]:
        edges = edges[np.argsort(edges[:, 3])]
    elif time_shift_mode == -1:
        edges = edges[np.argsort(edges[:, 3])][::-1]

    if time_shift_mode == 0:
        anchor = int(len(edges)*0.5)
        train_edges = np.vstack([edges[:anchor-len(edges2)], edges[anchor+len(edges3):]])
        valid_data = edges[anchor-len(edges2):anchor]
        test_data = edges[anchor:anchor + len(edges3)]
    else:
        train_edges = edges[:len(edges1)]
        valid_data = edges[len(edges1):len(edges1)+len(edges2)]
        test_data = edges[len(edges1)+len(edges2):]

    train_idx = train_edges[:,-1].tolist()
    valid_idx = valid_data[:,-1].tolist()
    test_idx = test_data[:,-1].tolist()
    train_edges = train_edges[:,:-1]
    valid_data = valid_data[:,:-1]
    test_data = test_data[:,:-1]

    if flag_save_data:
        lines = []
        for edge in train_edges:
            lines.append('\t'.join([str(x) for x in edge]) + '\n')

        with open('../data/'+ dataset +'/train_time_shifting.txt', 'w') as f:
            f.writelines(lines)

        lines = []
        for edge in valid_data:
            lines.append('\t'.join([str(x) for x in edge]) + '\n')

        with open('../data/'+ dataset +'/valid_time_shifting.txt', 'w') as f:
            f.writelines(lines)

        lines = []
        for edge in test_data:
            lines.append('\t'.join([str(x) for x in edge]) + '\n')

        with open('../data/'+ dataset +'/test_time_shifting.txt', 'w') as f:
            f.writelines(lines)


    if not flag_use_valid_for_training:
        edges = np.vstack((train_edges, test_data))
        num_train = len(edges1)
    else:
        edges = np.vstack((train_edges, valid_data, test_data))
        num_train = [len(edges1), len(edges2)]


    if flag_save_idx:
        with open('../data/' + dataset + '_time_shift_idx.json','w') as f:
            json.dump({'train_idx':train_idx, 'valid_idx':valid_idx, 'test_idx':test_idx}, f)

    return edges, num_train





def create_padding_mask(seq, targ_value):
    seq1 = (seq == targ_value).float()
    return seq1.unsqueeze(1).unsqueeze(2)



def create_attn_mask(seq, targ_value):
    seq1 = (seq == targ_value).float()
    return 1 - seq1



def create_indices_with_given_ranges(vec, bounds):
    indices = []
    for i in range(len(bounds)):
        # Find indices of elements within range
        indices.append(np.where((vec >= bounds[i][0]) & (vec <= bounds[i][1]))[0])
    return indices



def num_sample_stat(rel_ls, edges, num_train):
    num_sample_dict = {}
    idxs = np.arange(len(edges))
    for rel in rel_ls:
        # print('Relation:', rel)
        train_idx = [idx for idx in idxs[edges[:,1] == rel] if idx < num_train]
        test_idx = [idx for idx in idxs[edges[:,1] == rel] if idx >= num_train]
        num_sample_dict[rel] = [len(train_idx), len(test_idx)]
 
    dist_rel_ls = []
    sorted_dict = sorted(num_sample_dict.items(), key=lambda item: item[1][0])
    # print(sorted_dict)
    num_sample_int_ls = [[0, 100], [100, 500], [500, 1000], [1000, 5000], [5000, 10000], [10000, 100000000]]
    for i in range(len(num_sample_int_ls)):
        result = [x[0] for x in sorted_dict if num_sample_int_ls[i][0] <= x[1][0] <= num_sample_int_ls[i][1]]
        if num_sample_int_ls[i][0] >= 5000:
            result = [[x] for x in result]
            dist_rel_ls += result
        else:
            dist_rel_ls.append(result)

    return dist_rel_ls





def obtain_inv_edge(edges, num_rel):
    return np.hstack([edges[:,2:3], edges[:,1:2] + num_rel, edges[:,0:1], edges[:,3:]])







def obtain_gap_stat(targ_rel, idx_range, edge_range=None):
    keys_ls = ['h_t', 'h_nt', 'nh_t'][:1]
    num1_ls = [0, 0, 0]
    num2_ls = [0, 0, 0]
    num3_ls = [2, 2, 2]
    num4_ls = [2, 2, 2]

    outDict = {}
    for k in keys_ls:
        outDict[k] = {}

    train_edges = edges[:num_train,:]
    train_edges_inv = obtain_inv_edge(train_edges, num_rel)
    train_edges_cmp = np.vstack((train_edges, train_edges_inv))

    for (j, idx) in enumerate(idx_range):
        if edge_range is None:
            e = edges[idx]
        else:
            e = copy.copy(idx)

        e_inv = obtain_inv_edge(e.reshape((1, -1)), num_rel)[0]

        if e[1] not in targ_rel:
            continue

        # print(e, e_inv)

        masked_train_edges = train_edges_cmp[~np.all(train_edges_cmp == e, axis=1)]
        masked_train_edges = masked_train_edges[~np.all(masked_train_edges == e_inv, axis=1)]
        # masked_train_edges = masked_train_edges[masked_train_edges[:,3] <= e[3]]

        for i in range(len(keys_ls)):
            k = keys_ls[i]
            num1 = num1_ls[i]
            num2 = num2_ls[i]
            num3 = num3_ls[i]
            num4 = num4_ls[i]

            for rel in range(num_rel):
                if k == 'h_nt':
                    x = masked_train_edges[(e[num1] == masked_train_edges[:,num2]) & (masked_train_edges[:,1] == rel) \
                                            & (e[num3] != masked_train_edges[:,num4])]
                elif k == 'nh_t':
                    x = masked_train_edges[(e[num1] != masked_train_edges[:,num2]) & (masked_train_edges[:,1] == rel) \
                                            & (e[num3] == masked_train_edges[:,num4])]
                else:
                    x = masked_train_edges[(e[num1] == masked_train_edges[:,num2]) & (masked_train_edges[:,1] == rel) \
                                            & (e[num3] == masked_train_edges[:,num4])]

                if len(x) > 0:
                    # sorted_array = sorted(x, key=lambda x: x[3])
                    if rel not in outDict[k]:
                        outDict[k][rel] = {'ts':[], 'te':[]}
                    # outDict[k][rel]['ts'].append(e[3] - sorted_array[-1][3])
                    # outDict[k][rel]['te'].append(e[4] - sorted_array[-1][3])

                    outDict[k][rel]['ts'].append(x[:, 3] - e[3])
                    outDict[k][rel]['te'].append(x[:, 3] - e[4])


    for k in outDict:
        print(k)
        for k1 in outDict[k]:
            print(k1)
            print(outDict[k][k1])
            if len(outDict[k][k1]['ts'])>10:
                num_bins = max(5, min(len(outDict[k][k1]['ts']) - 5, 10))
                plot_freq_hist(remove_outliers(outDict[k][k1]['ts']), num_bins, ' '.join([str(k),str(k1), 'ts']))
                plot_freq_hist(remove_outliers(outDict[k][k1]['te']), num_bins, ' '.join([str(k),str(k1), 'ts']))

        print('-----------------------')



def plot_freq_hist(data, num_bins, fig_name):
    plt.figure()
    # Plot the histogram
    plt.hist(data, bins=num_bins)
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.title('Frequency Distribution')
    # plt.show()
    if '.png' not in fig_name:
        fig_name = fig_name + '.png'
    plt.savefig(fig_name)
    plt.close()



def remove_outliers(data):
    threshold = 5
    mean = np.mean(data)
    std = np.std(data)
    z_scores = [(x - mean) / std for x in data]
    return [x for x, z in zip(data, z_scores) if abs(z) <= threshold]




def list_subtract(ls1, ls2):
    # Repeat the last element of the shorter list until their lengths are equal
    len1, len2 = len(ls1), len(ls2)
    if len1 < len2:
        ls1 += [ls1[-1]] * (len2 - len1)
    elif len2 < len1:
        ls2 += [ls2[-1]] * (len1 - len2)

    # Subtract corresponding elements of ls2 from ls1
    result = [ls1[i] - ls2[i] for i in range(len(ls1))]
    return result




def read_random_walk_results(dataset, rel_ls, file_paths, file_suffix, data1=None, flag_interval=True, flag_plot=False, mode=None, flag_time_shifting=False):
    output_dir = '../output/' + dataset + '/'
    if flag_time_shifting:
        output_dir = '../output/' + dataset + '_time_shifting/'
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)
            os.mkdir(output_dir + 'samples')

    for rel in rel_ls:
        # print('rel: ', rel)
        samples = {}
        samples_edges = {}
        stat_res = {}

        # stat_res[rel] = {}

        if data1 is None:
            data = {}
            for file_path in file_paths:
                if '_rel_' + str(rel) in file_path:
                    with open(file_path) as f:
                        data.update(json.load(f))
        else:
            data = data1

        # if str(rel) not in data:
        #     continue

        # print(data)

        if len(data) == 0:
            continue

        for rule_length in [1,2,3,4,5]:
            if str(rule_length) not in data:
                continue
            # res_dict = {}
            for TR_ls in data[str(rule_length)]:
                # print(TR_ls)
                # res_dict[TR_ls] = {}
                for x in data[str(rule_length)][TR_ls]:
                    # if tuple(x["relations"]) not in res_dict[TR_ls]:
                    #     res_dict[TR_ls][tuple(x["relations"])] = {'ts':[], 'te':[]}

                    # res_dict[TR_ls][tuple(x["relations"])]['ts'].append(list_subtract(x['ts'][0:1], x['ts'][1:]))
                    # res_dict[TR_ls][tuple(x["relations"])]['te'].append(list_subtract(x['te'][0:1], x['ts'][1:]))

                    if flag_interval:
                        e = tuple([x['entities'][0], x["relations"][0], x['entities'][1], x['ts'][0], x['te'][0]])
                    else:
                        e = tuple([x['entities'][0], x["relations"][0], x['entities'][1], x['ts'][0]])

                    rule_pattern = ' '.join([str(cur_rel) for cur_rel in x["relations"][1:]]) + ' ' + TR_ls

                    if flag_interval:
                        cur_time_gap = list_subtract(x['ts'][0:1], x['ts'][-1:]) + list_subtract(x['te'][0:1], x['ts'][-1:])
                    else:
                        if mode == 'Train':
                            cur_time_gap = list_subtract(x['ts'][0:1], x['ts'][1:2]) + list_subtract(x['ts'][0:1], x['ts'][-1:])

                        cur_ref_time = x['ts'][1:2] + x['ts'][-1:]
                        cur_ref_edge = [[x['entities'][1], x["relations"][1], x['entities'][2], x['ts'][1]], 
                                        [x['entities'][-2], x["relations"][-1], x['entities'][-1], x['ts'][-1]]]

                    if rule_pattern not in stat_res:
                        stat_res[rule_pattern] = []

                    if mode == 'Train':
                        stat_res[rule_pattern].append(cur_time_gap)

                    if e not in samples:
                        samples[e] = {}
                        samples_edges[e] = {}
                    if rule_pattern not in samples[e]:
                        samples[e][rule_pattern] = []
                        samples_edges[e][rule_pattern] = []

                    samples[e][rule_pattern].append(cur_ref_time)
                    samples_edges[e][rule_pattern].append(cur_ref_edge)

                # for k in res_dict[TR_ls]:
                #     res_dict[TR_ls][k]['ts'] = stat_para_estimation(np.vstack(res_dict[TR_ls][k]['ts']))
                #     res_dict[TR_ls][k]['te'] = stat_para_estimation(np.vstack(res_dict[TR_ls][k]['te']))
        
        
        if mode == 'Train':
            for rule_pattern in stat_res:
                res = np.vstack(stat_res[rule_pattern]).astype(float)
                # stat_res[rel][rule_pattern] = stat_para_estimation_v2(res, dist_type='Gaussian')

                stat_res[rule_pattern] = {'ts_first_event':{}, 'ts_last_event':{}}

                mean_ls, std_ls, prop_ls = adaptive_Gaussian_dist_estimate_new_ver(res[:, 0])
                stat_res[rule_pattern]['ts_first_event'] = {'mean_ls': mean_ls, 'std_ls': std_ls, 'prop_ls': prop_ls}
                mean_ls, std_ls, prop_ls = adaptive_Gaussian_dist_estimate_new_ver(res[:, 1])
                stat_res[rule_pattern]['ts_last_event'] = {'mean_ls': mean_ls, 'std_ls': std_ls, 'prop_ls': prop_ls}

                if flag_plot:
                    for rule_length in [1,2,3,4,5]:
                        plot_freq_hist(res[:, 0], 10, 'fig/len' + str(rule_length) + '/' + '_'.join([str(rel), rule_pattern, 'ts']))
                        plot_freq_hist(res[:, 1], 10, 'fig/len' + str(rule_length) + '/' + '_'.join([str(rel), rule_pattern, 'te']))

                # print(res)
                # print(rule_pattern)
                # print(stat_res[rel][rule_pattern])
                # sys.exit()

                #     print(res_dict[TR_ls])
                #     print('---------------------------')

            # print('\n\n\n\n')


            # for e in samples:
            #     print(e, samples[e])
            #     print('---------------------------')

            # for rel in stat_res:
            #     print(rel, stat_res[rel])
            #     print('---------------------------')


        if mode == 'Train':
            with open(output_dir +  dataset + file_suffix + 'stat_res_rel_' + str(rel) + '_' + mode +'.json', 'w') as f:
                json.dump(stat_res, f)

            new_samples = {}
            for e in samples:
                new_samples[str(e)] = samples[e]

                with open(output_dir + 'samples/' + dataset + file_suffix +  mode + '_sample_' + str(e) + ".json", "w") as f:
                    json.dump(convert_dict({e: samples_edges[e]}), f)

            rule_nums = rule_num_stat(new_samples)

            pattern_ls = {}
            if rel not in rule_nums:
                pattern_ls = []
            else:
                pattern_ls = sorted(rule_nums[rel], key=lambda p: rule_nums[rel][p], reverse=True)[:1000]
                random.shuffle(pattern_ls)

            with open(output_dir + dataset + file_suffix + "pattern_ls_rel_" + str(rel) + ".json", 'w') as file:
                json.dump(pattern_ls, file)


    return



def rule_num_stat(data):
    res = {}
    for e in data:
        cur_rel = int(e[1:-1].split(',')[1])
        if cur_rel not in res:
            res[cur_rel] = {}
        for rule in data[e]:
            if rule not in res[cur_rel]:
                res[cur_rel][rule] = 0
            res[cur_rel][rule] += 1
    return res



def stat_para_estimation(data):
    mu_ls = [-1] * data.shape[1]
    std_ls = [-1] * data.shape[1]
    loc_ls = [-1] * data.shape[1]
    lamda_ls = [-1] * data.shape[1]

    if len(data)>10:
        for i in range(data.shape[1]):
            range_of_data = max(data[:, i]) - min(data[:, i])
            if range_of_data>0:
                stat, p = shapiro(data[:, i])
                if p > 0.05:
                    mu, std = norm.fit(data[:, i])
                    mu_ls[i] = mu
                    std_ls[i] = std
                else:
                    loc, scale = expon.fit(data[:, i])
                    lamda = 1 / scale
                    loc_ls[i] = loc
                    lamda_ls[i] = lamda
            else:
                    mu, std = norm.fit(data[:, i])
                    std = max(0.1, std)
                    mu_ls[i] = mu
                    std_ls[i] = std
    elif len(data)>1:
        for i in range(data.shape[1]):
            mu, std = norm.fit(data[:, i])
            std = max(0.1, std)
            mu_ls[i] = mu
            std_ls[i] = std
    else:
        for i in range(data.shape[1]):
            mu = data[0, i]
            mu_ls[i] = mu



    return mu_ls, std_ls, loc_ls, lamda_ls




def merge_ls(ls1):
    res = []
    for l in ls1:
        res += l
    return [[x] for x in res]





def exponential(x, lamb, offset):
    return lamb * np.exp(-lamb * (x - offset))

def negative_exponential(x, lamb, offset):
    return lamb * np.exp(lamb * (x - offset))

def fit_data(data):
    # fit the data to the Gaussian distribution
    try:
        mu, std = norm.fit(data)
        popt_gauss = [mu, std]
        sse_gauss = np.sum((norm.pdf(data, mu, std) - np.zeros_like(data)) ** 2)
    except RuntimeError:
        popt_gauss = [np.nan, np.nan]
        sse_gauss = np.nan

    # calculate the mean and standard deviation of the data
    mean = np.mean(data)
    std = np.std(data)

    # define the initial guess for the parameters
    offset = np.min(data)
    lamb = 1 / (mean - offset)
    p0 = [lamb, offset]

    offset = np.max(data)
    lamb = 1 / (offset - mean)
    p1 = [lamb, offset]

    # fit the data to the exponential and negative exponential distributions
    try:
        popt_exp, _ = curve_fit(exponential, data, np.zeros_like(data), p0=p0)
    except RuntimeError:
        popt_exp = [np.nan, np.nan]
        
    try:
        popt_neg_exp, _ = curve_fit(negative_exponential, data, np.zeros_like(data), p0=p1)
    except RuntimeError:
        popt_neg_exp = [np.nan, np.nan]
    
    # calculate the sum of squared errors for the fits
    sse_exp = np.sum((exponential(data, *popt_exp) - np.zeros_like(data)) ** 2)
    sse_neg_exp = np.sum((negative_exponential(data, *popt_neg_exp) - np.zeros_like(data)) ** 2)
    
    
    # determine which distribution fits the data better based on SSE
    sse_list = [sse_exp, sse_neg_exp, sse_gauss]
    if np.nanmin(sse_list) == sse_gauss:
        return 'Gaussian', popt_gauss
    elif np.nanmin(sse_list) == sse_exp:
        return 'exponential', popt_exp
    else:
        return 'reflected exponential', popt_neg_exp


def stat_para_estimation_v2(data, dist_type=None):
    mu_ls = [-1] * data.shape[1]
    std_ls = [-1] * data.shape[1]
    lamda_ls = [-1] * data.shape[1]
    offset_ls = [-1] * data.shape[1]
    exp_dist_dir_ls = ['pos'] * data.shape[1]

    if len(data)>1:
        for i in range(data.shape[1]):
            if np.min(data[:, i]) == np.max(data[:, i]):
                mu_ls[i] = np.mean(data[:, i])
                std_ls[i] = 0.1
                continue
            if dist_type is None:
                est_dist_type, pars = fit_data(data[:, i])
                if est_dist_type == 'Gaussian':
                    mu_ls[i] = pars[0]
                    std_ls[i] = pars[1]
                elif est_dist_type == 'exponential':
                    lamda_ls[i] = pars[0]
                    offset_ls[i] = pars[1]
                else:
                    lamda_ls[i] = pars[0]
                    offset_ls[i] = pars[1]
                    exp_dist_dir_ls[i] = ['neg']
            elif dist_type == 'Gaussian':
                mu_ls[i] = np.mean(data[:, i])
                std_ls[i] = np.std(data[:, i])
    else:
        for i in range(data.shape[1]):
            mu = data[0, i]
            mu_ls[i] = mu



    return mu_ls, std_ls, lamda_ls, offset_ls, exp_dist_dir_ls







# def process_random_walk_results_dist_ver(dataset, mode, num_rel, file_paths, file_suffix, num_workers=10):
#     index_pieces = split_list_into_pieces(range(num_rel), num_workers)

#     data = {}
#     for file_path in file_paths:
#         with open(file_path) as f:
#             data.update(json.load(f))

#     results = Parallel(n_jobs=num_workers)(delayed(read_random_walk_results)(dataset, piece, data=data, flag_interval=False, 
#                                                     flag_plot=False, mode=mode) for piece in index_pieces)



def fun2():
    data = {}

    for i in range(5):
        with open('stat_res_len_'+ str(i+1) +'.json', 'r') as f:
            # Load the JSON data from the file
            data1 = json.load(f)
        for k in data1:
            if k not in data:
                data[k] = data1[k]
            else:
                data[k].update(data1[k])



    # Create a Pandas Excel writer using xlsxwriter as the engine
    writer = pd.ExcelWriter('output.xlsx', engine='xlsxwriter')

    # Loop through the dictionary and convert each inner dictionary to a Pandas DataFrame
    # and write it to a separate sheet in the Excel file
    for sheet_name, sheet_data in data.items():
        for pattern in sheet_data:
            sheet_data[pattern] = merge_ls(sheet_data[pattern])

        if len(sheet_data)>0:
            df = pd.DataFrame(sheet_data).T.reset_index()
            df.columns = ['rule_pattern', 'ts_mu', 'te_mu', 'ts_std', 'te_std', 'ts_lambda', 
                                'te_lambda', 'ts_offset', 'te_offset', 'ts_exp_dir', 'te_exp_dir']
            df.to_excel(writer, sheet_name=sheet_name, index=False)

    # Save the Excel file
    writer.save()




def fun3():
    # Read each sheet into a dictionary of DataFrames
    sheets_dict = pd.read_excel('output.xlsx', sheet_name=None)

    # Create a new Excel file for writing
    writer = pd.ExcelWriter('output_sorted.xlsx', engine='xlsxwriter')

    # Loop through each sheet and sort based on the last number before "ukn"
    for sheet_name, sheet_df in sheets_dict.items():
        # Extract the last number before "ukn" as a new column
        sheet_df['last_rel'] = sheet_df['rule_pattern'].str.extract('(\d+)(?!.*\d)')

        # Convert the "last_num" column to numeric type
        sheet_df['last_rel'] = pd.to_numeric(sheet_df['last_rel'])

        # Sort the DataFrame by the "last_num" column
        sheet_df_sorted = sheet_df.sort_values('last_rel')

        # Write the sorted DataFrame to a new sheet in the Excel file
        sheet_df_sorted.to_excel(writer, sheet_name=sheet_name, index=False)

    # Save the Excel file and close the writer
    writer.save()



def fun4(dataset):
    train_samples = {}
    for i in range(5):
        file_path = '../output/'+ dataset +'_train_walks_suc_with_TR_len_'+ str(i+1) +'.json'
        train_samples1, stat_res = read_random_walk_results(range(num_rel*2), file_path, flag_plot=False)

        for e in train_samples1:
            if e in train_samples:
                train_samples[e].update(train_samples1[e])
            else:
                train_samples[e] = train_samples1[e]

    new_train_samples = {}
    for e in train_samples:
        new_train_samples[str(e)] = train_samples[e]

    with open(dataset + "_train_samples.json", "w") as f:
        json.dump(new_train_samples, f)

    return train_samples




def gaussian_pdf(x, mu, std):
    """
    Calculate the probability density function (PDF) of a Gaussian distribution
    with mean mu and standard deviation std at the point x.
    """
    return (1 / (std * np.sqrt(2 * np.pi))) * np.exp(-(x - mu)**2 / (2 * std**2))


def calculate_TR(interval1, interval2):
    interval1 = [min(interval1), max(interval1)]
    interval2 = [min(interval2), max(interval2)]
    if 9999 in interval1 or 9999 in interval2:
        return 'ukn'
    if interval1[0] < interval2[0] and interval1[1] <= interval2[0]:
        return 'bf'
    if interval1[0] >= interval2[1] and interval1[1] > interval2[1]:
        return 'af'
    return 'touch'


def calculate_TR_mat_ver(interval1, interval2):
    interval1 = np.array(interval1)
    interval2 = np.array(interval2)
    
    # Check for presence of 9999 in any row of either matrix
    invalid_rows = np.any((interval1 == 9999) | (interval2 == 9999), axis=1)
    
    # Set invalid rows to 0
    result = np.ones(interval1.shape[0], dtype=int) * 2
    
    # Check conditions for each row
    result[np.logical_and(interval1[:, 0] < interval2[:, 0], interval1[:, 1] <= interval2[:, 0])] = 1
    result[np.logical_and(interval1[:, 0] >= interval2[:, 1], interval1[:, 1] > interval2[:, 1])] = 3
    result[invalid_rows] = 0
    
    return result


def my_timegap(timestamp1, timestamp2):
    if timestamp1 == 9999 or timestamp2 == 9999:
        return 9999
    else:
        return timestamp1 - timestamp2


def read_TEILP_results(res_dict, with_ref_end_time=False, flag_capture_dur_only=False, rel=None, known_edges=None):
    target_query = res_dict['query'][1]
    targ_interval = res_dict['query'][3:]
    output = {}

    if rel is not None:
        if int(target_query) !=rel:
            return output

    output[target_query] = {}
    for rlen in res_dict.keys():
        if rlen == 'query':
            continue

        for walk in res_dict[rlen]:
            walk_cp = copy.copy(walk)
            if known_edges is not None:
                # flag_show = 0
                for i in range(int(rlen)):
                    if ([walk[4*i:4*i+5][num] for num in [0,1,4,2,3]] not in known_edges.tolist()) and (walk[4*i+2:4*i+4] != [9999, 9999]):
                        walk[4*i+2:4*i+4] = [9999, 9999]
                        # flag_show = 1
                        # print(i, [walk[4*i:4*i+5][num] for num in [0,1,4,2,3]], [walk[4*i:4*i+5][num] for num in [0,1,4,2,3]] in known_edges)

                # if flag_show:
                #     print(walk_cp)
                #     print(walk)
                #     print(' ')

            rel_ls = [str(walk[4*i+1]) for i in range(int(rlen))]
            interval_ls = [[9999, 9999]] + [walk[4*i+2:4*i+4] for i in range(int(rlen))]
            TR_ls = [calculate_TR(interval_ls[i], interval_ls[i+1]) for i in range(int(rlen))]
            edge_ls = [walk[4*i:4*i+5] for i in range(int(rlen))]

            # print(rel_ls)
            # print(interval_ls)
            # print(TR_ls)
            # print(edge_ls)

            if not flag_capture_dur_only:
                if not with_ref_end_time:
                    time_gap_ts = [my_timegap(targ_interval[0], interval_ls[1][0]), my_timegap(targ_interval[0], interval_ls[-1][0])]
                    time_gap_te = [my_timegap(targ_interval[1], interval_ls[1][0]), my_timegap(targ_interval[1], interval_ls[-1][0])]
                    time_ref = [interval_ls[1][0], interval_ls[-1][0]]
                    # print(time_gap_ts)
                    # print(time_gap_te)
                else:
                    time_gap_ts_ref_ts = [my_timegap(targ_interval[0], interval_ls[1][0]), my_timegap(targ_interval[0], interval_ls[-1][0])]
                    time_gap_ts_ref_te = [my_timegap(targ_interval[0], interval_ls[1][1]), my_timegap(targ_interval[0], interval_ls[-1][1])]
                    time_gap_te_ref_ts = [my_timegap(targ_interval[1], interval_ls[1][0]), my_timegap(targ_interval[1], interval_ls[-1][0])]
                    time_gap_te_ref_te = [my_timegap(targ_interval[1], interval_ls[1][1]), my_timegap(targ_interval[1], interval_ls[-1][1])]
                    time_ref = [interval_ls[1], interval_ls[-1]]
                    edge_ref = [edge_ls[0], edge_ls[-1]]

            dur_with_ref = [my_timegap(targ_interval[1], targ_interval[0]), interval_ls[1], interval_ls[-1]]


            cur_rulePattern = ' '.join(rel_ls + TR_ls)
            if cur_rulePattern not in output[target_query]:
                output[target_query][cur_rulePattern] = {'dur_with_ref':[]}

                if not flag_capture_dur_only:
                    if not with_ref_end_time:
                        output[target_query][cur_rulePattern].update({'time_gap_ts':[], 'time_gap_te':[], 'time_ref':[]})
                    else:
                        output[target_query][cur_rulePattern].update({'time_gap_ts_ref_ts':[], 'time_gap_ts_ref_te':[], 
                                                                               'time_gap_te_ref_ts':[], 'time_gap_te_ref_te':[], 
                                                                               'time_ref':[], 'edge_ref':[], 'edge_ls':[]})

            output[target_query][cur_rulePattern]['dur_with_ref'].append(dur_with_ref)

            if not flag_capture_dur_only:
                if not with_ref_end_time:
                    output[target_query][cur_rulePattern]['time_gap_ts'].append(time_gap_ts)
                    output[target_query][cur_rulePattern]['time_gap_te'].append(time_gap_te)
                    output[target_query][cur_rulePattern]['time_ref'].append(time_ref)
                else:
                    output[target_query][cur_rulePattern]['time_gap_ts_ref_ts'].append(time_gap_ts_ref_ts)
                    output[target_query][cur_rulePattern]['time_gap_ts_ref_te'].append(time_gap_ts_ref_te)
                    output[target_query][cur_rulePattern]['time_gap_te_ref_ts'].append(time_gap_te_ref_ts)
                    output[target_query][cur_rulePattern]['time_gap_te_ref_te'].append(time_gap_te_ref_te)
                    output[target_query][cur_rulePattern]['time_ref'].append(time_ref)
                    output[target_query][cur_rulePattern]['edge_ref'].append(edge_ref)
                    output[target_query][cur_rulePattern]['edge_ls'].append(edge_ls)

    # print(output)
    return output






def convert_walks_into_rules(path, dataset, idx_ls=None, with_ref_end_time=True, flag_time_shift=0, 
                             flag_capture_dur_only=False, rel=None, known_edges=None, flag_few_training=0,
                             ratio=None, imbalanced_rel=None, flag_biased=0, exp_idx=None):
    if idx_ls is not None:
        file_ls = []
        for data_idx in idx_ls:
            cur_path = dataset + '_idx_' + str(data_idx)
            if ratio is not None:
                cur_path += '_ratio_' + str(ratio)
            if exp_idx is not None:
                cur_path += '_exp_' + str(exp_idx)
            cur_path += '.json'
            file_ls.append(cur_path)
    else:
        file_ls = os.listdir(path)
        file_ls = [f for f in file_ls if dataset in f]
        if ratio is not None:
            file_ls = [f for f in file_ls if '_ratio_' + str(ratio) in f]
        if imbalanced_rel is not None:
            file_ls = [f for f in file_ls if '_rel_' + str(imbalanced_rel) in f]
        if exp_idx is not None:
            file_ls = [f for f in file_ls if '_exp_' + str(exp_idx) in f]


    if flag_biased:
        if imbalanced_rel is None:
            file_ls = [f for f in file_ls if '_rel_' not in f]
        else:
            file_ls = [f for f in file_ls if '_rel_' + str(imbalanced_rel) + '.json' in f]
            rel = imbalanced_rel


    output = {}
    for file in file_ls:
        # print(file)
        with open(path + '/' + file, 'r') as f:
            data = f.read()
            json_data = json.loads(data)

        # print(json_data)

        # cur_output = create_stat_dict_for_empty_walk_res(json_data, with_ref_end_time=with_ref_end_time)
        cur_output = read_TEILP_results(json_data, with_ref_end_time=with_ref_end_time, flag_capture_dur_only=flag_capture_dur_only, rel=rel, known_edges=known_edges)

        for k in cur_output:
            if k not in output:
                output[k] = cur_output[k]
            else:
                for k1 in cur_output[k]:
                    if k1 not in output[k]:
                        output[k][k1] = cur_output[k][k1]
                    else:
                        if not flag_capture_dur_only:
                            if not with_ref_end_time:
                                output[k][k1]['time_gap_ts'] += cur_output[k][k1]['time_gap_ts']
                                output[k][k1]['time_gap_te'] += cur_output[k][k1]['time_gap_te']
                            else:
                                output[k][k1]['time_gap_ts_ref_ts'] += cur_output[k][k1]['time_gap_ts_ref_ts']
                                output[k][k1]['time_gap_ts_ref_te'] += cur_output[k][k1]['time_gap_ts_ref_te']
                                output[k][k1]['time_gap_te_ref_ts'] += cur_output[k][k1]['time_gap_te_ref_ts']
                                output[k][k1]['time_gap_te_ref_te'] += cur_output[k][k1]['time_gap_te_ref_te']

                        output[k][k1]['dur_with_ref'] += cur_output[k][k1]['dur_with_ref']

    # for rel in output:
    #     for rule in output[rel]:
    #         print(rel, rule, output[rel][rule]['dur_with_ref'])

    # sys.exit()


    if flag_capture_dur_only:
        return output


    output_stat = {}
    for rel in output:
        for rule in output[rel]:
            if rel not in output_stat:
                output_stat[rel] = {}
            if rule not in output_stat[rel]:
                output_stat[rel][rule] = {}

            if not with_ref_end_time:
                ts_first_event = [x[0] for x in output[rel][rule]['time_gap_ts'] if x[0]!=9999]
                ts_last_event = [x[1] for x in output[rel][rule]['time_gap_ts'] if x[1]!=9999]
                te_first_event = [x[0] for x in output[rel][rule]['time_gap_te'] if x[0]!=9999]
                te_last_event = [x[1] for x in output[rel][rule]['time_gap_te'] if x[1]!=9999]

                if len(ts_first_event)>0:
                    output_stat[rel][rule].update({'ts_first_event': [np.mean(ts_first_event), np.std(ts_first_event)]})
                if len(ts_last_event)>0:
                    output_stat[rel][rule].update({'ts_last_event': [np.mean(ts_last_event), np.std(ts_last_event)]})
                if len(te_first_event)>0:
                    output_stat[rel][rule].update({'te_first_event': [np.mean(te_first_event), np.std(te_first_event)]})
                if len(te_last_event)>0:
                    output_stat[rel][rule].update({'te_last_event': [np.mean(te_last_event), np.std(te_last_event)]})
            else:
                event_type = ['ts_first_event_ts', 'ts_last_event_ts', 'ts_first_event_te', 'ts_last_event_te',
                                 'te_first_event_ts', 'te_last_event_ts', 'te_first_event_te', 'te_last_event_te']
                time_gap_type = ['time_gap_ts_ref_ts', 'time_gap_ts_ref_ts', 'time_gap_ts_ref_te', 'time_gap_ts_ref_te',
                                     'time_gap_te_ref_ts', 'time_gap_te_ref_ts', 'time_gap_te_ref_te', 'time_gap_te_ref_te']
                idx_event = [0, 1, 0, 1, 0, 1, 0, 1]

                time_gap = {}
                for i in range(len(time_gap_type)):
                    time_gap[event_type[i]] = [x[idx_event[i]] for x in output[rel][rule][time_gap_type[i]] if x[idx_event[i]]!=9999]
                    if len(time_gap[event_type[i]])>0:
                        output_stat[rel][rule].update({event_type[i]: [np.mean(time_gap[event_type[i]]), np.std(time_gap[event_type[i]])]})



            # print(rel, [rule], output[rel][rule])
            # print('------------------------------------')


    # print(output_stat)
    return output_stat



def processing_stat_res(dataset, num_rel, flag_with_ref_end_time=False, flag_time_shifting=0):
    if not flag_time_shifting:
        with open('../output/' + dataset + '/' + dataset + "_stat_res.json", "r") as f:
            stat_res = json.load(f)
    else:
        with open('../output/' + dataset + '/' + dataset + "_time_shifting_stat_res.json", "r") as f:
            stat_res = json.load(f)

    pattern_ls = {}
    ts_stat_ls = {}
    te_stat_ls = {}

    for i in range(num_rel):
        pattern_ls[i] = []
        ts_stat_ls[i] = []
        te_stat_ls[i] = []

        for rule in stat_res[str(i)]:
            if len(stat_res[str(i)][rule].keys()) == 0:
                continue

            pattern_ls[i].append(rule)

            if not flag_with_ref_end_time:
                k_ls = ['ts_first_event', 'ts_last_event']
            else:
                k_ls = ['ts_first_event_ts', 'ts_first_event_te', 'ts_last_event_ts', 'ts_last_event_te']
            
            cur_ts_stat_ls = []
            for k in k_ls:
                if k in stat_res[str(i)][rule]:
                    cur_ts_stat_ls.append(stat_res[str(i)][rule][k][0]) # mean
                    cur_ts_stat_ls.append(max(5, stat_res[str(i)][rule][k][1])) # variance
                else:
                    cur_ts_stat_ls.append(9999)
                    cur_ts_stat_ls.append(9999)

            ts_stat_ls[i].append(cur_ts_stat_ls)



            if not flag_with_ref_end_time:
                k_ls = ['te_first_event', 'te_last_event']
            else:
                k_ls = ['te_first_event_ts', 'te_first_event_te', 'te_last_event_ts', 'te_last_event_te']

            cur_te_stat_ls = []
            for k in k_ls:
                if k in stat_res[str(i)][rule]:
                    cur_te_stat_ls.append(stat_res[str(i)][rule][k][0])
                    cur_te_stat_ls.append(max(5, stat_res[str(i)][rule][k][1]))
                else:
                    cur_te_stat_ls.append(9999)
                    cur_te_stat_ls.append(9999)

            te_stat_ls[i].append(cur_te_stat_ls)


        #     print(pattern_ls[i][:5])
        #     print(ts_stat_ls[i][:5])
        #     print(te_stat_ls[i][:5])
        #     print('-----------------------')
        #     break


    return pattern_ls, ts_stat_ls, te_stat_ls



def fun5():
    pattern_ls = {}
    ts_mu_ls = {}
    te_mu_ls = {}
    ts_std_ls = {}
    te_std_ls = {}

    # read the excel file into a pandas dataframe
    for i in range(num_rel*2):
        df = pd.read_excel('output_sorted.xlsx', sheet_name=str(i))

        # extract the first five rows of the second column
        pattern_ls[i] = df.iloc[:, 0].values.tolist()
        ts_mu_ls[i] = df.iloc[:, 1].values.tolist()
        te_mu_ls[i] = df.iloc[:, 2].values.tolist()
        ts_std_ls[i] = df.iloc[:, 3].values.tolist()
        te_std_ls[i] = df.iloc[:, 4].values.tolist()

        ts_mu_ls[i] = [float(x[1:-1]) for x in ts_mu_ls[i]]
        ts_std_ls[i] = [float(x[1:-1]) if float(x[1:-1])>5 else 5 for x in ts_std_ls[i]]

        te_mu_ls[i] = [float(x[1:-1]) for x in te_mu_ls[i]]
        te_std_ls[i] = [float(x[1:-1]) if float(x[1:-1])>5 else 5 for x in te_std_ls[i]]


    with open("train_samples.json", "r") as f:
        train_samples = json.load(f)

    with open("test_samples.json", "r") as f:
        test_samples = json.load(f)



def create_inputs():
    inputs = []
    for e in train_samples:
        if int(e.split(',')[1]) != cur_rel:
            continue
        cur_input = np.zeros((1, len(pattern_ls)))
        cur_ts = int(e.split(',')[3])

        for p in train_samples[e]:
            p_idx = pattern_ls.index(p)
            prob_ts = []
            for walk in train_samples[e][p]:
                delta_t = timestamp_range - (cur_ts - walk[0])
                prob_ts_ls = gaussian_pdf(delta_t, ts_mu_ls[p_idx], ts_std_ls[p_idx])
                prob_ts_ls /= np.sum(prob_ts_ls)

                prob_ts.append(prob_ts_ls[delta_t.tolist().index(walk[0])])

            prob_ts = np.mean(prob_ts)

            # print(e, p_idx, prob_ts)
            cur_input[0, p_idx] = prob_ts

        inputs.append(cur_input)

    inputs = np.vstack((inputs))

    # print(input)


def obtain_one_hot_encoding_from_rule_pattern(p, num_rel, mode='without_ukn'):
    p_ls = p.split(' ')
    rule_length = len(p_ls)//2
    rlens = np.zeros((5, 1))
    rlens[rule_length - 1, 0] = 1

    # if rule_length <5:
    #     print(p_ls)

    rels = np.zeros((num_rel * 2, 15))
    rel_ls = list(range(num_rel * 2))
    for i in range(rule_length):
        rels[rel_ls.index(int(p_ls[i])), rule_length*(rule_length-1)//2 + i] = 1
        # if rule_length <5:
        #     print(rel_ls.index(int(p_ls[i])), i)

    if mode == 'without_ukn':
        TRs = np.zeros((3, 15))
        TR_ls = ['bf', 'touch', 'af']
    else:
        TRs = np.zeros((4, 15))
        TR_ls = ['ukn', 'bf', 'touch', 'af']
    for i in range(1, rule_length):
        TRs[TR_ls.index(p_ls[rule_length + i]), rule_length*(rule_length-1)//2 + i] = 1
        # if rule_length <5:
        #     print(TR_ls.index(p_ls[rule_length + i]), i)


    TRs = TRs[:, [2,4,5,7,8,9,11,12,13,14]]

    # if rule_length <5:
    #     print(rels)
    #     print(TRs)

    return rlens, rels, TRs


def my_concat(array_list):
    # determine the maximum shape along each dimension
    max_shape = np.max([a.shape for a in array_list], axis=0)
    # pad each array with zeros to match the maximum shape
    padded_arrays = [np.pad(a, [(0, max_shape[0]-a.shape[0]), (0, max_shape[1]-a.shape[1])], mode='constant') for a in array_list]
    # concatenate the padded arrays into a 3D np.array at axis 0
    concatenated_array = np.stack(padded_arrays, axis=0)

    return concatenated_array


def create_inputs_v2(cur_rel):
    input_rlens = []
    input_rels = []
    input_TRs = []
    input_probs_ts = []
    input_probs_te = []
    for e in train_samples:
        if int(e.split(',')[1]) != cur_rel:
            continue
        # cur_input = np.zeros((1, len(pattern_ls)))
        cur_ts = int(e.split(',')[3])

        rlens = []
        rels = []
        TRs = []
        probs_ts = []
        probs_te = []
        for p in train_samples[e]:
            this_rlens, this_rels, this_TRs = obtain_one_hot_encoding_from_rule_pattern(p)
            rlens.append(this_rlens)
            rels.append(this_rels)
            TRs.append(this_TRs)


            # if len(p.split(' '))//2 <5:
            #     print(p)
            #     print(rule_length, rels, TRs)

            # print(p)
            # print(rlens, rels, TRs)

            p_idx = pattern_ls[cur_rel].index(p)
            prob_ts = []
            prob_te = []
            for walk in train_samples[e][p]:
                delta_t = timestamp_range - (cur_ts - walk[0])
                prob_ts_ls = gaussian_pdf(delta_t, ts_mu_ls[cur_rel][p_idx], ts_std_ls[cur_rel][p_idx])
                prob_ts_ls /= np.sum(prob_ts_ls)

                prob_te_ls = gaussian_pdf(delta_t, te_mu_ls[cur_rel][p_idx], te_std_ls[cur_rel][p_idx])
                prob_te_ls /= np.sum(prob_te_ls)

                prob_ts.append(prob_ts_ls[delta_t.tolist().index(walk[0])])
                prob_te.append(prob_te_ls[delta_t.tolist().index(walk[0])])


            prob_ts = np.mean(prob_ts)
            prob_te = np.mean(prob_te)

            # print(e, p_idx, prob_ts, prob_te)

            # cur_input[0, p_idx] = prob_ts
            probs_ts.append(prob_ts)
            probs_te.append(prob_te)


        rlens = np.hstack(rlens)
        rels = np.hstack(rels)
        TRs = np.hstack(TRs)
        probs_ts = np.array(probs_ts).reshape((1,-1))
        probs_te = np.array(probs_te).reshape((1,-1))

        # for x in [e, rlens, rels, TRs]:
        #     print(x)

        # print('--------------------------')

        input_rlens.append(rlens)
        input_rels.append(rels)
        input_TRs.append(TRs)
        input_probs_ts.append(probs_ts)
        input_probs_te.append(probs_te)


    input_rlens = my_concat(input_rlens)
    input_rels = my_concat(input_rels)
    input_TRs = my_concat(input_TRs)
    input_probs_ts = my_concat(input_probs_ts)
    input_probs_te = my_concat(input_probs_te)

    # print(input_rlens.shape)
    # print(input_rels.shape)
    # print(input_TRs.shape)
    # print(input_probs_ts.shape)
    # print(input_probs_te.shape)

    # print(input_probs)

    return input_rlens, input_rels, input_TRs, input_probs_ts, input_probs_te





def find_pos_of_1(matrix):
    rows, cols = np.where(matrix == 1)
    # create a list of tuples from the row and column indices
    positions = list(zip(rows, cols))
    # print the list of positions
    return positions


def find_non_elements(matrix):
    indices = np.nonzero(matrix)
    values = matrix[indices]

    # print the results
    print("Indices of non-zero elements:")
    print(indices)
    print("Values of non-zero elements:")
    print(values)



def create_inputs_v3(path, dataset, edges, idx_ls, targ_rel, num_samples_dist, pattern_ls, timestamp_range, num_rel, 
                     ts_stat_ls, te_stat_ls, mode=None, rm_ls=None, with_ref_end_time=False, 
                     only_find_samples_with_empty_rules=False, flag_output_probs_with_ref_edges=False,
                     flag_acceleration=False, flag_time_shift=False):

    input_intervals = {}
    cnt = 0

    if flag_output_probs_with_ref_edges:
        output_probs_with_ref_edges = {}

    sample_with_empty_rules_ls = []

    for data_idx in idx_ls:
        file = dataset + '_idx_' + str(data_idx) + '.json'
        if not os.path.exists(path + file):
            sample_with_empty_rules_ls.append(data_idx)
            continue

        with open(path + file, 'r') as f:
            data = f.read()
            json_data = json.loads(data)

        if mode == 'Test' and rm_ls !=None:
            if (data_idx - num_samples_dist[0] - num_samples_dist[1]) in rm_ls:
                # print(json_data['query'])
                continue

        if targ_rel != None:
            if json_data['query'][1] != targ_rel:
                continue

        cur_rel = json_data['query'][1]
        cur_output = read_TEILP_results(json_data, with_ref_end_time)

        if mode == 'Train':
            cur_ts = int(json_data['query'][3])
            cur_te = int(json_data['query'][4])
            cur_interval = [cur_ts, cur_te]
        else:
            # data_idx -= num_samples_dist[1]
            if data_idx >= len(edges):
                cur_interval = edges[data_idx - len(edges), 3:]
            else:
                cur_interval = edges[data_idx, 3:]

        input_intervals[data_idx] = cur_interval

        if len(cur_output[cur_rel]) == 0:
            sample_with_empty_rules_ls.append(data_idx)
            continue

        cur_valid_rules = [p for p in cur_output[cur_rel] if p in pattern_ls[cur_rel]]
        if len(cur_valid_rules) == 0:
            sample_with_empty_rules_ls.append(data_idx)
            continue

        if only_find_samples_with_empty_rules:
            continue


        if mode == 'Train':
            dim = 1
        else:
            dim = len(timestamp_range)


        if flag_output_probs_with_ref_edges:
            output_probs_with_ref_edges[data_idx] = {'last_event':{}, 'first_event':{}}

        for p in cur_valid_rules:
            p_idx = pattern_ls[cur_rel].index(p)

            # print(ts_stat_ls[cur_rel][p_idx])
            # print(te_stat_ls[cur_rel][p_idx])

            if not with_ref_end_time:
                prob_ls = {0:[], 1:[], 2:[], 3:[]}
            else:
                prob_ls = {0:[], 1:[], 2:[], 3:[], 4:[], 5:[], 6:[], 7:[]}

            if not with_ref_end_time:
                cur_stat_ls = {0: {0: {}, 1:{}}, 1: {0: {}, 1:{}}}
                cur_stat_ls[0][0] = [ts_stat_ls[cur_rel][p_idx][0], ts_stat_ls[cur_rel][p_idx][1]]
                cur_stat_ls[0][1] = [ts_stat_ls[cur_rel][p_idx][2], ts_stat_ls[cur_rel][p_idx][3]]
                cur_stat_ls[1][0] = [te_stat_ls[cur_rel][p_idx][0], te_stat_ls[cur_rel][p_idx][1]]
                cur_stat_ls[1][1] = [te_stat_ls[cur_rel][p_idx][2], te_stat_ls[cur_rel][p_idx][3]]
            else:
                cur_stat_ls = {0: {0: {0:{}, 1:{}}, 1:{0:{}, 1:{}}}, 1: {0: {0:{}, 1:{}}, 1:{0:{}, 1:{}}}}
                cur_stat_ls[0][0][0] = [ts_stat_ls[cur_rel][p_idx][0], ts_stat_ls[cur_rel][p_idx][1]]
                cur_stat_ls[0][0][1] = [ts_stat_ls[cur_rel][p_idx][2], ts_stat_ls[cur_rel][p_idx][3]]
                cur_stat_ls[0][1][0] = [ts_stat_ls[cur_rel][p_idx][4], ts_stat_ls[cur_rel][p_idx][5]]
                cur_stat_ls[0][1][1] = [ts_stat_ls[cur_rel][p_idx][6], ts_stat_ls[cur_rel][p_idx][7]]

                cur_stat_ls[1][0][0] = [te_stat_ls[cur_rel][p_idx][0], te_stat_ls[cur_rel][p_idx][1]]
                cur_stat_ls[1][0][1] = [te_stat_ls[cur_rel][p_idx][2], te_stat_ls[cur_rel][p_idx][3]]
                cur_stat_ls[1][1][0] = [te_stat_ls[cur_rel][p_idx][4], te_stat_ls[cur_rel][p_idx][5]]
                cur_stat_ls[1][1][1] = [te_stat_ls[cur_rel][p_idx][6], te_stat_ls[cur_rel][p_idx][7]]


            # print(cur_output[cur_rel][p]['time_ref'])
            # print(cur_output[cur_rel][p]['edge_ref'])

            for (idx_time_ref, time_ref) in enumerate(cur_output[cur_rel][p]['time_ref']):
                # print(time_ref)

                if flag_output_probs_with_ref_edges:
                    refEdge = cur_output[cur_rel][p]['edge_ref'][idx_time_ref]

                    if tuple(refEdge[0]) not in output_probs_with_ref_edges[data_idx]['first_event']:
                        if not with_ref_end_time:
                            output_probs_with_ref_edges[data_idx]['first_event'][tuple(refEdge[0])] = {0:[], 1:[]}
                        else:
                            output_probs_with_ref_edges[data_idx]['first_event'][tuple(refEdge[0])] = {0: {0:[], 1:[]}, 1:{0:[], 1:[]}}

                    if tuple(refEdge[1]) not in output_probs_with_ref_edges[data_idx]['last_event']:
                        if not with_ref_end_time:
                            output_probs_with_ref_edges[data_idx]['last_event'][tuple(refEdge[1])] = {0:[], 1:[]}
                        else:
                            output_probs_with_ref_edges[data_idx]['last_event'][tuple(refEdge[1])] = {0: {0:[], 1:[]}, 1:{0:[], 1:[]}}


                for idx_ts_or_te in [0,1]:
                    for idx_first_or_last_event in [0,1]:
                        if not with_ref_end_time:
                            refTime = time_ref[idx_first_or_last_event]
                            delta_t = timestamp_range - refTime
                            cur_Gaussian_mean = cur_stat_ls[idx_ts_or_te][idx_first_or_last_event][0]
                            cur_Gaussian_std = cur_stat_ls[idx_ts_or_te][idx_first_or_last_event][1]
                            idx_prob_ls = 2*idx_ts_or_te + idx_first_or_last_event

                            if (refTime != 9999) and (cur_Gaussian_mean != 9999) and (cur_Gaussian_std != 9999):
                                cur_probs = gaussian_pdf(delta_t, cur_Gaussian_mean, cur_Gaussian_std)

                                if flag_time_shift:
                                    cur_probs[delta_t < 0] = 0
                                if len(cur_probs[np.isnan(cur_probs)])>0 or len(cur_probs[np.isinf(cur_probs)])>0:
                                    continue
                                if np.sum(cur_probs) == 0:
                                    continue

                                cur_probs /= np.sum(cur_probs)

                                if mode == 'Train':
                                    prob_ls[idx_prob_ls].append(cur_probs[delta_t.tolist().index(cur_interval[idx_ts_or_te] - refTime)])
                                else:
                                    prob_ls[idx_prob_ls].append(cur_probs)

                        else:
                            for idx_ref_time_ts_or_te in [0,1]:
                                refTime = time_ref[idx_first_or_last_event][idx_ref_time_ts_or_te]
                                delta_t = timestamp_range - refTime
                                cur_Gaussian_mean = cur_stat_ls[idx_ts_or_te][idx_first_or_last_event][idx_ref_time_ts_or_te][0]
                                cur_Gaussian_std = cur_stat_ls[idx_ts_or_te][idx_first_or_last_event][idx_ref_time_ts_or_te][1]
                                idx_prob_ls = 4*idx_ts_or_te + 2*idx_first_or_last_event + idx_ref_time_ts_or_te

                                if (refTime != 9999) and (cur_Gaussian_mean != 9999) and (cur_Gaussian_std != 9999):
                                    cur_probs = gaussian_pdf(delta_t, cur_Gaussian_mean, cur_Gaussian_std)

                                    if flag_time_shift:
                                        cur_probs[delta_t < 0] = 0
                                    if len(cur_probs[np.isnan(cur_probs)])>0 or len(cur_probs[np.isinf(cur_probs)])>0:
                                        continue
                                    if np.sum(cur_probs) == 0:
                                        continue

                                    cur_probs /= np.sum(cur_probs)

                                    if mode == 'Train':
                                        # if (cur_interval[idx_ts_or_te] - refTime) not in delta_t.tolist():
                                        #     print(data_idx, cur_interval, refTime, delta_t)
                                        cur_probs_for_gt = cur_probs[delta_t.tolist().index(cur_interval[idx_ts_or_te] - refTime)]
                                        prob_ls[idx_prob_ls].append(cur_probs_for_gt)

                                        if flag_acceleration:
                                            cur_probs_for_gt = [cur_probs_for_gt, p_idx]
                                            # cur_probs_for_gt = [cur_probs_for_gt, pattern_ls[cur_rel].index(p)]

                                        if flag_output_probs_with_ref_edges:
                                            if idx_first_or_last_event == 0:
                                                output_probs_with_ref_edges[data_idx]['first_event'][tuple(refEdge[idx_first_or_last_event])]\
                                                                        [idx_ts_or_te][idx_ref_time_ts_or_te].append(cur_probs_for_gt)
                                            else:
                                                output_probs_with_ref_edges[data_idx]['last_event'][tuple(refEdge[idx_first_or_last_event])]\
                                                                        [idx_ts_or_te][idx_ref_time_ts_or_te].append(cur_probs_for_gt)

                                    else:
                                        prob_ls[idx_prob_ls].append(cur_probs)

                                        if flag_acceleration:
                                            cur_probs = [cur_probs, p_idx]

                                        if flag_output_probs_with_ref_edges:
                                            if idx_first_or_last_event == 0:
                                                output_probs_with_ref_edges[data_idx]['first_event'][tuple(refEdge[idx_first_or_last_event])]\
                                                                        [idx_ts_or_te][idx_ref_time_ts_or_te].append(cur_probs)
                                            else:
                                                output_probs_with_ref_edges[data_idx]['last_event'][tuple(refEdge[idx_first_or_last_event])]\
                                                                        [idx_ts_or_te][idx_ref_time_ts_or_te].append(cur_probs)


            for idx_prob_ls in range(len(prob_ls)):
                if len(prob_ls[idx_prob_ls])>0:
                    if mode == 'Train':
                        prob_ls[idx_prob_ls] = np.mean(prob_ls[idx_prob_ls])
                    else:
                        prob_ls[idx_prob_ls] = np.mean(prob_ls[idx_prob_ls], axis=0)
                else:
                    if mode == 'Train':
                        prob_ls[idx_prob_ls] = 1./len(timestamp_range)
                    else:
                        prob_ls[idx_prob_ls] = np.array([1./len(timestamp_range)] * len(timestamp_range))

        
        if len(cur_valid_rules) == 0:
            continue

        cnt += 1

        # Loop end for one sample

    output_ls = [input_intervals]
    if flag_output_probs_with_ref_edges:
        output_ls.append(output_probs_with_ref_edges)

    # print(input_intervals)
    return output_ls



def prepare_graph_random_walk_res(option, data, mode, num_workers=20, file_suffix=''):
    dataset = data['dataset_name']
    if dataset in ['wiki', 'YAGO']:
        if mode == 'Train':
            idx_ls = data['train_idx_ls']
        else:
            idx_ls = data['test_idx_ls']

        # num_workers = 24
        idx_pieces = split_list_into_pieces(idx_ls, num_workers)
        # print(idx_pieces)
        # sys.exit()
        outputs = Parallel(n_jobs=num_workers)(delayed(create_TEKG_in_batch)(option, data, one_piece, mode) for one_piece in idx_pieces)

        output_probs_with_ref_edges = {}
        for output in outputs:
            output_probs_with_ref_edges.update(output)

        return output_probs_with_ref_edges

    else:
        timestamp_range = data['timestamp_range']
        num_rel = data['num_rel']

        flag_process_rw_res = False
        if (not os.path.exists('output/' + dataset + '/' + dataset + "_"+ mode + "_samples_edge"+ file_suffix + ".json")) and \
           (not os.path.exists('output/' + dataset + '/' + dataset + "_"+ mode + "_samples_edge_rel_0"+ file_suffix + ".json")):
            flag_process_rw_res = True

        flag_process_rw_res = False

        if flag_process_rw_res:
            if dataset in ['icews14', 'icews05-15']:
                if mode == 'Train':
                    file_paths = ['../output/'+ dataset + '/'+ dataset + '_train_walks_suc_with_TR.json']
                else:
                    file_paths = []
                    if dataset == 'icews14':
                        k_ls = [0,1,2,3]
                    else:
                        k_ls = [0,1,2] + ['3_split'+str(i) for i in range(10)]

                    for k in k_ls:
                        file_paths.append('../output/'+ dataset + '/'+ dataset + '_test_samples_part'+ str(k) +'.json')


            elif dataset in ['gdelt']:
                if mode == 'Train':
                    file_paths = ['../output/gdelt_train_walks_suc_with_TR.json']
                else:
                    file_paths = ['../output/gdelt_test_samples.json']

            # print('xsh')

            # cnt = 0
            # for path in file_paths:
            #     # path = '../output/'+ dataset + '/'+ dataset + '_test_samples_part'+ str(0) +'.json'
            #     with open(path, 'r') as file:
            #         data = json.load(file)
            #     cnt += len(data)

            # print(cnt)

            # for k in data:
            #     print(k, data[k])
            #     cnt += 1

            #     if cnt>2:
            #         break

            # sys.exit()

            process_random_walk_results_dist_ver(dataset, mode = mode, num_rel=num_rel, file_paths=file_paths, file_suffix=file_suffix, num_workers=num_workers)

        # sys.exit()


        if dataset in ['icews14', 'icews05-15']:
            dataset1 = dataset
        elif dataset in ['gdelt']:
            dataset1 = 'gdelt_with_num_walks_per_rule_30'

        with open('../output/'+ dataset1 +'/' + dataset + '_pattern_ls'+ file_suffix +'.json', 'r') as file:
            pattern_ls = json.load(file)

        samples = None
        if os.path.exists(dataset + "_"+ mode +"_samples_edge"+ file_suffix + ".json"):
            with open(dataset + '_'+ mode +'_samples_edge'+ file_suffix + '.json', 'r') as file:
                samples = json.load(file)

        with open('../output/'+ dataset1 +'/' + dataset + '_train_stat_res_via_random_walk'+ file_suffix + '.json', 'r') as file:
            stat_res = json.load(file)


        rel_ls = list(range(num_rel//2))
        idx_pieces = split_list_into_pieces(rel_ls, num_workers)
        outputs = Parallel(n_jobs=num_workers)(delayed(create_inputs_v4)\
                                               (samples, one_piece, pattern_ls, timestamp_range, 
                                                num_rel//2, stat_res, mode=mode,
                                                flag_only_find_samples_with_empty_rules=False, dataset=dataset) for one_piece in idx_pieces)

        # input_edge_probs_first_ref, input_edge_probs_last_ref, input_edge_probs_first_ref_inv_rel, input_edge_probs_last_ref_inv_rel = {}, {}, {}, {}
        # for output in outputs:
        #     input_edge_probs_first_ref.update(output[0][0])
        #     input_edge_probs_first_ref.update(output[0][1])
        #     input_edge_probs_first_ref.update(output[0][2])
        #     input_edge_probs_first_ref.update(output[0][3])

        # input_edge_probs = [input_edge_probs_first_ref, input_edge_probs_last_ref, input_edge_probs_first_ref_inv_rel, input_edge_probs_last_ref_inv_rel]

        # return input_edge_probs
        return None








def create_TEKG_in_batch(option, data, idx_ls, mode):
    path = data['path']
    dataset = data['dataset']
    dataset_name = data['dataset_name']

    edges = np.vstack((data['train_edges'], data['test_edges']))
    num_samples_dist = data['num_samples_dist']
    timestamp_range = data['timestamp_range']

    num_rel = data['num_rel']
    num_entity = data['num_entity']

    pattern_ls = data['pattern_ls']
    ts_stat_ls = data['ts_stat_ls']
    te_stat_ls = data['te_stat_ls']

    rm_ls = data['rm_ls']

    prob_type_for_training = option.prob_type_for_training
    num_step = option.num_step-1
    flag_ruleLen_split_ver = option.flag_ruleLen_split_ver
    flag_acceleration = option.flag_acceleration
    num_rule = option.num_rule

    assert data['dataset_name'] in ['wiki', 'YAGO'], 'This function is designed for interval dataset.'

    # print(idx_ls)
    if flag_acceleration:
        output = create_inputs_v3(edges=edges, path=path, 
                                     dataset=dataset_name, 
                                     idx_ls=idx_ls,
                                     pattern_ls=pattern_ls, 
                                     timestamp_range=timestamp_range, 
                                     num_rel=num_rel//2,
                                     ts_stat_ls=ts_stat_ls, 
                                     te_stat_ls=te_stat_ls,
                                     with_ref_end_time=True,
                                     targ_rel=None, num_samples_dist=num_samples_dist, 
                                     mode=mode, rm_ls=rm_ls,
                                     flag_output_probs_with_ref_edges=True,
                                     flag_acceleration=flag_acceleration)
        output_probs_with_ref_edges = output[-1]

        return output_probs_with_ref_edges

    if option.flag_ruleLen_split_ver:
        output = create_inputs_v3_ruleLen_split_ver(edges=edges, path=path, 
                                                     dataset=dataset_name, 
                                                     idx_ls=idx_ls,
                                                     pattern_ls=pattern_ls, 
                                                     timestamp_range=timestamp_range, 
                                                     num_rel=num_rel//2,
                                                     ts_stat_ls=ts_stat_ls, 
                                                     te_stat_ls=te_stat_ls,
                                                     with_ref_end_time=True,
                                                     targ_rel=None, num_samples_dist=num_samples_dist, 
                                                     mode=mode, rm_ls=rm_ls,
                                                     flag_output_probs_with_ref_edges=True)
    else:
        output = create_inputs_v3(edges=edges, path=path, 
                                     dataset=dataset_name, 
                                     idx_ls=idx_ls,
                                     pattern_ls=pattern_ls, 
                                     timestamp_range=timestamp_range, 
                                     num_rel=num_rel//2,
                                     ts_stat_ls=ts_stat_ls, 
                                     te_stat_ls=te_stat_ls,
                                     with_ref_end_time=True,
                                     targ_rel=None, num_samples_dist=num_samples_dist, 
                                     mode=mode, rm_ls=rm_ls,
                                     flag_output_probs_with_ref_edges=True)

    output_probs_with_ref_edges = output[-1]
    return output_probs_with_ref_edges





def create_inputs_v3_ruleLen_split_ver(path, dataset, edges, idx_ls, targ_rel, num_samples_dist, pattern_ls, timestamp_range, num_rel, 
                                     ts_stat_ls, te_stat_ls, mode='Train', rm_ls=None, with_ref_end_time=False, 
                                     only_find_samples_with_empty_rules=False, flag_output_probs_with_ref_edges=False):
    input_intervals = {}
    cnt = 0

    if flag_output_probs_with_ref_edges:
        output_probs_with_ref_edges = {}

    sample_with_empty_rules_ls = []
    for data_idx in idx_ls:
        file = dataset + '_idx_' + str(data_idx) + '.json'
        with open(path + file, 'r') as f:
            data = f.read()
            json_data = json.loads(data)

        if mode == 'Test' and rm_ls !=None:
            if (data_idx - num_samples_dist[0] - num_samples_dist[1]) in rm_ls:
                continue

        if targ_rel != None:
            if json_data['query'][1] != targ_rel:
                continue

        cur_rel = json_data['query'][1]

        cur_output = read_TEILP_results(json_data, with_ref_end_time)

        if mode == 'Train':
            cur_ts = int(json_data['query'][3])
            cur_te = int(json_data['query'][4])
            cur_interval = [cur_ts, cur_te]
        elif mode == 'Test':
            data_idx -= num_samples_dist[1]
            cur_interval = edges[data_idx, 3:]

        input_intervals[data_idx] = cur_interval

        if len(cur_output[cur_rel]) == 0:
            sample_with_empty_rules_ls.append(data_idx)
            continue

        cur_valid_rules = [p for p in cur_output[cur_rel] if p in pattern_ls[cur_rel]]
        if len(cur_valid_rules) == 0:
            sample_with_empty_rules_ls.append(data_idx)
            continue

        if only_find_samples_with_empty_rules:
            continue


        if mode == 'Train':
            dim = 1
        elif mode == 'Test':
            dim = len(timestamp_range)

        if flag_output_probs_with_ref_edges:
            output_probs_with_ref_edges[data_idx] = {'last_event':{}, 'first_event':{}}

        for p in cur_valid_rules:
            # print(p)
            p_idx = pattern_ls[cur_rel].index(p)
            ruleLen = len(p.split(' '))//2

            if not with_ref_end_time:
                prob_ls = {0:[], 1:[], 2:[], 3:[]}
            else:
                prob_ls = {0:[], 1:[], 2:[], 3:[], 4:[], 5:[], 6:[], 7:[]}

            if not with_ref_end_time:
                cur_stat_ls = {0: {0: {}, 1:{}}, 1: {0: {}, 1:{}}}
                cur_stat_ls[0][0] = [ts_stat_ls[cur_rel][p_idx][0], ts_stat_ls[cur_rel][p_idx][1]]
                cur_stat_ls[0][1] = [ts_stat_ls[cur_rel][p_idx][2], ts_stat_ls[cur_rel][p_idx][3]]
                cur_stat_ls[1][0] = [te_stat_ls[cur_rel][p_idx][0], te_stat_ls[cur_rel][p_idx][1]]
                cur_stat_ls[1][1] = [te_stat_ls[cur_rel][p_idx][2], te_stat_ls[cur_rel][p_idx][3]]
            else:
                cur_stat_ls = {0: {0: {0:{}, 1:{}}, 1:{0:{}, 1:{}}}, 1: {0: {0:{}, 1:{}}, 1:{0:{}, 1:{}}}}
                cur_stat_ls[0][0][0] = [ts_stat_ls[cur_rel][p_idx][0], ts_stat_ls[cur_rel][p_idx][1]]
                cur_stat_ls[0][0][1] = [ts_stat_ls[cur_rel][p_idx][2], ts_stat_ls[cur_rel][p_idx][3]]
                cur_stat_ls[0][1][0] = [ts_stat_ls[cur_rel][p_idx][4], ts_stat_ls[cur_rel][p_idx][5]]
                cur_stat_ls[0][1][1] = [ts_stat_ls[cur_rel][p_idx][6], ts_stat_ls[cur_rel][p_idx][7]]

                cur_stat_ls[1][0][0] = [te_stat_ls[cur_rel][p_idx][0], te_stat_ls[cur_rel][p_idx][1]]
                cur_stat_ls[1][0][1] = [te_stat_ls[cur_rel][p_idx][2], te_stat_ls[cur_rel][p_idx][3]]
                cur_stat_ls[1][1][0] = [te_stat_ls[cur_rel][p_idx][4], te_stat_ls[cur_rel][p_idx][5]]
                cur_stat_ls[1][1][1] = [te_stat_ls[cur_rel][p_idx][6], te_stat_ls[cur_rel][p_idx][7]]


            # print(cur_output[cur_rel][p]['time_ref'])
            # print(cur_output[cur_rel][p]['edge_ref'])

            for (idx_time_ref, time_ref) in enumerate(cur_output[cur_rel][p]['time_ref']):
                # print(time_ref)

                if flag_output_probs_with_ref_edges:
                    refEdge = cur_output[cur_rel][p]['edge_ref'][idx_time_ref]

                    if tuple(refEdge[0]) not in output_probs_with_ref_edges[data_idx]['first_event']:
                        if not with_ref_end_time:
                            output_probs_with_ref_edges[data_idx]['first_event'][tuple(refEdge[0])] = {0: {1:[], 2:[], 3:[], 4:[], 5:[]},
                                                                                                       1: {1:[], 2:[], 3:[], 4:[], 5:[]}}
                        else:
                            output_probs_with_ref_edges[data_idx]['first_event'][tuple(refEdge[0])] = {0: {0: {1:[], 2:[], 3:[], 4:[], 5:[]}, 
                                                                                                           1: {1:[], 2:[], 3:[], 4:[], 5:[]}}, 
                                                                                                       1: {0: {1:[], 2:[], 3:[], 4:[], 5:[]}, 
                                                                                                           1: {1:[], 2:[], 3:[], 4:[], 5:[]}}}

                    if tuple(refEdge[1]) not in output_probs_with_ref_edges[data_idx]['last_event']:
                        if not with_ref_end_time:
                            output_probs_with_ref_edges[data_idx]['last_event'][tuple(refEdge[1])] = {0: {1:[], 2:[], 3:[], 4:[], 5:[]},
                                                                                                      1: {1:[], 2:[], 3:[], 4:[], 5:[]}}
                        else:
                            output_probs_with_ref_edges[data_idx]['last_event'][tuple(refEdge[1])] = {0: {0: {1:[], 2:[], 3:[], 4:[], 5:[]}, 
                                                                                                          1: {1:[], 2:[], 3:[], 4:[], 5:[]}}, 
                                                                                                      1: {0: {1:[], 2:[], 3:[], 4:[], 5:[]}, 
                                                                                                          1: {1:[], 2:[], 3:[], 4:[], 5:[]}}}


                for idx_ts_or_te in [0,1]:
                    for idx_first_or_last_event in [0,1]:
                        if not with_ref_end_time:
                            refTime = time_ref[idx_first_or_last_event]
                            delta_t = timestamp_range - refTime
                            cur_Gaussian_mean = cur_stat_ls[idx_ts_or_te][idx_first_or_last_event][0]
                            cur_Gaussian_std = cur_stat_ls[idx_ts_or_te][idx_first_or_last_event][1]
                            idx_prob_ls = 2*idx_ts_or_te + idx_first_or_last_event

                            if (refTime != 9999) and (cur_Gaussian_mean != 9999) and (cur_Gaussian_std != 9999):
                                cur_probs = gaussian_pdf(delta_t, cur_Gaussian_mean, cur_Gaussian_std)

                                if len(cur_probs[np.isnan(cur_probs)])>0 or len(cur_probs[np.isinf(cur_probs)])>0:
                                    continue
                                if np.sum(cur_probs) == 0:
                                    continue

                                cur_probs /= np.sum(cur_probs)

                                if mode == 'Train':
                                    prob_ls[idx_prob_ls].append(cur_probs[delta_t.tolist().index(cur_interval[idx_ts_or_te] - refTime)])
                                elif mode == 'Test':
                                    prob_ls[idx_prob_ls].append(cur_probs)

                        else:
                            for idx_ref_time_ts_or_te in [0,1]:
                                refTime = time_ref[idx_first_or_last_event][idx_ref_time_ts_or_te]
                                delta_t = timestamp_range - refTime
                                cur_Gaussian_mean = cur_stat_ls[idx_ts_or_te][idx_first_or_last_event][idx_ref_time_ts_or_te][0]
                                cur_Gaussian_std = cur_stat_ls[idx_ts_or_te][idx_first_or_last_event][idx_ref_time_ts_or_te][1]
                                idx_prob_ls = 4*idx_ts_or_te + 2*idx_first_or_last_event + idx_ref_time_ts_or_te

                                if (refTime != 9999) and (cur_Gaussian_mean != 9999) and (cur_Gaussian_std != 9999):
                                    cur_probs = gaussian_pdf(delta_t, cur_Gaussian_mean, cur_Gaussian_std)

                                    if len(cur_probs[np.isnan(cur_probs)])>0 or len(cur_probs[np.isinf(cur_probs)])>0:
                                        continue
                                    if np.sum(cur_probs) == 0:
                                        continue

                                    cur_probs /= np.sum(cur_probs)

                                    if mode == 'Train':
                                        # if (cur_interval[idx_ts_or_te] - refTime) not in delta_t.tolist():
                                        #     print(data_idx, cur_interval, refTime, delta_t)
                                        cur_probs_for_gt = cur_probs[delta_t.tolist().index(cur_interval[idx_ts_or_te] - refTime)]
                                        prob_ls[idx_prob_ls].append(cur_probs_for_gt)

                                        if flag_output_probs_with_ref_edges:
                                            if idx_first_or_last_event == 0:
                                                output_probs_with_ref_edges[data_idx]['first_event'][tuple(refEdge[idx_first_or_last_event])]\
                                                                        [idx_ts_or_te][idx_ref_time_ts_or_te][ruleLen].append(cur_probs_for_gt)
                                            else:
                                                output_probs_with_ref_edges[data_idx]['last_event'][tuple(refEdge[idx_first_or_last_event])]\
                                                                        [idx_ts_or_te][idx_ref_time_ts_or_te][ruleLen].append(cur_probs_for_gt)

                                    elif mode == 'Test':
                                        prob_ls[idx_prob_ls].append(cur_probs)

                                        if flag_output_probs_with_ref_edges:
                                            if idx_first_or_last_event == 0:
                                                output_probs_with_ref_edges[data_idx]['first_event'][tuple(refEdge[idx_first_or_last_event])]\
                                                                        [idx_ts_or_te][idx_ref_time_ts_or_te][ruleLen].append(cur_probs)
                                            else:
                                                output_probs_with_ref_edges[data_idx]['last_event'][tuple(refEdge[idx_first_or_last_event])]\
                                                                        [idx_ts_or_te][idx_ref_time_ts_or_te][ruleLen].append(cur_probs)


            for idx_prob_ls in range(len(prob_ls)):
                if len(prob_ls[idx_prob_ls])>0:
                    if mode == 'Train':
                        prob_ls[idx_prob_ls] = np.mean(prob_ls[idx_prob_ls])
                    elif mode == 'Test':
                        prob_ls[idx_prob_ls] = np.mean(prob_ls[idx_prob_ls], axis=0)
                else:
                    if mode == 'Train':
                        prob_ls[idx_prob_ls] = 1./len(timestamp_range)
                    elif mode == 'Test':
                        prob_ls[idx_prob_ls] = np.array([1./len(timestamp_range)] * len(timestamp_range))

            # print('--------------------------')

            # Loop end for rules in one sample


        # print('--------------------------')

        if len(cur_valid_rules) == 0:
            continue

        # input_intervals.append(cur_interval)
        cnt += 1

        # Loop end for one sample


    output_ls = [input_intervals]
    if flag_output_probs_with_ref_edges:
        output_ls.append(output_probs_with_ref_edges)

    return output_ls






def prepare_whole_TEKG_graph(data):
    masked_edges = np.vstack((data['valid_edges'], data['test_edges']))
    masked_edges[:, 3:] = 9999
    masked_edges = np.unique(masked_edges, axis=0)

    batch_edges = np.vstack((data['train_edges'], masked_edges))
    # print(batch_edges)

    num_rel = data['num_rel']

    batch_edges_idx = np.arange(len(batch_edges))
    num_entity = len(batch_edges)*2

    mdb = {}
    for r in range(num_rel//2):
        idx_cur_rel = batch_edges_idx[batch_edges[:, 1] == r].reshape((-1,1))
        if len(idx_cur_rel) == 0:
            mdb[r] = [[[0,0]], [0.0], [num_entity, num_entity]]
            mdb[r + num_rel//2] = [[[0,0]], [0.0], [num_entity, num_entity]]
        else:
            mdb[r] = [np.hstack([idx_cur_rel, idx_cur_rel]).tolist(), [1.0]*len(idx_cur_rel), [num_entity, num_entity]]
            mdb[r + num_rel//2] = [np.hstack([idx_cur_rel + num_entity//2, idx_cur_rel + num_entity//2]).tolist(), 
                                        [1.0]*len(idx_cur_rel), [num_entity, num_entity]]


    batch_edges_inv = batch_edges[:, [2,1,0,3,4]]
    batch_edges_inv[:,1] += num_rel//2
    batch_edges = np.vstack((batch_edges, batch_edges_inv))
    batch_edges_idx_cmp = np.hstack((batch_edges_idx, batch_edges_idx + num_entity//2))

    connectivity = {}  # [x,y] means y to x
    connectivity[0] = []
    connectivity[1] = []
    connectivity[2] = []
    connectivity[3] = []


    for ent in np.unique(np.hstack([batch_edges[:, 0], batch_edges[:, 2]])):
        b = batch_edges_idx_cmp[batch_edges[:, 2] == ent]
        a = batch_edges_idx_cmp[batch_edges[:, 0] == ent]
        combinations = list(itertools.product(a, b))
        combinations = np.array(combinations)

        combinations_TR = combinations.copy()
        combinations_TR[combinations_TR>=num_entity//2] -= num_entity//2
        TRs = calculate_TR_mat_ver(batch_edges[combinations_TR[:,0], 3:], batch_edges[combinations_TR[:,1], 3:])

        connectivity[0] += combinations.tolist()
        connectivity[1] += combinations[TRs==1].tolist()
        connectivity[2] += combinations[TRs==2].tolist()
        connectivity[3] += combinations[TRs==3].tolist()


    for TR in range(4):
        if len(connectivity[TR])>0:
            A = np.array(connectivity[TR])
            B = A.copy()
            B[B>=num_entity//2] -= num_entity//2
            A = A[~(B[:,0] == B[:,1])]
            if len(A) == 0:
                connectivity[TR] = [[[0,0]], [0.0], [num_entity, num_entity]]
            else:
                A = np.unique(A, axis=0)

                connectivity[TR] = [A.tolist()]
                connectivity[TR].append([1.0] * len(connectivity[TR][0]))
                connectivity[TR].append([num_entity, num_entity])
        else:
            connectivity[TR] = [[[0,0]], [0.0], [num_entity, num_entity]]

    return mdb, connectivity, batch_edges, num_entity






def create_inputs_v4(samples, samples_inv, targ_rels, pattern_ls, timestamp_range, num_rel, stat_res, mode=None,
                     flag_only_find_samples_with_empty_rules=False, dataset=None, file_suffix='', 
                     flag_compression=True, flag_write=False, flag_time_shifting=False, pattern_ls_fkt=None, stat_res_fkt=None,
                     shift_ref_time=None, flag_rm_seen_timestamp=False):

    # input_pattern_probs_first_ref = []
    # input_pattern_probs_last_ref = []
    # input_pattern_probs_first_ref_inv_rel = []
    # input_pattern_probs_last_ref_inv_rel = []

    input_edge_probs_first_ref = {}
    input_edge_probs_last_ref = {}
    input_edge_probs_first_ref_inv_rel = {}
    input_edge_probs_last_ref_inv_rel = {}

    input_samples = []
    cnt = 0

    sample_with_empty_rules_ls = []

    if samples is None:
        samples = {}
        samples_inv = {}
        for rel in targ_rels:
            with open('output/' + dataset + '/' + dataset + '_'+ mode +'_samples_edge_rel_'+ str(rel) + file_suffix + '.json', 'r') as file:
                samples.update(json.load(file))
            with open('output/' + dataset + '/' + dataset + '_'+ mode +'_samples_edge_rel_'+ str(rel + num_rel) + file_suffix + '.json', 'r') as file:
                samples_inv.update(json.load(file))

    dataset_index = ['wiki', 'YAGO', 'icews14', 'icews05-15', 'gdelt100'].index(dataset)
    relaxed_std = [10, 10, 30, 300, 30][dataset_index]
    max_mean = [100, 100, 120, 900, 120][dataset_index]

    edge_probs = []
    for e_str in samples:
        # print(samples[e_str])
        # continue
        # if isinstance(e_str, str):
        #     e = [int(num) for num in e_str[1:-1].split(',')]
        # else:
        #     e = e_str

        e = [int(num) for num in e_str[1:-1].split(',')]

        if targ_rels is not None:
            if e[1] not in targ_rels:
                continue

        cur_rel = e[1]
        cur_ts = e[3]

        e_inv = obtain_inv_edge(np.array([e]), num_rel)[0]
        cur_rel_inv = e_inv[1]
        e_inv_str = str(tuple(e_inv))

        if len(samples[e_str]) == 0:
            if e_inv_str not in samples_inv:
                sample_with_empty_rules_ls.append(e)
                continue
            elif len(samples_inv[e_inv_str]) == 0:
                sample_with_empty_rules_ls.append(e)
                continue


        # print(samples)
        # print(samples[e_str])
        # print(samples_inv)
        # print(samples_inv[e_inv_str])
        # print(pattern_ls[str(cur_rel)])
        # print('----------------------')


        cur_valid_rules = [p for p in samples[e_str] if p in pattern_ls[str(cur_rel)]]
        cur_valid_rules_inv_rel = []
        if e_inv_str in samples_inv:
            if not flag_time_shifting:
                cur_valid_rules_inv_rel = [p for p in samples_inv[e_inv_str] if p in pattern_ls[str(cur_rel_inv)]]
            else:
                cur_valid_rules_inv_rel = [p for p in samples_inv[e_inv_str] if p in pattern_ls_fkt[str(cur_rel_inv)]]


        if len(cur_valid_rules) == 0 and len(cur_valid_rules_inv_rel) == 0:
            sample_with_empty_rules_ls.append(e)
            continue

        if flag_only_find_samples_with_empty_rules:
            continue

        if mode == 'Train':
            dim = 1
        elif mode == 'Test':
            dim = len(timestamp_range)

        # print(e)
        # print(e_inv)
        # print(samples[e_str])
        # print(samples_inv[e_inv_str])
        # print(cur_valid_rules)
        # print(cur_valid_rules_inv_rel)
        # sys.exit()

        type_ref_time = ['ts_first_event', 'ts_last_event']

        # cur_pattern_probs_first_ref = np.zeros((dim, len(pattern_ls[str(cur_rel)])))
        # cur_pattern_probs_last_ref = np.zeros((dim, len(pattern_ls[str(cur_rel)])))

        cur_edge_probs_first_ref = {}
        cur_edge_probs_last_ref = {}

        for p in cur_valid_rules:
            p_idx = pattern_ls[str(cur_rel)].index(p)
            prob_ls = {0:[], 1:[]}

            for idx in range(2):
                mean_ls = stat_res[str(cur_rel)][p][type_ref_time[idx]]['mean_ls']
                std_ls = stat_res[str(cur_rel)][p][type_ref_time[idx]]['std_ls']
                prop_ls = stat_res[str(cur_rel)][p][type_ref_time[idx]]['prop_ls']

                if mode == 'Train':
                    time_gap = cur_ts - np.array(samples[e_str][p])[:, idx, 3]
                    prob_ls[idx] = calculate_prob_for_adaptive_Gaussian_dist(time_gap, mean_ls, std_ls, prop_ls, relaxed_std, max_mean)

                else:
                    if flag_compression:
                        prob_ls[idx] = np.array(samples[e_str][p])[:, idx, 3]
                    else:
                        time_gap = []
                        for ref_time in np.array(samples[e_str][p])[:, idx, 3]:
                            time_gap.append(timestamp_range - ref_time)
                        max_ref_time = max(np.array(samples[e_str][p])[:, idx, 3])
                        if shift_ref_time is not None:
                            max_ref_time = copy.copy(shift_ref_time)
                        time_gap = np.hstack(time_gap)
                        cur_probs = calculate_prob_for_adaptive_Gaussian_dist(time_gap, mean_ls, std_ls, prop_ls, relaxed_std, max_mean)
                        # if np.sum(cur_probs) == 0:
                        #     print(time_gap, mean_ls, std_ls, prop_ls)
                        # if flag_time_shifting:
                            # cur_probs[time_gap < 0] = 0
                        prob_ls[idx] = cur_probs
                        prob_ls[idx] = cut_vector(prob_ls[idx], len(samples[e_str][p]))
                        if flag_time_shifting and flag_rm_seen_timestamp:
                            # print(prob_ls[idx])
                            prob_ls[idx] = rm_seen_timestamp(prob_ls[idx], timestamp_range, max_ref_time, [time_gap, mean_ls, std_ls, prop_ls, relaxed_std])
                            # print(prob_ls[idx])
                            # # print(timestamp_range)
                            # print(max_ref_time)
                            # for prob_tmp in prob_ls[idx]:
                            #     print(prob_tmp[:max_ref_time])
                            # print('----------------')

                # print(prob_ls[idx])
                # print('----------------')

            # print(samples[e_str][p])
            # print(prob_ls[0])
            # print(prob_ls[1])
            # sys.exit()

            for idx_edge in range(len(samples[e_str][p])):
                if tuple(samples[e_str][p][idx_edge][0]) not in cur_edge_probs_first_ref:
                    cur_edge_probs_first_ref[tuple(samples[e_str][p][idx_edge][0])] = []
                if mode == 'Train':
                    cur_edge_probs_first_ref[tuple(samples[e_str][p][idx_edge][0])].append([prob_ls[0][idx_edge], p_idx])
                else:
                    cur_edge_probs_first_ref[tuple(samples[e_str][p][idx_edge][0])].append([prob_ls[0][idx_edge].tolist(), p_idx])

                if tuple(samples[e_str][p][idx_edge][1]) not in cur_edge_probs_last_ref:
                    cur_edge_probs_last_ref[tuple(samples[e_str][p][idx_edge][1])] = []
                if mode == 'Train':
                    cur_edge_probs_last_ref[tuple(samples[e_str][p][idx_edge][1])].append([prob_ls[1][idx_edge], p_idx])
                else:
                    cur_edge_probs_last_ref[tuple(samples[e_str][p][idx_edge][1])].append([prob_ls[1][idx_edge].tolist(), p_idx])


            #     if mode == 'Train':
            #         prob_ls[idx] = np.mean(prob_ls[idx])
            #     else:
            #         prob_ls[idx] = np.mean(prob_ls[idx], axis=0)

            # cur_pattern_probs_first_ref[:, p_idx] = prob_ls[0]
            # cur_pattern_probs_last_ref[:, p_idx] = prob_ls[1]


        # if not flag_time_shifting:
        #     cur_pattern_probs_first_ref_inv_rel = np.zeros((dim, len(pattern_ls[str(cur_rel_inv)])))
        #     cur_pattern_probs_last_ref_inv_rel = np.zeros((dim, len(pattern_ls[str(cur_rel_inv)])))
        # else:
        #     cur_pattern_probs_first_ref_inv_rel = np.zeros((dim, len(pattern_ls_fkt[str(cur_rel_inv)])))
        #     cur_pattern_probs_last_ref_inv_rel = np.zeros((dim, len(pattern_ls_fkt[str(cur_rel_inv)])))

        cur_edge_probs_first_ref_inv_rel = {}
        cur_edge_probs_last_ref_inv_rel = {}

        for p in cur_valid_rules_inv_rel:
            if not flag_time_shifting:
                p_idx = pattern_ls[str(cur_rel_inv)].index(p)
            else:
                p_idx = pattern_ls_fkt[str(cur_rel_inv)].index(p)

            prob_ls = {0:[], 1:[]}

            for idx in range(2):
                if not flag_time_shifting:
                    mean_ls = stat_res[str(cur_rel_inv)][p][type_ref_time[idx]]['mean_ls']
                    std_ls = stat_res[str(cur_rel_inv)][p][type_ref_time[idx]]['std_ls']
                    prop_ls = stat_res[str(cur_rel_inv)][p][type_ref_time[idx]]['prop_ls']
                else:
                    mean_ls = stat_res_fkt[str(cur_rel_inv)][p][type_ref_time[idx]]['mean_ls']
                    std_ls = stat_res_fkt[str(cur_rel_inv)][p][type_ref_time[idx]]['std_ls']
                    prop_ls = stat_res_fkt[str(cur_rel_inv)][p][type_ref_time[idx]]['prop_ls']

                if mode == 'Train':
                    time_gap = cur_ts - np.array(samples_inv[e_inv_str][p])[:, idx, 3]
                    prob_ls[idx] = calculate_prob_for_adaptive_Gaussian_dist(time_gap, mean_ls, std_ls, prop_ls, relaxed_std, max_mean)
                else:
                    if flag_compression:
                        prob_ls[idx] = np.array(samples_inv[e_inv_str][p])[:, idx, 3]
                    else:
                        time_gap = []
                        for ref_time in np.array(samples_inv[e_inv_str][p])[:, idx, 3]:
                            time_gap.append(timestamp_range - ref_time)
                        max_ref_time = max(np.array(samples_inv[e_inv_str][p])[:, idx, 3])
                        if shift_ref_time is not None:
                            max_ref_time = copy.copy(shift_ref_time)
                        time_gap = np.hstack(time_gap)
                        cur_probs = calculate_prob_for_adaptive_Gaussian_dist(time_gap, mean_ls, std_ls, prop_ls, relaxed_std, max_mean)
                        # cur_probs[time_gap < 0] = 0
                        if np.sum(cur_probs) == 0:
                            print(time_gap, mean_ls, std_ls, prop_ls)
                        prob_ls[idx] = cur_probs
                        prob_ls[idx] = cut_vector(prob_ls[idx], len(samples_inv[e_inv_str][p]))
                        if flag_time_shifting and flag_rm_seen_timestamp:
                            prob_ls[idx] = rm_seen_timestamp(prob_ls[idx], timestamp_range, max_ref_time, [time_gap, mean_ls, std_ls, prop_ls, relaxed_std])

                # print(prob_ls[idx])
                # print('----------------')


            for idx_edge in range(len(samples_inv[e_inv_str][p])):
                if tuple(samples_inv[e_inv_str][p][idx_edge][0]) not in cur_edge_probs_first_ref_inv_rel:
                    cur_edge_probs_first_ref_inv_rel[tuple(samples_inv[e_inv_str][p][idx_edge][0])] = []
                if mode == 'Train':
                    cur_edge_probs_first_ref_inv_rel[tuple(samples_inv[e_inv_str][p][idx_edge][0])].append([prob_ls[0][idx_edge], p_idx])
                else:
                    cur_edge_probs_first_ref_inv_rel[tuple(samples_inv[e_inv_str][p][idx_edge][0])].append([prob_ls[0][idx_edge].tolist(), p_idx])

                if tuple(samples_inv[e_inv_str][p][idx_edge][1]) not in cur_edge_probs_last_ref_inv_rel:
                    cur_edge_probs_last_ref_inv_rel[tuple(samples_inv[e_inv_str][p][idx_edge][1])] = []
                if mode == 'Train':
                    cur_edge_probs_last_ref_inv_rel[tuple(samples_inv[e_inv_str][p][idx_edge][1])].append([prob_ls[1][idx_edge], p_idx])
                else:
                    cur_edge_probs_last_ref_inv_rel[tuple(samples_inv[e_inv_str][p][idx_edge][1])].append([prob_ls[1][idx_edge].tolist(), p_idx])

                # if mode == 'Train':
                #     prob_ls[idx] = np.max(prob_ls[idx])
                # else:
                #     prob_ls[idx] = np.mean(prob_ls[idx], axis=0)

            # cur_pattern_probs_first_ref_inv_rel[:, p_idx] = prob_ls[0]
            # cur_pattern_probs_last_ref_inv_rel[:, p_idx] = prob_ls[1]


        # input_pattern_probs_first_ref.append(cur_pattern_probs_first_ref)
        # input_pattern_probs_last_ref.append(cur_pattern_probs_last_ref)
        # input_pattern_probs_first_ref_inv_rel.append(cur_pattern_probs_first_ref_inv_rel)
        # input_pattern_probs_last_ref_inv_rel.append(cur_pattern_probs_last_ref_inv_rel)

        # input_edge_probs_first_ref[tuple(e)] = cur_edge_probs_first_ref
        # input_edge_probs_last_ref[tuple(e)] = cur_edge_probs_last_ref
        # input_edge_probs_first_ref_inv_rel[tuple(e)] = cur_edge_probs_first_ref_inv_rel
        # input_edge_probs_last_ref_inv_rel[tuple(e)] = cur_edge_probs_last_ref_inv_rel

        # input_samples.append(np.array([e]))
        cnt += 1



        cur_edge_probs = {'input_edge_probs_first_ref': convert_dict(cur_edge_probs_first_ref), 
                          'input_edge_probs_last_ref': convert_dict(cur_edge_probs_last_ref),
                          'input_edge_probs_first_ref_inv_rel': convert_dict(cur_edge_probs_first_ref_inv_rel), 
                          'input_edge_probs_last_ref_inv_rel': convert_dict(cur_edge_probs_last_ref_inv_rel)}

        edge_probs.append(cur_edge_probs)

        if flag_write:
            # Open a file in write mode
            with open("output/"+ dataset + '/' + dataset +"_"+ mode +"_input_edge_probs_edge_"+ str(e) + file_suffix + ".json", "w") as json_file:
                # Write the dictionary to the file in JSON format
                json.dump(cur_edge_probs, json_file)


        # Loop end for one sample

    # Loop end for all samples

    # if cnt>0:
    #     # if mode == 'Train':
    #     #     input_pattern_probs_first_ref = np.vstack(input_pattern_probs_first_ref)
    #     #     input_pattern_probs_last_ref = np.vstack(input_pattern_probs_last_ref)
    #     #     input_pattern_probs_first_ref_inv_rel = np.vstack(input_pattern_probs_first_ref_inv_rel)
    #     #     input_pattern_probs_last_ref_inv_rel = np.vstack(input_pattern_probs_last_ref_inv_rel)

    #     input_samples = np.vstack(input_samples)


    # input_pattern_probs = [input_pattern_probs_first_ref,
    #                        input_pattern_probs_last_ref,
    #                        input_pattern_probs_first_ref_inv_rel,
    #                        input_pattern_probs_last_ref_inv_rel]


    # input_edge_probs_first_ref1 = {}
    # for e in input_edge_probs_first_ref:
    #     input_edge_probs_first_ref1[str(e)] = input_edge_probs_first_ref[e]


    # input_edge_probs = {'input_edge_probs_first_ref': input_edge_probs_first_ref, 'input_edge_probs_last_ref': input_edge_probs_last_ref,
    #                     'input_edge_probs_first_ref_inv_rel': input_edge_probs_first_ref_inv_rel, 'input_edge_probs_last_ref_inv_rel': input_edge_probs_last_ref_inv_rel}

    # # Open a file in write mode
    # with open("input_edge_probs_rel_"+ str(targ_rels) +".json", "w") as json_file:
    #     # Write the dictionary to the file in JSON format
    #     json.dump(input_edge_probs, json_file)

    # input_edge_probs = [input_edge_probs_first_ref, input_edge_probs_last_ref, input_edge_probs_first_ref_inv_rel, input_edge_probs_last_ref_inv_rel]

    # output_ls = [input_edge_probs]

    # # for x in input_pattern_probs:
    # #     print(x.shape)
    # #     print('-----------------')

    # # print(input_ts)
    # # print('-----------------')
    # # print(sample_with_empty_rules_ls)

    # return output_ls
    return edge_probs


def convert_dict(original_dict):
    converted_dict = {str(key): value for key, value in original_dict.items()}
    return converted_dict


def inv_convert_dict(original_dict):
    converted_dict = {tuple([int(num) for num in key[1:-1].split(',')]): value for key, value in original_dict.items()}
    return converted_dict


def cut_vector(vector, num):
    sub_vector_length = len(vector) // num
    sub_vectors = [vector[i * sub_vector_length: (i + 1) * sub_vector_length] for i in range(num)]
    return sub_vectors


def rm_seen_timestamp(probs_ls, timestamp_range, ref_time, tmp=None):
    output = []
    for probs in probs_ls:
        # probs_cp = probs.copy()
        probs[timestamp_range < ref_time] = 0
        # if np.sum(probs) == 0:
        #     print(probs_cp)
        #     print(timestamp_range, ref_time)
        #     print(tmp)
        #     print('------------------------')
        # if np.sum(probs) == 0:
        #     probs[timestamp_range >= ref_time] = 1
        # probs = probs / np.sum(probs)
        output.append(probs)
    return output


def generate_exp_dist(weight, scale, offset, timestamp_range, ref_time):
    dist = np.zeros((len(timestamp_range), ))
    time_gap = timestamp_range - ref_time
    dist[time_gap>=offset] = weight * expon.pdf(time_gap[time_gap>=offset], scale=scale)
    # print(dist[time_gap>=offset])
    return dist


def apply_kmeans(lst):
    # Reshape the list to a 2-dimensional array
    X = [[x] for x in lst]

    # Define the range of possible k values
    k_values = range(2, len(lst))

    # Perform K-means clustering for each k value
    sil_scores = []
    sum_sq_distances = []
    sublists = []
    for k in k_values:
        kmeans = KMeans(n_clusters=k)
        kmeans.fit(X)
        labels = kmeans.labels_
        
        # Calculate the silhouette score and sum of squared distances
        sil_score = silhouette_score(X, labels)
        sum_sq_distance = kmeans.inertia_

        sil_scores.append(sil_score)
        sum_sq_distances.append(sum_sq_distance)
        
        # Create sublists based on the labels
        sublist = [[] for _ in range(k)]
        for i, label in enumerate(labels):
            sublist[label].append(lst[i])
        sublists.append(sublist)

    # Find the index of the optimal k value based on the elbow method
    optimal_idx = find_optimal_k(sil_scores)

    # Return the sublists corresponding to the optimal k value
    optimal_sublists = sublists[optimal_idx]

    # # Plot the results
    # plt.figure(figsize=(10, 4))
    # plt.subplot(1, 2, 1)
    # plt.plot(k_values, sil_scores, 'bo-')
    # plt.xlabel('Number of Clusters (k)')
    # plt.ylabel('Silhouette Score')

    # plt.subplot(1, 2, 2)
    # plt.plot(k_values, sum_sq_distances, 'bo-')
    # plt.xlabel('Number of Clusters (k)')
    # plt.ylabel('Sum of Squared Distances')

    # plt.tight_layout()
    # plt.show()
    
    return optimal_sublists

def find_optimal_k(scores):
    # Find the index of the maximum score
    max_score_idx = scores.index(max(scores))
    
    # Check for the elbow point using the second derivative
    prev_diff = scores[1] - scores[0]
    optimal_idx = 1
    for i in range(2, len(scores)):
        curr_diff = scores[i] - scores[i-1]
        if curr_diff < prev_diff:
            optimal_idx = i - 1
            break
        prev_diff = curr_diff
    
    # If no elbow point is found, use the maximum score index
    if optimal_idx == 1:
        optimal_idx = max_score_idx
    
    return optimal_idx



def apply_kmeans_with_k(lst, k):
    # Reshape the list to a 2-dimensional array
    X = [[x] for x in lst]
    
    # Create a KMeans object and fit the data
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(X)
    
    # Get the labels assigned to each data point
    labels = kmeans.labels_
    
    # Create sublists based on the labels
    sublists = [[] for _ in range(k)]
    for i, label in enumerate(labels):
        sublists[label].append(lst[i])
    
    return sublists


def adaptive_Gaussian_dist_estimate_new_ver(input):
    unique_input = np.unique(input)
    if len(unique_input)>20:
        input_split = apply_kmeans_with_k(unique_input, k=5)
    elif len(unique_input)>10:
        input_split = apply_kmeans_with_k(unique_input, k=3)
    else:
        input_split = [unique_input]

    mean_ls = []
    std_ls = []
    prop_ls = []
    for sublist in input_split:
        mean_ls.append(np.mean(sublist))
        std_ls.append(max(5, np.std(sublist)))

        cur_num_samples_real = 0
        for element in sublist:
            cur_num_samples_real += len([value for value in input if value == element])

        prop_ls.append(1.0 * cur_num_samples_real)

    prop_ls /= np.sum(prop_ls)

    return mean_ls, std_ls, prop_ls.tolist()


def adaptive_Gaussian_dist_estimate(input):
    if len(input)>20:
        input_split = apply_kmeans_with_k(input, k=5)
    elif len(input)>10:
        input_split = apply_kmeans_with_k(input, k=3)
    else:
        input_split = [input]

    mean_ls = []
    std_ls = []
    prop_ls = []
    for sublist in input_split:
        mean_ls.append(np.mean(sublist))
        std_ls.append(max(5, np.std(sublist)))
        prop_ls.append(1.0 * len(sublist))

    prop_ls /= np.sum(prop_ls)

    return mean_ls, std_ls, prop_ls.tolist()



def calculate_prob_for_adaptive_Gaussian_dist(values, mean_ls, std_ls, prop_ls, relaxed_std, max_mean, flag_correct=False):
    probs = []
    for i in range(len(mean_ls)):
        if flag_correct:
            prob_cluster = gaussian_pdf(values, min(max_mean, mean_ls[i]), std_ls[i]).reshape(-1,1)
            if np.sum(prob_cluster) == 0:
                prob_cluster = gaussian_pdf(values, min(max_mean, mean_ls[i]), relaxed_std).reshape(-1,1)
        else:
            prob_cluster = gaussian_pdf(values, mean_ls[i], std_ls[i]).reshape(-1,1)

        probs.append(prob_cluster)

    probs = np.hstack(probs)
    # print(probs)
    # print(prop_ls)
    max_index = np.argmax(probs, axis=1)

    output = np.array(prop_ls)[max_index] * np.max(probs, axis=1)
    # print(output)
    # print('----------------------------')
    return output


def obtain_feature_maps_stat(outDict, dataset, flag_time_shifting=False):
    type_ls = ['hh', 'ht', 'th', 'tt']
    stat_dict = {}
    for single_outDict in outDict:
        cur_rel = list(single_outDict['Target'].keys())[0]
        cur_interval = single_outDict['Target'][cur_rel]
        if cur_rel not in stat_dict:
            stat_dict[cur_rel] = {}
        for type1 in type_ls:
            if type1 not in stat_dict[cur_rel]:
                stat_dict[cur_rel][type1] = {}
            for related_rel in single_outDict[type1]:
                if related_rel not in stat_dict[cur_rel][type1]:
                    stat_dict[cur_rel][type1][related_rel] = {'time_gap_ts_ref_ts': [], 'time_gap_ts_ref_te': [], \
                                                              'time_gap_te_ref_ts': [], 'time_gap_te_ref_te': []}

                time_gap_ts_ref_ts = (cur_interval[0] - single_outDict[type1][related_rel][:, 0]).tolist()
                time_gap_ts_ref_te = (cur_interval[0] - single_outDict[type1][related_rel][:, 1]).tolist()
                time_gap_te_ref_ts = (cur_interval[1] - single_outDict[type1][related_rel][:, 0]).tolist()
                time_gap_te_ref_te = (cur_interval[1] - single_outDict[type1][related_rel][:, 1]).tolist()

                # if np.max(np.abs(time_gap_ts))>100:
                #     print(cur_interval, single_outDict[type1][related_rel])

                # if np.max(np.abs(time_gap_te))>100:
                #     print(cur_interval, single_outDict[type1][related_rel])

                stat_dict[cur_rel][type1][related_rel]['time_gap_ts_ref_ts'] += time_gap_ts_ref_ts
                stat_dict[cur_rel][type1][related_rel]['time_gap_ts_ref_te'] += time_gap_ts_ref_te
                stat_dict[cur_rel][type1][related_rel]['time_gap_te_ref_ts'] += time_gap_te_ref_ts
                stat_dict[cur_rel][type1][related_rel]['time_gap_te_ref_te'] += time_gap_te_ref_te


    for targ_rel in stat_dict:
        for type1 in type_ls:
            for related_rel in stat_dict[targ_rel][type1]:
                for time_gap_type in stat_dict[targ_rel][type1][related_rel]:
                    mean_ls, std_ls, prop_ls = adaptive_Gaussian_dist_estimate_new_ver(stat_dict[targ_rel][type1][related_rel][time_gap_type])
                    # print(stat_dict[targ_rel][type1][related_rel][time_gap_type])
                    stat_dict[targ_rel][type1][related_rel][time_gap_type] = {'mean_ls': mean_ls, 'std_ls': std_ls, 'prop_ls': prop_ls}
                    print(stat_dict[targ_rel][type1][related_rel][time_gap_type])

    if not flag_time_shifting:
        with open(dataset + '_stat_sample_with_empty_rules_ls.json', 'w') as file:
            json.dump(stat_dict, file)
    else:
        with open(dataset + '_stat_sample_with_empty_rules_ls_time_shifting.json', 'w') as file:
            json.dump(stat_dict, file)



def calculate_prob_for_feature_maps(outDict, dataset, num_rel, targ_rel, flag_time_shifting=False):
    type_ls = ['hh', 'ht', 'th', 'tt']
    idx_cur_interval = [0, 0, 1, 1]
    idx_ref_time = [0, 1, 0, 1]
    time_gap_type = ['time_gap_ts_ref_ts', 'time_gap_ts_ref_te', 'time_gap_te_ref_ts', 'time_gap_te_ref_te']

    if not flag_time_shifting:
        with open(dataset + '_stat_sample_with_empty_rules_ls.json', 'r') as file:
            stat_dict = json.load(file)
    else:
        with open(dataset + '_stat_sample_with_empty_rules_ls_time_shifting.json', 'r') as file:
            stat_dict = json.load(file)

    output_probs = {}
    for i in range(len(time_gap_type)):
        output_probs[time_gap_type[i]] = []

    for (j, single_outDict) in enumerate(outDict):
        # print(j, single_outDict)
        cur_rel = list(single_outDict['Target'].keys())[0]
        if cur_rel != targ_rel:
            continue
        cur_interval = single_outDict['Target'][cur_rel]

        cur_num_valid_events = 0
        for type1 in type_ls:
            cur_num_valid_events += len(single_outDict[type1])

        if cur_num_valid_events == 0:
            continue

        cur_pattern_probs = {}
        for i in range(len(time_gap_type)):
            cur_pattern_probs[time_gap_type[i]] = []
            for type1 in type_ls:
                cur_type_pattern_prob = np.zeros([1, num_rel])
                for related_rel in single_outDict[type1]:
                    time_gap = cur_interval[idx_cur_interval[i]] - single_outDict[type1][related_rel][:, idx_ref_time[i]]
                    # print(str(cur_rel), str(related_rel), stat_dict[str(cur_rel)][type1].keys())
                    mean_ls = stat_dict[str(cur_rel)][type1][str(related_rel)][time_gap_type[i]]['mean_ls']
                    std_ls = stat_dict[str(cur_rel)][type1][str(related_rel)][time_gap_type[i]]['std_ls']
                    prop_ls = stat_dict[str(cur_rel)][type1][str(related_rel)][time_gap_type[i]]['prop_ls']
                    cur_type_pattern_prob[0, related_rel] = np.mean(calculate_prob_for_adaptive_Gaussian_dist(time_gap, mean_ls, std_ls, prop_ls))
                cur_pattern_probs[time_gap_type[i]].append(cur_type_pattern_prob)
            cur_pattern_probs[time_gap_type[i]] = np.hstack(cur_pattern_probs[time_gap_type[i]])

            # print(cur_pattern_probs[time_gap_type[i]].shape)
            output_probs[time_gap_type[i]].append(cur_pattern_probs[time_gap_type[i]])

        # print(len(output_probs[time_gap_type[0]]))


    for i in range(len(time_gap_type)):
        # print(targ_rel, time_gap_type[i], len(output_probs[time_gap_type[i]]))
        if len(output_probs[time_gap_type[i]])>0:
            output_probs[time_gap_type[i]] = np.vstack(output_probs[time_gap_type[i]])

    # for (j, cur_pattern_probs) in enumerate(output_prob):
    #     print(outDict[j])
    #     print(cur_pattern_probs)
    #     print('------------------------')

    return output_probs


def find_range(ranges, num):
    for (i,range_) in enumerate(ranges):
        if range_[0] <= num <= range_[1]:
            return i
    return None


def save_data(filename, line):
    # Get the current date and time
    current_datetime = datetime.now()

    # Open the file in append mode
    with open(filename, "a") as file:
        file.write(str(current_datetime) + ' ' + str(line) + "\n")






class DataLoader():
    def __init__(self, option, data):
        self.option = option
        self.data = data


    def one_epoch(self, mode, idx_ls, idx_batch_ls):
        dataset = self.data['dataset']
        for i in idx_batch_ls:
            batch_idx_ls = idx_ls[i]

            if self.option.flag_acceleration:
                qq, query_rels, refNode_source, res_random_walk, probs, valid_sample_idx, input_intervals, input_samples = self.create_TEKG_in_batch(batch_idx_ls, mode)
            else:
                qq, hh, tt, mdb, connectivity, probs, valid_sample_idx, valid_ref_event_idx, input_intervals, input_samples = self.create_TEKG_in_batch(batch_idx_ls, mode)

            ts_probs_first_event, ts_probs_last_event, ts_probs_first_event_inv_rel, ts_probs_last_event_inv_rel = probs

            output = []
            for x in [qq, query_rels, refNode_source, res_random_walk, ts_probs_first_event, ts_probs_last_event, ts_probs_first_event_inv_rel, ts_probs_last_event_inv_rel,
                        valid_sample_idx, input_intervals, input_samples]:
                output.append(np.array(x).tolist())

            with open('output/' + dataset + '/' + dataset + '_' + mode + "_batch_input_" + str(i) +".json", "w") as json_file:
                # Write the dictionary to the file in JSON format
                json.dump(output, json_file)



    def create_TEKG_in_batch(self, idx_ls, mode):
        dataset = self.data['dataset']
        dataset_name = self.data['dataset_name']

        num_samples_dist = self.data['num_samples_dist']
        timestamp_range = self.data['timestamp_range']

        num_rel = self.data['num_rel']
        num_entity = self.data['num_entity']

        prob_type_for_training = self.option.prob_type_for_training
        num_step = self.option.num_step-1
        flag_ruleLen_split_ver = self.option.flag_ruleLen_split_ver
        flag_acceleration = self.option.flag_acceleration
        num_rule = self.option.num_rule

        mdb = self.data['mdb']
        connectivity = self.data['connectivity']
        TEKG_nodes = self.data['TEKG_nodes']
        file_suffix = ''

        if self.data['dataset_name'] in ['wiki', 'YAGO']:
            path = self.data['path']
            edges = np.vstack((self.data['train_edges'], self.data['test_edges']))
            pattern_ls = self.data['pattern_ls']
            ts_stat_ls = self.data['ts_stat_ls']
            te_stat_ls = self.data['te_stat_ls']
            rm_ls = self.data['rm_ls']
            output_probs_with_ref_edges = self.data['random_walk_res']
        else:
            edges = np.vstack((self.data['train_edges'], self.data['valid_edges'], self.data['test_edges']))
            if self.data['random_walk_res'] is not None:
                input_edge_probs = self.data['random_walk_res']


        if flag_acceleration:
            if self.data['dataset_name'] in ['wiki', 'YAGO']:
                query_edges = []
                input_intervals = []
                for data_idx in idx_ls:
                    file = dataset_name + '_train_query_' + str(data_idx) + '.json'
                    with open(path + file, 'r') as f:
                        json_data = json.loads(f.read())
                    cur_query = json_data['query']
                    query_edges.append(cur_query)
                    input_intervals.append(cur_query[3:])

                qq = [query[1] for query in query_edges]

                if output_probs_with_ref_edges is None:
                    output = create_inputs_v3(edges=edges, path=path, 
                                                 dataset=dataset_name, 
                                                 idx_ls=idx_ls,
                                                 pattern_ls=pattern_ls, 
                                                 timestamp_range=timestamp_range, 
                                                 num_rel=num_rel//2,
                                                 ts_stat_ls=ts_stat_ls, 
                                                 te_stat_ls=te_stat_ls,
                                                 with_ref_end_time=True,
                                                 targ_rel=None, num_samples_dist=num_samples_dist, 
                                                 mode=mode, rm_ls=rm_ls,
                                                 flag_output_probs_with_ref_edges=True,
                                                 flag_acceleration=flag_acceleration)


                if output_probs_with_ref_edges is None:
                    output_probs_with_ref_edges = output[-1]
                    if mode == 'Test':
                        input_intervals_dict = output[-2]
                else:
                    if mode == 'Test':
                        input_intervals_dict = {}
                        for data_idx in idx_ls:
                            data_idx -= num_samples_dist[1]
                            cur_interval = edges[data_idx, 3:]
                            input_intervals_dict[data_idx] = cur_interval

                input_samples = []

            else:
                input_samples = edges[idx_ls]
                input_intervals = edges[idx_ls, 3]
                qq = edges[idx_ls, 1]

                # print(input_samples, input_intervals, qq)
                # sys.exit()


            dim = 1
            if mode == 'Test':
                dim = len(timestamp_range)


            if self.data['dataset_name'] in ['wiki', 'YAGO']:
                if flag_ruleLen_split_ver:
                    ruleLen_embedding = {}
                    for rel in pattern_ls:
                        ruleLen = np.array([len(rule.split(' '))//2 for rule in pattern_ls[rel]])
                        ruleLen = np.hstack((ruleLen, np.zeros((num_rule - len(pattern_ls[rel]),))))
                        ruleLen_embedding[int(rel)] = ruleLen.copy()


                valid_sample_idx = []
                query_rels = []
                refNode_source = []
                res_random_walk = []

                ts_probs_last_event_ts = []
                ts_probs_last_event_te = []
                ts_probs_first_event_ts = []
                ts_probs_first_event_te = []

                te_probs_last_event_ts = []
                te_probs_last_event_te = []
                te_probs_first_event_ts = []
                te_probs_first_event_te = []


                for (i, data_idx) in enumerate(idx_ls):
                    flag_valid = 0
                    if mode == 'Test':
                        data_idx -= num_samples_dist[1]

                        if data_idx in input_intervals_dict:
                            input_intervals[i] = input_intervals_dict[data_idx].tolist()

                    if data_idx not in output_probs_with_ref_edges:
                        continue

                    refNode_probs = {}
                    refNode_res_rw = {}

                    for idx_first_or_last in [0,1]:
                        event_type = ['first_event', 'last_event'][idx_first_or_last]

                        for edge in output_probs_with_ref_edges[data_idx][event_type]:
                            if edge not in refNode_probs:
                                refNode_probs[edge] = 1./len(timestamp_range) * np.ones((num_rule, 2, 2, 2, dim))
                                refNode_res_rw[edge] = np.zeros((num_rule,))

                            for idx_query_ts_or_te in [0,1]:
                                for idx_ref_ts_or_te in [0,1]:
                                    x = output_probs_with_ref_edges[data_idx][event_type][edge][idx_query_ts_or_te][idx_ref_ts_or_te]

                                    if len(x)>0:
                                        flag_valid = 1
                                    for x1 in x:
                                        refNode_probs[edge][x1[1], idx_first_or_last, idx_query_ts_or_te, idx_ref_ts_or_te, :] = x1[0]
                                        refNode_res_rw[edge][x1[1]] = 1

                        # print('------------------------')


                    if flag_valid:
                        num_valid_edge = 0
                        for edge in refNode_res_rw:
                            if len(refNode_res_rw[edge][refNode_res_rw[edge] == 1]) == 0:
                                continue

                            num_valid_edge += 1
                            res_random_walk.append(refNode_res_rw[edge])

                            # print(refNode_probs[edge][refNode_res_rw[edge] == 1])

                            if flag_ruleLen_split_ver:
                                x = [np.mean(refNode_probs[edge][(refNode_res_rw[edge] == 1) & (ruleLen_embedding[qq[i]] == l)], axis=0) for l in range(num_step-1)]
                            else:
                                x = np.mean(refNode_probs[edge][refNode_res_rw[edge] == 1], axis=0)

                            if mode == 'Train':
                                if prob_type_for_training == 'max':
                                    if flag_ruleLen_split_ver:
                                        x = [np.max(refNode_probs[edge][(refNode_res_rw[edge] == 1) & (ruleLen_embedding[qq[i]] == l)], axis=0) for l in range(num_step-1)]
                                    else:
                                        x = np.max(refNode_probs[edge][refNode_res_rw[edge] == 1], axis=0)

                            if flag_ruleLen_split_ver:
                                ts_probs_last_event_ts.append([x[l][1, 0, 0, :] for l in range(num_step-1)])
                                ts_probs_last_event_te.append([x[l][1, 0, 1, :] for l in range(num_step-1)])
                                ts_probs_first_event_ts.append([x[l][0, 0, 0, :] for l in range(num_step-1)])
                                ts_probs_first_event_te.append([x[l][0, 0, 1, :] for l in range(num_step-1)])

                                te_probs_last_event_ts.append([x[l][1, 1, 0, :] for l in range(num_step-1)])
                                te_probs_last_event_te.append([x[l][1, 1, 1, :] for l in range(num_step-1)])
                                te_probs_first_event_ts.append([x[l][0, 1, 0, :] for l in range(num_step-1)])
                                te_probs_first_event_te.append([x[l][0, 1, 1, :] for l in range(num_step-1)])
                            else:
                                ts_probs_last_event_ts.append(x[1, 0, 0, :])
                                ts_probs_last_event_te.append(x[1, 0, 1, :])
                                ts_probs_first_event_ts.append(x[0, 0, 0, :])
                                ts_probs_first_event_te.append(x[0, 0, 1, :])

                                te_probs_last_event_ts.append(x[1, 1, 0, :])
                                te_probs_last_event_te.append(x[1, 1, 1, :])
                                te_probs_first_event_ts.append(x[0, 1, 0, :])
                                te_probs_first_event_te.append(x[0, 1, 1, :])


                        if num_valid_edge>0:
                            valid_sample_idx.append(i)
                            refNode_source.append(num_valid_edge)
                            query_rels += [qq[i]] * num_valid_edge

                # print(refNode_source)
                # print(len(res_random_walk))

                refNode_num = 0
                for i in range(len(refNode_source)):
                    # print(refNode_num)
                    x = np.zeros((len(res_random_walk), ))
                    x[refNode_num: refNode_num + refNode_source[i]] = 1
                    refNode_num += refNode_source[i]
                    refNode_source[i] = x.copy()


                probs = [ts_probs_first_event_ts, ts_probs_first_event_te, ts_probs_last_event_ts, ts_probs_last_event_te,
                         te_probs_first_event_ts, te_probs_first_event_te, te_probs_last_event_ts, te_probs_last_event_te ]

            else:
                valid_sample_idx = []
                query_rels = []
                refNode_source = []
                res_random_walk = []

                ts_probs_first_event = []
                ts_probs_last_event = []
                ts_probs_first_event_inv_rel = []
                ts_probs_last_event_inv_rel = []


                for (i, sample) in enumerate(input_samples):
                    flag_valid = 0
                    refNode_probs = {}
                    refNode_res_rw = {}

                    if self.data['random_walk_res'] is None:
                        cur_random_walk_path = "output/"+ dataset + '/' + dataset +"_" + mode +"_input_edge_probs_edge_"+ str(sample.tolist()) + file_suffix + ".json"
                        # print(cur_random_walk_path)
                        if not os.path.exists(cur_random_walk_path):
                            continue

                        # Open a file in write mode
                        with open(cur_random_walk_path, "r") as json_file:
                            # Write the dictionary to the file in JSON format
                            input_edge_probs1 = json.load(json_file)

                        input_edge_probs = []
                        for j_k in ['input_edge_probs_first_ref', 'input_edge_probs_last_ref', 'input_edge_probs_first_ref_inv_rel', 'input_edge_probs_last_ref_inv_rel']:
                            input_edge_probs.append({tuple(sample): inv_convert_dict(input_edge_probs1[j_k])})


                    for j in range(4):
                        if tuple(sample) not in input_edge_probs[j]:
                            continue

                        for edge in input_edge_probs[j][tuple(sample)]:
                            if edge not in refNode_probs:
                                refNode_probs[edge] = 1./len(timestamp_range) * np.ones((num_rule, 4, dim))
                                refNode_res_rw[edge] = np.zeros((num_rule,))

                                x = input_edge_probs[j][tuple(sample)][edge]
                                if len(x)>0:
                                    flag_valid = 1
                                for x1 in x:
                                    refNode_probs[edge][x1[1], j, :] = x1[0]
                                    refNode_res_rw[edge][x1[1]] = 1

                    if flag_valid:
                        num_valid_edge = 0
                        for edge in refNode_res_rw:
                            if len(refNode_res_rw[edge][refNode_res_rw[edge] == 1]) == 0:
                                continue

                            num_valid_edge += 1
                            res_random_walk.append(refNode_res_rw[edge])


                            x = np.mean(refNode_probs[edge][refNode_res_rw[edge] == 1], axis=0)

                            if mode == 'Train':
                                if prob_type_for_training == 'max':
                                    x = np.max(refNode_probs[edge][refNode_res_rw[edge] == 1], axis=0)


                            ts_probs_first_event.append(x[0, :])
                            ts_probs_last_event.append(x[1, :])
                            ts_probs_first_event_inv_rel.append(x[2, :])
                            ts_probs_last_event_inv_rel.append(x[3, :])


                        if num_valid_edge>0:
                            valid_sample_idx.append(i)
                            refNode_source.append(num_valid_edge)
                            query_rels += [qq[i]] * num_valid_edge


                refNode_num = 0
                for i in range(len(refNode_source)):
                    x = np.zeros((len(res_random_walk), ))
                    x[refNode_num: refNode_num + refNode_source[i]] = 1
                    refNode_num += refNode_source[i]
                    refNode_source[i] = x.copy()


                probs = [ts_probs_first_event, ts_probs_last_event, ts_probs_first_event_inv_rel, ts_probs_last_event_inv_rel]


            return qq, query_rels, refNode_source, res_random_walk, probs, valid_sample_idx, input_intervals, input_samples




        flag_use_batch_graph = False
        if (mdb is None) or (connectivity is None) or (TEKG_nodes is None):
            flag_use_batch_graph = True


        query_edges = []
        input_intervals = []

        if flag_use_batch_graph:
            batch_edges = []

        for data_idx in idx_ls:
            file = dataset_name + '_train_query_' + str(data_idx) + '.json'
            with open(path + file, 'r') as f:
                json_data = json.loads(f.read())

            cur_query = json_data['query']
            query_edges.append(cur_query)
            input_intervals.append(cur_query[3:])

            if flag_use_batch_graph:
                batch_edges.append(cur_query)

                for ruleLen in range(1,6):
                    if str(ruleLen) not in json_data:
                        continue
                    for walk in json_data[str(ruleLen)]:
                        for i in range(ruleLen):
                            x = walk[4*i:4*i+5]
                            batch_edges.append([x[j] for j in [0,1,4,2,3]])


        if flag_use_batch_graph:
            batch_edges = np.array(batch_edges)
            batch_edges = np.unique(batch_edges, axis=0)

            batch_edges_inv = batch_edges[:, [2,1,0,3,4]]
            batch_edges_inv[batch_edges[:,1] < num_rel//2, 1] += num_rel//2
            batch_edges_inv[batch_edges[:,1] >= num_rel//2, 1] -= num_rel//2

            # print(batch_edges)
            # print(batch_edges_inv)

            batch_edges = np.vstack((batch_edges, batch_edges_inv))
            batch_edges = np.unique(batch_edges, axis=0)
            # batch_edges_inv = batch_edges[batch_edges[:,1] >= num_rel//2]
            batch_edges = batch_edges[batch_edges[:,1] < num_rel//2]
            batch_edges_ori = batch_edges.copy()
            # print(batch_edges_ori)
            # print(batch_edges_inv)
            # sys.exit()

            assert len(batch_edges_ori) <= num_entity//2, 'You should increase num_entity or reduce batch_edges.'

            batch_edges_idx = np.arange(len(batch_edges))


            mdb = {}
            for r in range(num_rel//2):
                idx_cur_rel = batch_edges_idx[batch_edges[:, 1] == r].reshape((-1,1))
                if len(idx_cur_rel) == 0:
                    mdb[r] = [[[0,0]], [0.0], [num_entity, num_entity]]
                    mdb[r + num_rel//2] = [[[0,0]], [0.0], [num_entity, num_entity]]
                else:
                    mdb[r] = [np.hstack([idx_cur_rel, idx_cur_rel]).tolist(), [1.0]*len(idx_cur_rel), [num_entity, num_entity]]
                    mdb[r + num_rel//2] = [np.hstack([idx_cur_rel + num_entity//2, idx_cur_rel + num_entity//2]).tolist(), 
                                                [1.0]*len(idx_cur_rel), [num_entity, num_entity]]
                # print(r, mdb[r])
                # print(r + num_rel//2, mdb[r + num_rel//2])

            # sys.exit()


            batch_edges_inv = batch_edges[:, [2,1,0,3,4]]
            batch_edges_inv[:,1] += num_rel//2
            batch_edges = np.vstack((batch_edges, batch_edges_inv))
            batch_edges_idx_cmp = np.hstack((batch_edges_idx, batch_edges_idx + num_entity//2))


            connectivity = {}  # [x,y] means y to x
            connectivity[0] = []
            connectivity[1] = []
            connectivity[2] = []
            connectivity[3] = []


            for ent in np.unique(np.hstack([batch_edges[:, 0], batch_edges[:, 2]])):
                b = batch_edges_idx_cmp[batch_edges[:, 2] == ent]
                a = batch_edges_idx_cmp[batch_edges[:, 0] == ent]
                combinations = list(itertools.product(a, b))
                combinations = np.array(combinations)

                combinations_TR = combinations.copy()
                combinations_TR[combinations_TR>=num_entity//2] -= num_entity//2
                TRs = calculate_TR_mat_ver(batch_edges[combinations_TR[:,0], 3:], batch_edges[combinations_TR[:,1], 3:])

                # print(combinations)
                # print(combinations_TR)

                connectivity[0] += combinations.tolist()
                connectivity[1] += combinations[TRs==1].tolist()
                connectivity[2] += combinations[TRs==2].tolist()
                connectivity[3] += combinations[TRs==3].tolist()


            for TR in range(4):
                if len(connectivity[TR])>0:
                    A = np.array(connectivity[TR])
                    B = A.copy()
                    B[B>=num_entity//2] -= num_entity//2
                    A = A[~(B[:,0] == B[:,1])]
                    if len(A) == 0:
                        connectivity[TR] = [[[0,0]], [0.0], [num_entity, num_entity]]
                    else:
                        A = np.unique(A, axis=0)

                        connectivity[TR] = [A.tolist()]
                        connectivity[TR].append([1.0] * len(connectivity[TR][0]))
                        connectivity[TR].append([num_entity, num_entity])
                else:
                    connectivity[TR] = [[[0,0]], [0.0], [num_entity, num_entity]]

                # print(TR, connectivity[TR])

            # sys.exit()


        if output_probs_with_ref_edges is None:
            if flag_ruleLen_split_ver:
                output = create_inputs_v3_ruleLen_split_ver(edges=edges, path=path, 
                                                             dataset=dataset_name, 
                                                             idx_ls=idx_ls,
                                                             pattern_ls=pattern_ls, 
                                                             timestamp_range=timestamp_range, 
                                                             num_rel=num_rel//2,
                                                             ts_stat_ls=ts_stat_ls, 
                                                             te_stat_ls=te_stat_ls,
                                                             with_ref_end_time=True,
                                                             targ_rel=None, num_samples_dist=num_samples_dist, 
                                                             mode=mode, rm_ls=rm_ls,
                                                             flag_output_probs_with_ref_edges=True)
            else:
                output = create_inputs_v3(edges=edges, path=path, 
                                             dataset=dataset_name, 
                                             idx_ls=idx_ls,
                                             pattern_ls=pattern_ls, 
                                             timestamp_range=timestamp_range, 
                                             num_rel=num_rel//2,
                                             ts_stat_ls=ts_stat_ls, 
                                             te_stat_ls=te_stat_ls,
                                             with_ref_end_time=True,
                                             targ_rel=None, num_samples_dist=num_samples_dist, 
                                             mode=mode, rm_ls=rm_ls,
                                             flag_output_probs_with_ref_edges=True)


        if mode == 'Test':
            input_intervals_dict = output[-2]

        if output_probs_with_ref_edges is None:
            output_probs_with_ref_edges = output[-1]

        # print(output_probs_with_ref_edges.keys())


        qq = [query[1] for query in query_edges]
        if flag_use_batch_graph:
            hh = [batch_edges_ori.tolist().index(query) for query in query_edges]
        else:
            hh = [TEKG_nodes.tolist().index(query) for query in query_edges]

        tt = [h + num_entity//2 for h in hh]


        if mode == 'Train':
            if flag_ruleLen_split_ver:
                ts_probs_first_event = [(np.ones((len(idx_ls), num_entity, num_step)) * 1.0/len(timestamp_range)).tolist()] *2
                te_probs_first_event = [(np.ones((len(idx_ls), num_entity, num_step)) * 1.0/len(timestamp_range)).tolist()] *2
                ts_probs_last_event = [(np.ones((len(idx_ls), num_entity, num_step)) * 1.0/len(timestamp_range)).tolist()] *2
                te_probs_last_event = [(np.ones((len(idx_ls), num_entity, num_step)) * 1.0/len(timestamp_range)).tolist()] *2
            else:
                ts_probs_first_event = [[[1.0/len(timestamp_range)]*num_entity]* len(idx_ls), 
                                        [[1.0/len(timestamp_range)]*num_entity]* len(idx_ls)]
                te_probs_first_event = [[[1.0/len(timestamp_range)]*num_entity]* len(idx_ls), 
                                        [[1.0/len(timestamp_range)]*num_entity]* len(idx_ls)]

                ts_probs_last_event = [[[1.0/len(timestamp_range)]*num_entity]* len(idx_ls), 
                                        [[1.0/len(timestamp_range)]*num_entity]* len(idx_ls)]
                te_probs_last_event = [[[1.0/len(timestamp_range)]*num_entity]* len(idx_ls), 
                                        [[1.0/len(timestamp_range)]*num_entity]* len(idx_ls)]
        elif mode == 'Test':
            ts_probs_first_event = [[[1.0/len(timestamp_range) * np.ones((len(timestamp_range),))]*num_entity]* len(idx_ls), 
                                    [[1.0/len(timestamp_range) * np.ones((len(timestamp_range),))]*num_entity]* len(idx_ls)]
            te_probs_first_event = [[[1.0/len(timestamp_range) * np.ones((len(timestamp_range),))]*num_entity]* len(idx_ls), 
                                    [[1.0/len(timestamp_range) * np.ones((len(timestamp_range),))]*num_entity]* len(idx_ls)]

            ts_probs_last_event = [[[1.0/len(timestamp_range) * np.ones((len(timestamp_range),))]*num_entity]* len(idx_ls), 
                                    [[1.0/len(timestamp_range) * np.ones((len(timestamp_range),))]*num_entity]* len(idx_ls)]
            te_probs_last_event = [[[1.0/len(timestamp_range) * np.ones((len(timestamp_range),))]*num_entity]* len(idx_ls), 
                                    [[1.0/len(timestamp_range) * np.ones((len(timestamp_range),))]*num_entity]* len(idx_ls)]

        valid_sample_idx = []
        valid_ref_event_idx_ts_first = []
        valid_ref_event_idx_ts_last = []
        valid_ref_event_idx_te_first = []
        valid_ref_event_idx_te_last = []

        for (i, data_idx) in enumerate(idx_ls):
            flag_valid = 0
            if mode == 'Test':
                data_idx -= num_samples_dist[1]

                if data_idx in input_intervals_dict:
                    input_intervals[i] = input_intervals_dict[data_idx].tolist()

            # print(data_idx)

            if data_idx not in output_probs_with_ref_edges:
                continue

            ref_event_idx_ts_first = np.zeros((num_entity,))
            ref_event_idx_ts_last = np.zeros((num_entity,))
            ref_event_idx_te_first = np.zeros((num_entity,))
            ref_event_idx_te_last = np.zeros((num_entity,))

            for edge in output_probs_with_ref_edges[data_idx]['first_event']:
                if flag_use_batch_graph:
                    cur_edge_idx = batch_edges_idx_cmp[batch_edges.tolist().index([edge[j] for j in [0,1,4,2,3]])]
                else:
                    cur_edge_idx = TEKG_nodes.tolist().index([edge[j] for j in [0,1,4,2,3]])
                # print(cur_edge_idx)

                for j in [0,1]:
                    if flag_ruleLen_split_ver:
                        for l in range(num_step):
                            if len(output_probs_with_ref_edges[data_idx]['first_event'][edge][0][j][l+1])>0:
                                if mode == 'Train':
                                    if prob_type_for_training == 'max':
                                        ts_probs_first_event[j][i][cur_edge_idx][l] = max(output_probs_with_ref_edges[data_idx]['first_event'][edge][0][j][l+1])
                                    else:
                                        ts_probs_first_event[j][i][cur_edge_idx][l] = np.mean(output_probs_with_ref_edges[data_idx]['first_event'][edge][0][j][l+1])
                                elif mode == 'Test':
                                    ts_probs_first_event[j][i][cur_edge_idx] = np.mean(output_probs_with_ref_edges[data_idx]['first_event'][edge][0][j], axis=0)
                                flag_valid = 1
                                ref_event_idx_ts_first[cur_edge_idx] = 1
                                # print(i,j,l, ts_probs_first_event[j][i][cur_edge_idx][l])

                            if len(output_probs_with_ref_edges[data_idx]['first_event'][edge][1][j][l+1])>0:
                                if mode == 'Train':
                                    if prob_type_for_training == 'max':
                                        te_probs_first_event[j][i][cur_edge_idx][l] = max(output_probs_with_ref_edges[data_idx]['first_event'][edge][1][j][l+1])
                                    else:
                                        te_probs_first_event[j][i][cur_edge_idx][l] = np.mean(output_probs_with_ref_edges[data_idx]['first_event'][edge][1][j][l+1])
                                elif mode == 'Test':
                                    te_probs_first_event[j][i][cur_edge_idx] = np.mean(output_probs_with_ref_edges[data_idx]['first_event'][edge][1][j], axis=0)
                                flag_valid = 1
                                ref_event_idx_te_first[cur_edge_idx] = 1
                                # print(i,j,l, te_probs_first_event[j][i][cur_edge_idx][l])
                    else:
                        if len(output_probs_with_ref_edges[data_idx]['first_event'][edge][0][j])>0:
                            if mode == 'Train':
                                if prob_type_for_training == 'max':
                                    ts_probs_first_event[j][i][cur_edge_idx] = max(output_probs_with_ref_edges[data_idx]['first_event'][edge][0][j])
                                else:
                                    ts_probs_first_event[j][i][cur_edge_idx] = np.mean(output_probs_with_ref_edges[data_idx]['first_event'][edge][0][j])
                            elif mode == 'Test':
                                ts_probs_first_event[j][i][cur_edge_idx] = np.mean(output_probs_with_ref_edges[data_idx]['first_event'][edge][0][j], axis=0)
                            flag_valid = 1
                            ref_event_idx_ts_first[cur_edge_idx] = 1
                            # print(i,j, ts_probs_first_event[j][i][cur_edge_idx])


                        if len(output_probs_with_ref_edges[data_idx]['first_event'][edge][1][j])>0:
                            if mode == 'Train':
                                if prob_type_for_training == 'max':
                                    te_probs_first_event[j][i][cur_edge_idx] = max(output_probs_with_ref_edges[data_idx]['first_event'][edge][1][j])
                                else:
                                    te_probs_first_event[j][i][cur_edge_idx] = np.mean(output_probs_with_ref_edges[data_idx]['first_event'][edge][1][j])
                            elif mode == 'Test':
                                te_probs_first_event[j][i][cur_edge_idx] = np.mean(output_probs_with_ref_edges[data_idx]['first_event'][edge][1][j], axis=0)
                            flag_valid = 1
                            ref_event_idx_te_first[cur_edge_idx] = 1
                            # print(i,j, te_probs_first_event[j][i][cur_edge_idx])

            # print('------------------------')

            for edge in output_probs_with_ref_edges[data_idx]['last_event']:
                if flag_use_batch_graph:
                    cur_edge_idx = batch_edges_idx_cmp[batch_edges.tolist().index([edge[j] for j in [0,1,4,2,3]])]
                else:
                    cur_edge_idx = TEKG_nodes.tolist().index([edge[j] for j in [0,1,4,2,3]])
                # print(cur_edge_idx)

                for j in [0,1]:
                    if flag_ruleLen_split_ver:
                        for l in range(num_step):
                            if len(output_probs_with_ref_edges[data_idx]['last_event'][edge][0][j][l+1])>0:
                                if mode == 'Train':
                                    if prob_type_for_training == 'max':
                                        ts_probs_last_event[j][i][cur_edge_idx][l] = max(output_probs_with_ref_edges[data_idx]['last_event'][edge][0][j][l+1])
                                    else:
                                        ts_probs_last_event[j][i][cur_edge_idx][l] = np.mean(output_probs_with_ref_edges[data_idx]['last_event'][edge][0][j][l+1])
                                elif mode == 'Test':
                                    ts_probs_last_event[j][i][cur_edge_idx] = np.mean(output_probs_with_ref_edges[data_idx]['last_event'][edge][0][j], axis=0)
                                flag_valid = 1
                                ref_event_idx_ts_last[cur_edge_idx] = 1
                                # print(i,j,l, ts_probs_last_event[j][i][cur_edge_idx][l])

                            if len(output_probs_with_ref_edges[data_idx]['last_event'][edge][1][j][l+1])>0:
                                if mode == 'Train':
                                    if prob_type_for_training == 'max':
                                        te_probs_last_event[j][i][cur_edge_idx][l] = max(output_probs_with_ref_edges[data_idx]['last_event'][edge][1][j][l+1])
                                    else:
                                        te_probs_last_event[j][i][cur_edge_idx][l] = np.mean(output_probs_with_ref_edges[data_idx]['last_event'][edge][1][j][l+1])
                                elif mode == 'Test':
                                    te_probs_last_event[j][i][cur_edge_idx] = np.mean(output_probs_with_ref_edges[data_idx]['last_event'][edge][1][j], axis=0)
                                flag_valid = 1
                                ref_event_idx_te_last[cur_edge_idx] = 1
                                # print(i,j,l, te_probs_last_event[j][i][cur_edge_idx][l])

                    else:
                        if len(output_probs_with_ref_edges[data_idx]['last_event'][edge][0][j])>0:
                            if mode == 'Train':
                                if prob_type_for_training == 'max':
                                    ts_probs_last_event[j][i][cur_edge_idx] = max(output_probs_with_ref_edges[data_idx]['last_event'][edge][0][j])
                                else:
                                    ts_probs_last_event[j][i][cur_edge_idx] = np.mean(output_probs_with_ref_edges[data_idx]['last_event'][edge][0][j])
                            elif mode == 'Test':
                                ts_probs_last_event[j][i][cur_edge_idx] = np.mean(output_probs_with_ref_edges[data_idx]['last_event'][edge][0][j], axis=0)
                            flag_valid = 1
                            ref_event_idx_ts_last[cur_edge_idx] = 1
                            # print(i,j, ts_probs_last_event[j][i][cur_edge_idx])

                        if len(output_probs_with_ref_edges[data_idx]['last_event'][edge][1][j])>0:
                            if mode == 'Train':
                                if prob_type_for_training == 'max':
                                    te_probs_last_event[j][i][cur_edge_idx] = max(output_probs_with_ref_edges[data_idx]['last_event'][edge][1][j])
                                else:
                                    te_probs_last_event[j][i][cur_edge_idx] = np.mean(output_probs_with_ref_edges[data_idx]['last_event'][edge][1][j])
                            elif mode == 'Test':
                                te_probs_last_event[j][i][cur_edge_idx] = np.mean(output_probs_with_ref_edges[data_idx]['last_event'][edge][1][j], axis=0)
                            flag_valid = 1
                            ref_event_idx_te_last[cur_edge_idx] = 1
                            # print(i,j, te_probs_last_event[j][i][cur_edge_idx])

            # print('------------------------')

            if flag_valid:
                valid_sample_idx.append(i)
                valid_ref_event_idx_ts_first.append(ref_event_idx_ts_first)
                valid_ref_event_idx_ts_last.append(ref_event_idx_ts_last)
                valid_ref_event_idx_te_first.append(ref_event_idx_te_first)
                valid_ref_event_idx_te_last.append(ref_event_idx_te_last)


        # print(hh, tt, qq)
        # print(ts_probs_first_event)
        # print(output_probs_with_ref_edges)
        # print(valid_sample_idx)

        probs = [ts_probs_first_event, ts_probs_last_event, te_probs_first_event, te_probs_last_event]
        valid_ref_event_idx = [np.array(valid_ref_event_idx_ts_first), np.array(valid_ref_event_idx_ts_last), 
                               np.array(valid_ref_event_idx_te_first), np.array(valid_ref_event_idx_te_last)]

        return qq, hh, tt, mdb, connectivity, probs, valid_sample_idx, valid_ref_event_idx, input_intervals



def prepare_dataloader(option, data, mode, idx_ls, idx_batch_ls):
    myDataLoader = DataLoader(option, data)
    myDataLoader.one_epoch('Test', idx_ls, idx_batch_ls)



def process_random_walk_with_sampling_results(dataset, mode, num_rel, file_paths, file_suffix, num_workers=24, flag_time_shifting=False):
    index_pieces = split_list_into_pieces(range(num_rel*2), num_workers)
    results = Parallel(n_jobs=num_workers)(delayed(read_random_walk_results)(dataset, piece, file_paths, file_suffix, flag_interval=False, 
                                                flag_plot=False, mode=mode, flag_time_shifting=flag_time_shifting) for piece in index_pieces)

    # for rel in range(num_rel*2):
    #     with open('../output/' + dataset + '/' + dataset + file_suffix + 'samples_rel_' + str(rel) + '_' + mode + ".json", 'r') as f:
    #         samples = json.load(f)

    #     rule_nums = rule_num_stat(samples)

    #     pattern_ls = {}
       
    #     if rel not in rule_nums:
    #         pattern_ls = []

    #     else:
    #         pattern_ls = sorted(rule_nums[rel], key=lambda p: rule_nums[rel][p], reverse=True)[:1000]
    #         random.shuffle(pattern_ls)

    #     with open('../output/' + dataset + '/' + dataset + file_suffix + "pattern_ls_rel_" + str(rel) + ".json", 'w') as file:
    #         json.dump(pattern_ls, file)