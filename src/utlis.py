import os
import numpy as np
import copy
import random
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
from datetime import datetime


def str_tuple(e):
    return str(tuple(e))


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


def my_shuffle(data, seed=None):
    if seed is not None:
        np.random.seed(seed)

    idx = list(range(len(data)))
    random.shuffle(idx)
    new_data = data[idx]
    return new_data


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



def merge_list(ls_ls):
    output = []
    for ls in ls_ls:
        output += ls
    return output



def split_list_into_batches(lst, batch_size=None, num_batches=None):
    """
    Splits a list into batches (either of a given batch size or of a given number of batches).
    """
    assert (batch_size is not None) or (num_batches is not None), "Either batch_size or num_pieces must be provided."
    if num_batches is not None:
        indices_per_piece = len(lst) // num_batches
        index_pieces = [lst[i:i + indices_per_piece] for i in range(0, len(lst), indices_per_piece)]
        if len(index_pieces)>num_batches:
            output = index_pieces[:num_batches-1] + [merge_list(index_pieces[num_batches-1:])]
        else:
            output = index_pieces
        return output
    
    return [lst[i:i + batch_size] for i in range(0, len(lst), batch_size)]





def obtain_walk_file_ls(path, dataset, idx_ls=None,  ratio=None, imbalanced_rel=None,  flag_biased=0, exp_idx=None):
    rel = None
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
    return file_ls, rel


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


def cal_timegap(timestamp1, timestamp2, notation_invalid=9999):
    if timestamp1 == 9999 or timestamp2 == 9999:
        return notation_invalid
    else:
        return timestamp1 - timestamp2


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


def rule_num_stat(data):
    res = {}
    for e in data:
        e = str(e) if not isinstance(e, str) else e
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


def obtain_inv_edge(edges, num_rel):
    return np.hstack([edges[:,2:3], edges[:,1:2] + num_rel, edges[:,0:1], edges[:,3:]])


def save_data(filename, line):
    # Get the current date and time
    current_datetime = datetime.now()

    # Open the file in append mode
    with open(filename, "a") as file:
        file.write(str(current_datetime) + ' ' + str(line) + "\n")