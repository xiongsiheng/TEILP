import sys
import argparse
import numpy as np
import os
import json

from dataset_setting import obtain_dataset
from do_random_walk import random_walk
from utils import convert_walks_into_rules


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str)
parser.add_argument('--shift', default=0, type=int)
parser.add_argument('--few', default=0, type=int)
parser.add_argument('--biased', default=0, type=int)
parser.add_argument('--ratio', type=float)
parser.add_argument('--imbalanced_rel', type=int)
parser.add_argument('--targ_rel', type=int)
parser.add_argument('--exp_idx', type=int)


args = parser.parse_args()


dataset_using = args.dataset
targ_rel = args.targ_rel
imbalanced_rel = args.imbalanced_rel
ratio = args.ratio
exp_idx = args.exp_idx

flag_time_shift = args.shift
flag_biased = args.biased
flag_few_training = args.few




if flag_biased + flag_time_shift + flag_few_training > 1:
    print('Only one difficult setting is allowed.')
    sys.exit()



num_pattern = 3
const_pattern_ls = [-1, 0, 1]
f_interval = 1


if dataset_using == 'wiki':
    dataset_name = 'WIKIDATA12k'
    num_rel = 48
    num_entites = 12554
    num_ruleLen = 3

elif dataset_using == 'YAGO':
    dataset_name = 'YAGO11k'
    num_rel = 20
    num_entites = 10623
    num_ruleLen = 5


if flag_biased:
    dataset_name1 = 'difficult_settings/' + dataset_name + '_rel_' + str(imbalanced_rel) + '_exp_' + str(exp_idx)
    output_path = 'walk_res_biased'
elif flag_few_training:
    dataset_name1 = 'difficult_settings/' + dataset_name + '_ratio_' + str(ratio) + '_exp_' + str(exp_idx)
    output_path = 'walk_res_few'
elif flag_time_shift:
    dataset_name1 = 'difficult_settings/' + dataset_name + '_time_shifting'
    output_path = 'walk_res_time_shift'
else:
    dataset_name1 = dataset_name
    output_path = 'walk_res'



para_ls_for_walker = [num_rel, num_pattern, num_ruleLen, dataset_using, f_interval]
targ_rel_ls = range(num_rel)
if imbalanced_rel is not None:
    targ_rel_ls = [imbalanced_rel]
if targ_rel is not None:
    targ_rel_ls = [targ_rel]




train_edges, valid_data, valid_data_inv, test_data, test_data_inv = obtain_dataset(dataset_name1, num_rel)


# mask valid / test data
if not flag_time_shift:
    if f_interval:
        valid_data[:, [3,4]] = 9999
        valid_data_inv[:, [3,4]] = 9999
        test_data[:, [3,4]] = 9999
        test_data_inv[:, [3,4]] = 9999
    else:
        valid_data[:, 3] = 9999
        valid_data_inv[:, 3] = 9999
        test_data[:, 3] = 9999
        test_data_inv[:, 3] = 9999

num_train = len(train_edges)//2
num_valid = len(valid_data)
num_test = len(test_data)


edges = np.vstack([train_edges[:len(train_edges)//2], valid_data, test_data,
                        train_edges[len(train_edges)//2:], valid_data_inv, test_data_inv])


pos_examples_idx = None

if flag_few_training:
    pos_examples_idx = []
    with open('../'+ dataset_name + '/' + dataset_name +'_ratio_' + str(ratio) +'_exp_'+ str(exp_idx) + '_training_idx.json', 'r') as file:
        pos_examples_idx += json.load(file)
    pos_examples_idx += list(range(num_train, num_train + num_valid + num_test))
    pos_examples_idx += [idx + num_train + num_valid + num_test for idx in pos_examples_idx]


random_walk(targ_rel_ls, edges, para_ls_for_walker, pos_examples_idx = pos_examples_idx, time_shift_mode=flag_time_shift, output_path=output_path,
                         ratio=ratio, imbalanced_rel=imbalanced_rel, exp_idx=exp_idx)


output = convert_walks_into_rules(dataset=dataset_using, path='../output/' + output_path, idx_ls=pos_examples_idx, 
                                      flag_time_shift=flag_time_shift, flag_biased=flag_biased, flag_few_training=flag_few_training,
                                      ratio=ratio, imbalanced_rel=imbalanced_rel, exp_idx=exp_idx)



stat_res_path = '../output/' + dataset_using

if not os.path.exists(stat_res_path):
    os.mkdir(stat_res_path)

stat_res_path += '/' + dataset_using
if flag_few_training:
    stat_res_path += '_ratio_' + str(ratio)
if flag_biased:
    if imbalanced_rel is None:
        stat_res_path += '_balanced'
    else:
        stat_res_path += '_imbalanced_rel_' + str(imbalanced_rel)
if flag_time_shift:
    stat_res_path += '_time_shifting'

stat_res_path += "_stat_res.json"

with open(stat_res_path, "w") as f:
    json.dump(output, f)