import sys
import argparse
import numpy as np
import os
import json

from dataset_setting import obtain_dataset
from do_random_walk import random_walk
from utlis import *
from gadgets import *



parser = argparse.ArgumentParser()

# which dataset to use: 'wiki' or 'YAGO'
parser.add_argument('--dataset', type=str)   

# whether use time-shifting setting
parser.add_argument('--shift', default=False, type=bool)

# whether use few-shot learning setting
parser.add_argument('--few', default=False, type=bool)

# whether use biased setting
parser.add_argument('--biased', default=False, type=bool) 

# the ratio of the positive examples (only valid in few-shot learning setting)
parser.add_argument('--ratio', type=float)

# the relation to be imbalanced (only valid in biased setting)
parser.add_argument('--imbalanced_rel', type=int)

# the target relation to be learned (if we assign this, only this relation will be learned)
parser.add_argument('--targ_rel', type=int)

# the index of the experiment (only valid in few-shot learning or biased setting setting when we need to repeat the experiment multiple times)
parser.add_argument('--exp_idx', type=int)





# Get the arguments.
args = parser.parse_args()

dataset_using = args.dataset
targ_rel = args.targ_rel
imbalanced_rel = args.imbalanced_rel
ratio = args.ratio
exp_idx = args.exp_idx

flag_time_shift = args.shift
flag_biased = args.biased
flag_few_training = args.few



# Check the validity of the arguments for difficult settings.
if int(flag_biased) + int(flag_time_shift) + int(flag_few_training) > 1:
    print('Only one difficult setting is allowed.')
    sys.exit()



num_pattern = 3  # The number of Temporal relations
const_pattern_ls = [-1, 0, 1]  # Notations for the temporal relations: [before, touching, after]
f_interval = 1  # whether the dataset is interval-based or point-based

# Basic information of the datasets.
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


# The corresponding dataset names and output paths.
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


# Prepare the parameters for the walker.
paras = [num_rel, num_pattern, num_ruleLen, dataset_using, f_interval]
targ_rel_ls = range(num_rel)
if imbalanced_rel is not None:
    targ_rel_ls = [imbalanced_rel]
if targ_rel is not None:
    targ_rel_ls = [targ_rel]



# Get the data.
train_data_cmp, valid_data, valid_data_inv, test_data, test_data_inv = obtain_dataset(dataset_name1, num_rel)



def prepare_data(train_data_cmp, valid_data, valid_data_inv, test_data, test_data_inv):
    '''
    Mask valid/test data, we use 9999 as unknown value notation (normal (year) values are like 1800, 1923, 2017, etc.)
    '''
    if not flag_time_shift:
        process_data = lambda x: np.hstack((x[:, :3], np.full_like(x[:, 3:], 9999)))
        valid_data = process_data(valid_data)
        valid_data_inv = process_data(valid_data_inv)
        test_data = process_data(test_data)
        test_data_inv = process_data(test_data_inv)

    num_train, num_valid, num_test = len(train_data_cmp)//2, len(valid_data), len(test_data)

    nodes = np.vstack([train_data_cmp[:len(train_data_cmp)//2], valid_data, test_data,
                       train_data_cmp[len(train_data_cmp)//2:], valid_data_inv, test_data_inv])
    return nodes, num_train, num_valid, num_test


nodes, num_train, num_valid, num_test = prepare_data(train_data_cmp, valid_data, valid_data_inv, test_data, test_data_inv)


def prepare_pos_examples_idx():
    '''
    Prepare the positive examples index for few-shot learning setting (if applicable).
    '''
    pos_examples_idx = None
    if flag_few_training:
        pos_examples_idx = []
        with open('../'+ dataset_name + '/' + dataset_name +'_ratio_' + str(ratio) +'_exp_'+ str(exp_idx) + '_training_idx.json', 'r') as file:
            pos_examples_idx += json.load(file)
        pos_examples_idx += list(range(num_train, num_train + num_valid + num_test))
        pos_examples_idx += [idx + num_train + num_valid + num_test for idx in pos_examples_idx]
    return pos_examples_idx


pos_examples_idx = prepare_pos_examples_idx()




if not os.path.exists('../output/'):
    os.mkdir('../output/')



# Perform the random walk.
random_walk(targ_rel_ls, nodes, paras, pos_examples_idx = pos_examples_idx, 
            time_shift_mode=flag_time_shift, output_path=output_path,
            ratio=ratio, imbalanced_rel=imbalanced_rel, exp_idx=exp_idx, num_processes=24)




def create_stat_res_path():
    '''
    Create the path for the stat res
    '''
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

    stat_res_path += "_stat_res"
    return stat_res_path


stat_res_path = create_stat_res_path()


# Convert the walks into rules and save the results.
summarizer = Rule_summarizer()
summarizer.convert_walks_into_rules(dataset=dataset_using, path='../output/' + output_path, idx_ls=pos_examples_idx, 
                                    flag_time_shift=flag_time_shift, flag_biased=flag_biased, flag_few_training=flag_few_training,
                                    ratio=ratio, imbalanced_rel=imbalanced_rel, exp_idx=exp_idx, 
                                    targ_rel_ls=targ_rel_ls, num_processes=24, stat_res_path=stat_res_path)