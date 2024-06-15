import sys
import os
import argparse
import numpy as np
from joblib import Parallel, delayed
import json
import random
from tqdm import tqdm as tdqm
import itertools

from Walker_with_sampling import Grapher, Temporal_Walk
from utlis import *
from gadgets import *



parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str)
parser.add_argument("--shift", default=0, type=int)
parsed = vars(parser.parse_args())


dataset = parsed["dataset"]
flag_time_shifting = parsed["shift"]




# Basic settings. No need to change.
dataset_idx = ['icews14', 'icews05-15', 'gdelt100'].index(dataset)
num_entity = [7128, 10488, 100][dataset_idx]
num_rel = [230, 251, 20][dataset_idx]
timestamp_ls = [range(0, 365), range(0, 4017), range(90, 456)][dataset_idx]


# Load the dataset.
dataset_dir = "../data/{}/".format(dataset) if not flag_time_shifting else "../data/{}/time_shifting/".format(dataset)
data = Grapher(dataset_dir, num_entity, num_rel, timestamp_ls)





if dataset == 'gdelt100' and flag_time_shifting:
    # Prepare the data for gdelt under the time shifting setting.
    with open('../data/gdelt100_sparse_edges.json') as json_file:
        edges = json.load(json_file)

    num_train = [16000, 2000]
    edges = np.array(edges)

    train_edges = edges[:num_train[0]]
    valid_edges = edges[num_train[0]:num_train[0] + num_train[1]]
    test_edges = edges[num_train[0] + num_train[1]:]

    data.train_idx = np.vstack([train_edges, obtain_inv_edge(train_edges, num_rel)])
    data.valid_idx = np.vstack([valid_edges, obtain_inv_edge(valid_edges, num_rel)])
    data.test_idx = np.vstack([test_edges, obtain_inv_edge(test_edges, num_rel)])



# Transition distribution for temporal walk.
# "unif" for uniform distribution and "exp" for exponential distribution.
# The exponential distribution prefers to choose closer timestamps.
transition_distr = ["unif", "exp"][-1]
BG_idx = [data.train_idx, data.valid_idx, data.test_idx][0]


# Sampling settings. No need to change.
rule_lengths = [1, 2, 3]
num_processes = 24
seed = 12
num_walks_per_sample = [10, 3, 10][dataset_idx]
num_samples_per_rel = [100, 100, 500][dataset_idx]




pos_examples = my_shuffle(data.train_idx)
temporal_walk = Temporal_Walk(BG_idx, data.inv_relation_id, transition_distr, flag_interval=False)



def prepare_TR_ls_dict():
    # possible temporal relations for different rule lengths
    TR_ls_dict = {i: [] for i in range(1, 6)}
    TR_ukn_list = ['ukn']
    TR_list = ['bf', 'touch', 'af']
    TR_ls_dict[1] = TR_ukn_list
    TR_ls_dict[2] = list(itertools.product(TR_ukn_list, TR_list))
    TR_ls_dict[3] = list(itertools.product(TR_ukn_list, TR_list, TR_list))
    TR_ls_dict[4] = list(itertools.product(TR_ukn_list, TR_list, TR_list, TR_list))
    TR_ls_dict[5] = list(itertools.product(TR_ukn_list, TR_list, TR_list, TR_list, TR_list))
    return TR_ls_dict



def random_walk_with_sampling(relations_idx, fix_ref_time=False):
    if seed:
        np.random.seed(seed)
    
    TR_ls_dict = prepare_TR_ls_dict()
    for rel in tdqm(relations_idx, desc="Random walk: "):
        resulted_walk = {}
        cur_samples = (pos_examples[pos_examples[:,1] == rel]).copy()

        if num_samples_per_rel is not None:
            np.random.shuffle(cur_samples)
            cur_samples = cur_samples[:num_samples_per_rel]

        for length in rule_lengths:
            resulted_walk[length] = {}
            for TR_ls in TR_ls_dict[length]:
                resulted_walk[length][' '.join(TR_ls)] = []
            for sample in range(len(cur_samples)):
                for TR_ls in TR_ls_dict[length]:
                    cnt = 0
                    while cnt < num_walks_per_sample:
                        walk_successful, walk, _ = temporal_walk.sample_walk(length+1, rel, sample, TR_ls=TR_ls, window=None, 
                                                                            flag_time_shifting=flag_time_shifting, fix_ref_time=fix_ref_time)
                        cnt += 1
                        if walk_successful:
                            if walk not in resulted_walk[length][' '.join(TR_ls)]:
                                resulted_walk[length][' '.join(TR_ls)].append(walk)
        
        # Format the resulted_walk.
        resulted_walks_final = {}
        for length in resulted_walk:
            resulted_walks_final[length] = {}
            for TR_ls in resulted_walk[length]:
                resulted_walks_final[length][TR_ls] = []
                for rule in resulted_walk[length][TR_ls]:
                    for path in rule:
                        rule[path] = [int(x) for x in rule[path]]
                    resulted_walks_final[length][TR_ls].append(rule)

        # We do not distinguish the fix_ref_time setting in the output path.
        folder = 'walk_res' if not flag_time_shifting else 'walk_res_time_shift'
        output_path = "../output/{}/{}_random_walks_rel_{}_train.json".format(folder, dataset, rel)     
        with open(output_path, "w") as json_file:
            json.dump(resulted_walks_final, json_file)

    return 



def split_the_relations_into_groups(pos_examples, num_samples_per_rel):
    '''
    We split the relations into different groups based on the number of samples.
    Group 1 has the relations with less than 100 samples.
    Group 2 has the relations with 100-500 samples.
    Group 3 has the relations with 500-1000 samples.
    Group 4 has the relations with more than 1000 samples.
    '''
    def find_range(ranges, num):
        for (i, range_) in enumerate(ranges):
            if range_[0] <= num <= range_[1]:
                return i
        return None
    
    group_standard = [[0, 100], [100, 500], [500, 1000], [1000, 1e9]]

    num_samples_dict = {}
    for e in pos_examples:
        if e[1] not in num_samples_dict:
            num_samples_dict[e[1]] = 0
        num_samples_dict[e[1]] += 1

    if num_samples_per_rel is not None:
        for rel in num_samples_dict:
            num_samples_dict[rel] = min(num_samples_per_rel, num_samples_dict[rel])

    group_dict = {}
    for rel in num_samples_dict:
        c = find_range(ranges=group_standard, num=num_samples_dict[rel])
        if c not in group_dict:
            group_dict[c] = []
        group_dict[c].append(rel)

    if dataset == 'gdelt100':
        group_dict = {0: list(range(num_rel*2))}
    
    return group_dict


# Process each group of relations in parallel.
group_dict = split_the_relations_into_groups(pos_examples, num_samples_per_rel)
for c in range(len(group_dict)):
    all_relations = group_dict[c]
    num_processes = min(len(all_relations), num_processes)
    relations_idx_ls = split_list_into_batches(all_relations, num_processes)
    output = Parallel(n_jobs=num_processes)(
        delayed(random_walk_with_sampling)(relations_idx_ls[i], flag_time_shifting) for i in range(num_processes)
    )



# Save the results.
if flag_time_shifting:
    file_paths = os.listdir('../output/walk_res_time_shift/')
    file_paths = ['../output/walk_res_time_shift/' + f for f in file_paths if dataset in f]
    file_paths = [f for f in file_paths if '_train' in f]
else:
    file_paths = os.listdir('../output/walk_res/')
    file_paths = ['../output/walk_res/' + f for f in file_paths if dataset in f]
    file_paths = [f for f in file_paths if '_train' in f]


# Todo: Check process_random_walk_with_sampling_results for timestamp datasets.
summarizer = Rule_summarizer()
summarizer.process_random_walk_with_sampling_results(dataset, mode='Train', num_rel=num_rel, file_paths=file_paths, num_workers=24)