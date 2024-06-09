import time
import argparse
import numpy as np
from datetime import datetime
from joblib import Parallel, delayed
import json
import random
import sys
import os

from Walker_with_sampling import Grapher, Temporal_Walk
from utlis import process_random_walk_with_sampling_results




parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str)
parser.add_argument("--shift", default=0, type=int)
parsed = vars(parser.parse_args())


dataset = parsed["dataset"]
flag_time_shifting = parsed["shift"]

rule_lengths = [1, 2, 3]
num_processes = 24
seed = 12



dataset_idx = ['icews14', 'icews05-15', 'gdelt100'].index(dataset)
num_entity = [7128, 10488, 100][dataset_idx]
num_rel = [230, 251, 20][dataset_idx]
timestamp_ls = [range(0, 365), range(0, 4017), range(90, 456)][dataset_idx]

dataset_dir = "../data/" + dataset + "/"

if flag_time_shifting:
    dataset_dir = "../data/difficult_settings/" + dataset + "_time_shifting/"



data = Grapher(dataset_dir, num_entity, num_rel, timestamp_ls)

def half_split_data(data):
    return data[:len(data)//2], data[len(data)//2:]




if dataset == 'gdelt100' and flag_time_shifting:
    def obtain_inv_edge(edges, num_rel):
        return np.hstack([edges[:,2:3], edges[:,1:2] + num_rel, edges[:,0:1], edges[:,3:]])

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




transition_distr = ["unif", "exp"][-1]
BG_idx = [data.train_idx, data.valid_idx, data.test_idx][0]



num_walks_per_sample = [10, 3, 10][dataset_idx]
num_samples_per_rel = [100, 100, 500][dataset_idx]





def my_shuffle(data):
    idx = list(range(len(data)))
    random.shuffle(idx)
    new_data = data[idx]
    return new_data


def find_range(ranges, num):
    for (i,range_) in enumerate(ranges):
        if range_[0] <= num <= range_[1]:
            return i
    return None


pos_examples = my_shuffle(data.train_idx)
temporal_walk = Temporal_Walk(BG_idx, data.inv_relation_id, transition_distr, flag_interval=False)


def prepare_TR_ls_dict():
    # possible temporal relations for different rule lengths
    TR_ls_dict = {i: [] for i in range(1, 6)}
    TR1_list = ['ukn']
    TR2_list = TR3_list = TR4_list = TR5_list = ['bf', 'touch', 'af']
    TR_ls_dict[1] = [[TR1] for TR1 in TR1_list]
    TR_ls_dict[2] = [[TR1, TR2] for TR1 in TR1_list for TR2 in TR2_list]
    TR_ls_dict[3] = [[TR1, TR2, TR3] for TR1 in TR1_list for TR2 in TR2_list for TR3 in TR3_list]
    TR_ls_dict[4] = [[TR1, TR2, TR3, TR4] for TR1 in TR1_list for TR2 in TR2_list for TR3 in TR3_list for TR4 in TR4_list]
    TR_ls_dict[5] = [[TR1, TR2, TR3, TR4, TR5] for TR1 in TR1_list for TR2 in TR2_list for TR3 in TR3_list for TR4 in TR4_list for TR5 in TR5_list]
    return TR_ls_dict


TR_ls_dict = prepare_TR_ls_dict()



def random_walk_with_sampling(i, num_relations, all_relations, fix_ref_time=False):
    if seed:
        np.random.seed(seed)

    num_rest_relations = len(all_relations) - (i + 1) * num_relations
    if (num_rest_relations >= num_relations) and (i < num_processes-1):
        relations_idx = range(i * num_relations, (i + 1) * num_relations)
    else:
        relations_idx = range(i * num_relations, len(all_relations))


    for k in relations_idx:
        rel = all_relations[k]

        output_path1 = 'walk_res'
        if flag_time_shifting:
            output_path1 = 'walk_res_time_shift'
        output_path = "../output/" + output_path1 + "/" + dataset + "_random_walks_rel_"+ str(rel) + "_train.json"
        if fix_ref_time:
            output_path = "../output/" + output_path1 + "/" + dataset + "_random_walks_rel_"+ str(rel) + "_fix_ref_time_train.json"

        resulted_walk = {}
        this_idx = (pos_examples[pos_examples[:,1] == rel]).copy()

        if num_samples_per_rel is not None:
            np.random.shuffle(this_idx)
            this_idx = this_idx[:num_samples_per_rel]

        it_start = time.time()
        for length in rule_lengths:
            resulted_walk[length] = {}

            for TR_ls in TR_ls_dict[length]:
                resulted_walk[length][' '.join(TR_ls)] = []

            for i1 in range(len(this_idx)):
                for TR_ls in TR_ls_dict[length]:
                    cnt = 0
                    while cnt < num_walks_per_sample:
                        walk_successful, walk, ref_time = temporal_walk.sample_walk(length + 1, rel, this_idx[i1], TR_ls=TR_ls, window=None, 
                                                                            flag_time_shifting=flag_time_shifting, fix_ref_time=fix_ref_time)
                        cnt += 1
                        if walk_successful:
                            if walk not in resulted_walk[length][' '.join(TR_ls)]:
                                resulted_walk[length][' '.join(TR_ls)].append(walk)
                                # print(walk)

        resulted_walks_final = {}
        for length in resulted_walk:
            resulted_walks_final[length] = {}
            for TR_ls in resulted_walk[length]:
                resulted_walks_final[length][TR_ls] = []
                for rule in resulted_walk[length][TR_ls]:
                    for path in rule:
                        rule[path] = [int(x) for x in rule[path]]
                    resulted_walks_final[length][TR_ls].append(rule)

        it_end = time.time()
        it_time = round(it_end - it_start, 6)
        print("Random walk for rel {} finished in {} seconds.".format(rel, it_time))

        with open(output_path, "w") as json_file:
            json.dump(resulted_walks_final, json_file)

    return 




num_samples_dict = {}
for e in pos_examples:
    if e[1] not in num_samples_dict:
        num_samples_dict[e[1]] = 0
    num_samples_dict[e[1]] += 1



if num_samples_per_rel is not None:
    for rel in num_samples_dict:
        num_samples_dict[rel] = min(num_samples_per_rel, num_samples_dict[rel])


class_dict = {}
for rel in num_samples_dict:
    c = find_range(ranges=[[0, 100], [100, 500], [500, 1000], [1000, 1e9]], num=num_samples_dict[rel])
    if c not in class_dict:
        class_dict[c] = []
    class_dict[c].append(rel)


if dataset == 'gdelt100':
    class_dict[0] = list(range(num_rel*2))


for c in range(len(class_dict)):
    all_relations = class_dict[c]
    num_processes1 = min(len(all_relations), num_processes)
    num_relations = len(all_relations) // num_processes1
    output = Parallel(n_jobs=num_processes1)(
        delayed(random_walk_with_sampling)(i, num_relations, all_relations) for i in range(num_processes1)
    )

    if flag_time_shifting:
        output = Parallel(n_jobs=num_processes1)(
            delayed(random_walk_with_sampling)(i, num_relations, all_relations, True) for i in range(num_processes1)
        )


if flag_time_shifting:
    file_paths = os.listdir('../output/walk_res_time_shift/')
    file_paths = ['../output/walk_res_time_shift/' + f for f in file_paths if dataset in f]
    file_paths = [f for f in file_paths if '_train' in f]
else:
    file_paths = os.listdir('../output/walk_res/')
    file_paths = ['../output/walk_res/' + f for f in file_paths if dataset in f]
    file_paths = [f for f in file_paths if '_train' in f]


file_suffix = '_'
if flag_time_shifting:
    file_suffix = '_time_shifting_'
    file_paths1 = [f for f in file_paths if '_fix_ref_time_' in f]
    file_paths2 = [f for f in file_paths if '_fix_ref_time_' not in f]

    process_random_walk_with_sampling_results(dataset, mode='Train', num_rel=num_rel, file_suffix=file_suffix + 'fix_ref_time_', 
                                file_paths=file_paths1, num_workers=24, flag_time_shifting=flag_time_shifting)
    process_random_walk_with_sampling_results(dataset, mode='Train', num_rel=num_rel, file_suffix=file_suffix, 
                                file_paths=file_paths2, num_workers=24, flag_time_shifting=flag_time_shifting)
else:
    process_random_walk_with_sampling_results(dataset, mode='Train', num_rel=num_rel, file_suffix=file_suffix, 
                                    file_paths=file_paths, num_workers=24)