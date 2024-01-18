import os
import json
import random
import sys
import time
import argparse
import numpy as np
from datetime import datetime
from joblib import Parallel, delayed

from Walker_with_sampling import Grapher, Temporal_Walk
from utils import process_random_walk_with_sampling_results






parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str)
parser.add_argument("--shift", default=0, type=int)
parsed = vars(parser.parse_args())


dataset = parsed["dataset"]
flag_time_shifting = parsed["shift"]


if flag_time_shifting:
    file_suffix = '_time_shifting'
else:
    file_suffix = ''


dataset_idx = ['icews14', 'icews05-15', 'gdelt100'].index(dataset)
num_entity = [7128, 10488, 100][dataset_idx]
num_rel = [230, 251, 20][dataset_idx]
timestamp_ls = [range(0, 365), range(0, 4017), range(90, 456)][dataset_idx]

num_walks_per_rule = [20, 3, 20][dataset_idx]
num_rules_per_example = 20
max_num_rules_per_rel = 200
seed = 12




# cur_path = '../output/' + dataset + file_suffix + '/samples/'
# filenames = os.listdir(cur_path)
# filenames = [filename for filename in filenames if '_Test_sample' in filename]

# print(len(filenames))
# sys.exit()


dataset_dir = "../data/" + dataset + file_suffix + "/"
if flag_time_shifting:
    dataset_dir = "../data/difficult_settings/" + dataset + file_suffix + "/"

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

    # print(train_edges)

    data.train_idx = np.vstack([train_edges, obtain_inv_edge(train_edges, num_rel)])
    data.valid_idx = np.vstack([valid_edges, obtain_inv_edge(valid_edges, num_rel)])
    data.test_idx = np.vstack([test_edges, obtain_inv_edge(test_edges, num_rel)])

    # print(data.train_idx)

    # sys.exit()





transition_distr = ["unif", "exp"][-1]
data_idx = [data.train_idx, data.valid_idx, data.test_idx][0]

# todo: add known data to time shift
if flag_time_shifting:
    data_idx = np.vstack([data.train_idx, data.valid_idx, data.test_idx])





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


def obtain_inv_edge(edge, num_rel):
    return np.array([edge[2], edge[1] + num_rel, edge[0], edge[3]])


def merge_list(ls_ls):
    output = []
    for ls in ls_ls:
        output += ls
    return output


def split_list_into_pieces(all_indices, num_pieces):
    indices_per_piece = len(all_indices) // num_pieces
    index_pieces = [all_indices[i:i + indices_per_piece] for i in range(0, len(all_indices), indices_per_piece)]
    if len(index_pieces)>num_pieces:
        output = index_pieces[:num_pieces-1] + [merge_list(index_pieces[num_pieces-1:])]
    else:
        output = index_pieces
    return output




num_samples_dict = {}
for edge in data.test_idx:
    if edge[1] not in num_samples_dict:
        num_samples_dict[edge[1]] = 0
    num_samples_dict[edge[1]] += 1



class_dict = {}
for rel in num_samples_dict:
    if rel < num_rel:
        c = find_range(ranges=[[0, 100], [100, 500], [500, 1000], [1000, 1e9]], num=num_samples_dict[rel])
        if c not in class_dict:
            class_dict[c] = []
        class_dict[c].append(rel)





temporal_walk = Temporal_Walk(data_idx, data.inv_relation_id, transition_distr)




def apply_rules(rel_ls):
    if seed:
        np.random.seed(seed)


    for rel in rel_ls:
        resulted_walk = {}
        resulted_walk_rel_inv = {}
        cur_idx = data.test_idx[data.test_idx[:,1] == rel]

        stat_res = {}

        cur_path = "../output/"+ dataset + file_suffix + '/' + dataset + file_suffix + "_pattern_ls_rel_" + str(rel) +".json"
        if not os.path.exists(cur_path):
            stat_res[rel] = []
        else:
            with open(cur_path, "r") as file:
                pattern_ls = json.load(file)
            with open("../output/"+ dataset + file_suffix + '/' + dataset + file_suffix + "_rule_scores_rel_" + str(rel) +".json", "r") as file:
                rule_scores = json.load(file)
            rule_scores = rule_scores[:len(pattern_ls)]
            sorted_pattern = sorted(zip(pattern_ls, rule_scores), key=lambda x: x[1], reverse=True)
            stat_res[rel] = [item[0] for item in sorted_pattern[:max_num_rules_per_rel]]


        rel_inv = rel + num_rel
        if flag_time_shifting:
            file_suffix_inv_rel = file_suffix + '_fix_known_time'
            fix_ref_time = True
        else:
            file_suffix_inv_rel = file_suffix
            fix_ref_time = False

        cur_path = "../output/"+ dataset + file_suffix + '/' + dataset + file_suffix_inv_rel + "_pattern_ls_rel_"+ str(rel_inv) +".json"
        if not os.path.exists(cur_path):
            stat_res[rel_inv] = []
        else:
            with open(cur_path, "r") as file:
                pattern_ls = json.load(file)
            with open("../output/"+ dataset + file_suffix + '/' + dataset + file_suffix_inv_rel + "_rule_scores_rel_"+ str(rel_inv) +".json", "r") as file:
                rule_scores = json.load(file)
            rule_scores = rule_scores[:len(pattern_ls)]
            sorted_pattern = sorted(zip(pattern_ls, rule_scores), key=lambda x: x[1], reverse=True)
            stat_res[rel_inv] = [item[0] for item in sorted_pattern[:max_num_rules_per_rel]]

        
        
        for i1 in range(len(cur_idx)):
            sample = {}
            cnt1 = 0
            edge = cur_idx[i1]
            cur_path = '../output/' + dataset + file_suffix + '/samples/' + dataset + file_suffix + '_Test_sample_' + str(tuple(edge)) + ".json"
            # if os.path.exists(cur_path):
            #     continue
            # print(edge)
            resulted_walk[tuple(edge)] = {}
            sample[tuple(edge)] = {}

            if rel not in stat_res:
                continue

            ref_time = None

            for rule in stat_res[rel]:
                cnt1 += 1

                rule_ls = rule.split(' ')
                length = len(rule_ls)//2
                cnt = 0
                while cnt < num_walks_per_rule:
                    walk_successful, walk, ref_time = temporal_walk.sample_walk(length + 1, rel, cur_idx[i1], TR_ls=rule_ls[length:], rel_ls=rule_ls[:length], 
                                                                        flag_time_shifting=flag_time_shifting, window=None)
                    cnt += 1
                    # print(ref_time)
                    if walk_successful:
                        walk_pure = {'entities': walk['entities'][2:-1], 'ts': walk['ts'][1:]}
                        cur_ref_edge = [[walk['entities'][1], walk["relations"][1], walk['entities'][2], walk['ts'][1]], 
                                        [walk['entities'][-2], walk["relations"][-1], walk['entities'][-1], walk['ts'][-1]]]
                        if rule not in resulted_walk[tuple(edge)]:
                            resulted_walk[tuple(edge)][rule] = []
                            sample[tuple(edge)][rule] = []
                        if walk_pure not in resulted_walk[tuple(edge)][rule]:
                            resulted_walk[tuple(edge)][rule].append(walk_pure)
                        if cur_ref_edge not in sample[tuple(edge)][rule]:
                            sample[tuple(edge)][rule].append(cur_ref_edge)


                if cnt1 > num_rules_per_example:
                    break

        
            cnt2 = 0
            edge_inv = obtain_inv_edge(cur_idx[i1], num_rel)
            resulted_walk_rel_inv[tuple(edge_inv)] = {}
            sample[tuple(edge_inv)] = {}

            if rel_inv not in stat_res:
                continue

            for rule in stat_res[rel_inv]:
                cnt2 += 1

                rule_ls = rule.split(' ')
                length = len(rule_ls)//2
                cnt = 0
                while cnt < num_walks_per_rule:
                    walk_successful, walk, ref_time = temporal_walk.sample_walk(length + 1, rel_inv, edge_inv, TR_ls=rule_ls[length:], rel_ls=rule_ls[:length], 
                                                                        flag_time_shifting=flag_time_shifting, window=None, fix_ref_time=fix_ref_time)
                    cnt += 1
                    # print(ref_time)
                    if walk_successful:
                        walk_pure = {'entities': walk['entities'][2:-1], 'ts': walk['ts'][1:]}
                        cur_ref_edge = [[walk['entities'][1], walk["relations"][1], walk['entities'][2], walk['ts'][1]], 
                                        [walk['entities'][-2], walk["relations"][-1], walk['entities'][-1], walk['ts'][-1]]]
                        if rule not in resulted_walk_rel_inv[tuple(edge_inv)]:
                            resulted_walk_rel_inv[tuple(edge_inv)][rule] = []
                            sample[tuple(edge_inv)][rule] = []
                        if walk_pure not in resulted_walk_rel_inv[tuple(edge_inv)][rule]:
                            resulted_walk_rel_inv[tuple(edge_inv)][rule].append(walk_pure)
                        if cur_ref_edge not in sample[tuple(edge_inv)][rule]:
                            sample[tuple(edge_inv)][rule].append(cur_ref_edge)

                if cnt2 > num_rules_per_example:
                    break
            
            new_sample = {'ref_time': ref_time}
            for e in sample:
                new_sample[str(e)] = sample[e]
            with open(cur_path, "w") as f:
                json.dump(new_sample, f)
            # print('-------------------------------')

    return



num_processes1 = 24
for c in range(len(class_dict)):
    num_processes = min(num_processes1, len(class_dict[c]))
    idx_pieces = split_list_into_pieces(class_dict[c], num_processes)
    output = Parallel(n_jobs=num_processes)(
        delayed(apply_rules)(idx_pieces[i]) for i in range(num_processes)
    )