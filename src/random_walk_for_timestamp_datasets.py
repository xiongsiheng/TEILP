import sys
import os
import argparse
import numpy as np
from joblib import Parallel, delayed
import json
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


# We can choose the steps to do. 
# Random walk is to do random walk with training data and save the results.
# Rule summarization is to convert the walks into rules and save the results.
# Rule application is to apply the rules on validataion data and test data and save the results
steps_to_do = ['random_walk', 'rule_summarization', 'rule_application'][2:]



# Basic settings. No need to change.
dataset_idx = ['icews14', 'icews05-15', 'gdelt100'].index(dataset)
num_entity = [7128, 10488, 100][dataset_idx]
num_rel = [230, 251, 20][dataset_idx]
num_rel_aug = num_rel * 2 # augmented ori relations with inverse relations.
timestamp_range = [range(0, 365), range(0, 4017), range(90, 456)][dataset_idx]


# Load the dataset.
dataset_dir = "../data/{}/".format(dataset) if not flag_time_shifting else "../data/{}/time_shifting/".format(dataset)
data = Grapher(dataset_dir, num_entity, num_rel, timestamp_range)




# Prepare the data for gdelt under the time shifting setting.
if dataset == 'gdelt100' and flag_time_shifting:
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
# Note: This is only for rule body since we don't know the time of rule head.
transition_distr = ["unif", "exp"][-1]

# We use train data as the background knowledge.
BG_idx = data.train_idx.copy()




# Sampling settings. The default setting below is to balance efficiency and accuracy.
seed = 12
rule_lengths = [1, 2, 3]
num_walks_per_TR_ls = [30, 10, 10][dataset_idx]
num_samples_per_rel = [1000, 500, 1000][dataset_idx]
num_walks_per_rule = [10, 5, 5][dataset_idx]
num_rules_per_sample = [10, 10, 10][dataset_idx]
max_num_rules_attempted_per_sample = [10, 10, 10][dataset_idx]


# Create the output folders.
output_folder = ['../output/', '../output/{}'.format(dataset), '../output/{}/walk_res'.format(dataset), 
                 '../output/{}/walk_res_time_shift'.format(dataset), '../output/{}/stat_res'.format(dataset)]
for folder in output_folder:
    if not os.path.exists(folder):
        os.mkdir(folder)



def cut_in_half(data):  
    '''
    Cut the data into half.
    '''
    return data[:len(data)//2].copy(), data[len(data)//2:].copy(), len(data)//2


def prepare_data():
    '''
    Prepare the data for random walk.
    '''
    train_samples, train_samples_inv, num_train = cut_in_half(data.train_idx)
    valid_samples, valid_samples_inv, num_valid = cut_in_half(data.valid_idx)
    test_samples, test_samples_inv, num_test = cut_in_half(data.test_idx)

    # Mask the timestamps for the validation and test data.
    valid_samples[:, 3] = 9999
    valid_samples_inv[:, 3] = 9999
    test_samples[:, 3] = 9999
    test_samples_inv[:, 3] = 9999

    num_total = num_train + num_valid + num_test

    all_samples = np.vstack([train_samples, valid_samples, test_samples, train_samples_inv, valid_samples_inv, test_samples_inv])

    pos_examples_idx = list(range(num_train)) + list(range(num_total, num_total + num_train))
    pos_examples_idx = np.array(pos_examples_idx)

    valid_idx_ls = list(range(num_train, num_train + num_valid)) + list(range(num_total + num_train, num_total + num_train + num_valid))
    valid_idx_ls = np.array(valid_idx_ls)

    test_idx_ls = list(range(num_train + num_valid, num_total)) + list(range(num_total + num_train + num_valid, num_total * 2))
    test_idx_ls = np.array(test_idx_ls)

    return all_samples, pos_examples_idx, valid_idx_ls, test_idx_ls



all_samples, pos_examples_idx, valid_idx_ls, test_idx_ls = prepare_data()

temporal_walk = Temporal_Walk(BG_idx, data.inv_relation_id, transition_distr, flag_interval=False)


def prepare_TR_ls_dict():
    ''' prepare possible temporal relations for different rule lengths '''
    TR_ls_dict = {i: [] for i in range(1, 6)}
    TR_ukn_list = ['ukn']
    TR_list = ['bf', 'touch', 'af']
    TR_ls_dict[1] = [TR_ukn_list]
    TR_ls_dict[2] = list(itertools.product(TR_ukn_list, TR_list))
    TR_ls_dict[3] = list(itertools.product(TR_ukn_list, TR_list, TR_list))
    TR_ls_dict[4] = list(itertools.product(TR_ukn_list, TR_list, TR_list, TR_list))
    TR_ls_dict[5] = list(itertools.product(TR_ukn_list, TR_list, TR_list, TR_list, TR_list))
    return TR_ls_dict


def convert_walk_dict_into_path(walk_results):
    # Extract the lists from the dictionary
    entities = walk_results['entities']
    ts = walk_results['ts']
    relations = walk_results['relations']
    te = walk_results['te'] if 'te' in walk_results else None

    # Initialize the result vector
    result = []

    # Iterate through the entities and add elements to the result vector
    for i in range(len(relations)):
        result.append(entities[i])
        result.append(relations[i])
        result.append(ts[i])
        if te is not None:
            result.append(te[i])
    
    # Add the last entity
    result.append(entities[-1])

    return result


def random_walk_with_sampling(relations_idx, fix_cur_moment=False):
    '''
    Given the relations, perform the random walk with sampling.
    
    For time shifting setting, we are considering predicting the next occurrence of an event 
        given all previous events before the last time the same event (s,r are the same, i.e., (s,r,?); or r,o are the same, i.e., (?,r,o)) happened.
        We call this moment as current_moment, i.e., we know all the history before the current_moment, 
        and standing at this momoent we want to know the next occurrence, i.e., the future event.
    
    fix_cur_moment: In time shifting setting, we consider fix current_moment when consider the two situations: (s,r,?) or (?,r,o).
    '''
    if seed:
        np.random.seed(seed)
    
    TR_ls_dict = prepare_TR_ls_dict()
    for rel in tdqm(relations_idx, desc="Random walk: "):
        sample_rels = all_samples[pos_examples_idx, 1]
        cur_sample_idx = pos_examples_idx[sample_rels == rel].copy()

        rel_inv = rel + num_rel if rel < num_rel else rel - num_rel 

        if num_samples_per_rel is not None:
            np.random.shuffle(cur_sample_idx)
            cur_sample_idx = cur_sample_idx[:num_samples_per_rel]

        for idx in cur_sample_idx:
            sample = all_samples[idx:idx+1]
            sample_inv = obtain_inv_edge(sample, num_rel)[0]
            
            resulted_walk = {"query": sample[0].tolist()}
            for length in rule_lengths:
                resulted_walk[length] = [] 
                for TR_ls in TR_ls_dict[length]:
                    cnt = 0
                    while cnt < num_walks_per_TR_ls:
                        walk_successful, walk, _ = temporal_walk.sample_walk(length+1, rel_inv, sample_inv, TR_ls=TR_ls, window=None, 
                                                                            flag_time_shifting=flag_time_shifting, fix_cur_moment=fix_cur_moment)
                        cnt += 1
                        if walk_successful:
                            # Remove the first three elements in the walk, which are the same as the query.
                            walk = convert_walk_dict_into_path(walk)[3:]
                            if walk not in resulted_walk[length]:
                                resulted_walk[length].append(walk)

            # We do not distinguish the fix_cur_moment setting in the output pathname.
            folder = 'walk_res' if not flag_time_shifting else 'walk_res_time_shift'
            output_path = "../output/{}/{}/{}_idx_{}.json".format(dataset, folder, dataset, idx)     
            with open(output_path, "w") as json_file:
                json.dump(resulted_walk, json_file)

    return 



def split_the_relations_into_groups(pos_examples_idx, num_rel_aug, num_samples_per_rel=None):
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

    num_samples_dict = {rel: 0 for rel in range(num_rel_aug)}
    for idx in pos_examples_idx:
        sample = all_samples[idx]
        num_samples_dict[sample[1]] += 1

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
        group_dict = {0: list(range(num_rel_aug))}
    
    return group_dict


if 'random_walk' in steps_to_do:
    # Process each group of relations in parallel.
    group_dict = split_the_relations_into_groups(pos_examples_idx, num_rel_aug, num_samples_per_rel)
    num_processes = 20
    for c in group_dict:
        all_relations = group_dict[c]
        num_processes = min(len(all_relations), num_processes)
        relations_idx_ls = split_list_into_batches(all_relations, num_batches=num_processes)
        output = Parallel(n_jobs=num_processes)(
            delayed(random_walk_with_sampling)(relations_idx_ls[i], flag_time_shifting) for i in range(num_processes)
        )



def create_stat_res_path():
    '''
    Create the path for the stat res
    '''
    stat_res_path = '../output/{}/stat_res/{}'.format(dataset, dataset)

    if flag_time_shifting:
        stat_res_path += '_time_shifting'

    return stat_res_path



stat_res_path = create_stat_res_path()
targ_rel_ls = list(range(num_rel_aug))


if 'rule_summarization' in steps_to_do:
    # Convert the walks into rules and save the results.
    output_path = 'walk_res' if not flag_time_shifting else 'walk_res_time_shift'
    summarizer = Rule_summarizer()
    summarizer.convert_walks_into_rules(dataset=dataset, path='../output/{}/{}'.format(dataset, output_path), idx_ls=None, 
                                        flag_time_shift=flag_time_shifting, targ_rel_ls=targ_rel_ls, num_processes=24, 
                                        stat_res_path=stat_res_path, flag_interval=False)




def rule_application(targ_rel_ls, sample_idx_ls, flag_time_shifting):
    '''
    Apply the rules based on summarized results.

    all_samples = train_nodes, masked_valid_nodes, masked_test_nodes, train_nodes_inv, masked_valid_nodes_inv, masked_test_nodes_inv
    
    '''
    for rel in tdqm(targ_rel_ls, desc="Rule application: "):
        stat_path = stat_res_path + "_rel_" + str(rel) + ".json" 
        if not os.path.exists(stat_path):
            stat_path = []
        else:
            with open(stat_path) as f:
                stat_res = json.load(f)  
            stat_res = sorted(stat_res.keys(), key=lambda k: max(stat_res[k]['num_samples']), reverse=True)

        cur_samples = all_samples[sample_idx_ls]
        samples_idx_cur_rel = sample_idx_ls[cur_samples[:, 1] == rel]

        for idx in samples_idx_cur_rel:
            sample = all_samples[idx:idx+1]
            sample_inv = obtain_inv_edge(sample.reshape((1, -1)), num_rel)[0]
            resulted_walk = {"query": sample[0].tolist()}

            cnt_rule = 0
            for rule in stat_res[:max_num_rules_attempted_per_sample]:
                flag_rule_cnt = False
                if cnt_rule >= num_rules_per_sample:
                    break

                rule_ls = rule.split(' ')
                length = len(rule_ls)//2
                cnt_walk = 0
                while cnt_walk < num_walks_per_rule:
                    walk_successful, walk, _ = temporal_walk.sample_walk(length + 1, rel, sample_inv, TR_ls=rule_ls[length:], rel_ls=rule_ls[:length], 
                                                                        flag_time_shifting=flag_time_shifting, window=None)
                    cnt_walk += 1
                    if walk_successful:
                        # Remove the first three elements in the walk, which are the same as the query.
                        walk = convert_walk_dict_into_path(walk)[3:]
                        if length not in resulted_walk:
                            resulted_walk[length] = []
                        if walk not in resulted_walk[length]:
                            resulted_walk[length].append(walk)
                            cnt_rule += 1 if not flag_rule_cnt else 0
                            flag_rule_cnt = True

            folder = 'walk_res' if not flag_time_shifting else 'walk_res_time_shift'
            output_path = "../output/{}/{}/{}_idx_{}.json".format(dataset, folder, dataset, idx)     
            with open(output_path, "w") as json_file:
                json.dump(resulted_walk, json_file)
    return


if 'rule_application' in steps_to_do:
    # Apply the rules based on summarized results.
    group_dict = split_the_relations_into_groups(valid_idx_ls, num_rel_aug)

    num_processes = 20
    for c in group_dict:
        all_relations = group_dict[c]
        num_processes = min(len(all_relations), num_processes)
        relations_idx_ls = split_list_into_batches(all_relations, num_batches=num_processes)
        
        # Apply the rules on validation data. Here we comment out to save time.
        # output = Parallel(n_jobs=num_processes)(
        #     delayed(rule_application)(relations_idx_ls[i], valid_idx_ls, flag_time_shifting) for i in range(num_processes)
        # )

        output = Parallel(n_jobs=num_processes)(
            delayed(rule_application)(relations_idx_ls[i], test_idx_ls, flag_time_shifting) for i in range(num_processes)
        )