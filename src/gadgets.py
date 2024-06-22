import os
import json
import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm
import copy
import random
import sys
from collections import Counter
from utlis import *




class Data_Processor():
    def _prepare_basic_info(self, data, option):
        '''
        Prepare basic information for the dataset.

        Note that entity in TEKG is event in TKG.
        Previously, we use static value for num_entity in the whole graph.
        Now we use dynamic value for num_entity in local graph.

        Parameters:
            data: dict, store the data information
            option: Option, store the experiment settings

        Returns:
            dataset_index: int, index of the dataset
        '''
        dataset_index = ['wiki', 'YAGO', 'icews14', 'icews05-15', 'gdelt100'].index(option.dataset)

        data['walk_res_path'] = '../output/{}/walk_res'.format(option.dataset) if not option.shift else '../output/{}/walk_res_time_shift'.format(option.dataset)
        
        data['dataset'] = ['WIKIDATA12k', 'YAGO11k', 'icews14', 'icews05-15', 'gdelt100'][dataset_index]
        if option.shift:
            data['dataset'] = 'difficult_settings/{}_time_shifting'.format(data['dataset'])

        data['short_name'] = ['wiki', 'YAGO', 'icews14', 'icews05-15', 'gdelt100'][dataset_index]

        data['num_rel'] = [48, 20, 460, 502, 40][dataset_index]
        data['num_TR'] = 4
         
        return dataset_index
    

    def _trainig_data_sampling(self, train_edges, pos_examples_idx, num_rel, num_sample_per_rel=-1):
        '''
        Sample training data for efficiency.

        Parameters:
            train_edges: np.array, training data
            pos_examples_idx: list, indices of the positive examples
            num_rel: int, number of relations
            num_sample_per_rel: int, number of samples per relation (-1 means no sampling)

        Returns:
            pos_examples_idx_sample: list, indices of the sampled positive examples
        '''
        if num_sample_per_rel > 0:
            pos_examples_idx_sample = []
            pos_examples = train_edges[pos_examples_idx]
            pos_examples_idx = np.array(pos_examples_idx)
            
            for rel_idx in range(num_rel):
                sample_idx_cur_rel = pos_examples_idx[pos_examples[:, 1] == rel_idx]
                np.random.shuffle(sample_idx_cur_rel)
                pos_examples_idx_sample.append(sample_idx_cur_rel[:num_sample_per_rel])

            pos_examples_idx_sample = np.hstack(pos_examples_idx_sample).tolist()
            return pos_examples_idx_sample

        return pos_examples_idx


    def _prepare_gdelt_shift_data(self, data):
        '''
        Prepare the data for the GDELT dataset with time shifting.
        '''
        # We use the subset of the whole graph here.
        with open('../data/gdelt100_sparse_edges.json') as json_file:
            nodes = json.load(json_file)
            nodes = np.array(nodes)

        data['num_samples_dist'] = [16000, 2000, 2000]
        
        data['train_nodes'] = nodes[:data['num_samples_dist'][0]]
        data['valid_nodes'] = nodes[data['num_samples_dist'][0]: data['num_samples_dist'][0] + data['num_samples_dist'][1]]
        data['test_nodes'] = nodes[data['num_samples_dist'][0] + data['num_samples_dist'][1]:]
        
        return 


    def _prepare_nodes(self, data, option, dataset_index):
        '''
        Prepare nodes in TEKG which are edges in TKG.

        Parameters:
            data: dict, store the data information
            option: Option, store the experiment settings
            dataset_index: int, index of the dataset

        Returns:
            None
        '''
        data['train_nodes'], data['valid_nodes'], data['test_nodes'] = self.obtain_all_data(data['dataset'], shuffle_train_set=False)
        data['timestamp_range'] = [np.arange(-3, 2024, 1), np.arange(-431, 2024, 1), np.arange(0, 366, 1), np.arange(0, 4017, 1), np.arange(90, 456, 1)][dataset_index]
        data['num_samples_dist'] = [[32497, 4062, 4062], [16408, 2050, 2051], [72826, 8941, 8963], [368962, 46275, 46092], [390045, 48756, 48756]][dataset_index]

        if option.dataset == 'gdelt100' and option.shift:
           self._prepare_gdelt_shift_data(data)

        # Adjust train_idx_ls according to random walk results.
        data['train_idx_ls'] = []
        for idx in range(data['num_samples_dist'][0]):
            output_path = "{}/{}_idx_{}.json".format(data['walk_res_path'], option.dataset, idx)
            if os.path.exists(output_path):  
                data['train_idx_ls'].append(idx)

        data['valid_idx_ls'] = list(range(data['num_samples_dist'][0], data['num_samples_dist'][0] + data['num_samples_dist'][1]))
        data['test_idx_ls'] = list(range(data['num_samples_dist'][0] + data['num_samples_dist'][1], np.sum(data['num_samples_dist'])))
        
        if option.shift:
            data['train_idx_ls'] = [idx + np.sum(data['num_samples_dist']) for idx in data['train_idx_ls']]
            data['valid_idx_ls'] = [idx + np.sum(data['num_samples_dist']) for idx in data['valid_idx_ls']]
            data['test_idx_ls'] = [idx + np.sum(data['num_samples_dist']) for idx in data['test_idx_ls']]

        if option.dataset in ['wiki', 'YAGO']:
            with open('../data/{}_time_pred_eval_rm_idx.json'.format(data['short_name']), 'r') as file:
                data['rm_ls'] = json.load(file)
            if option.shift:
                with open('../data/{}_time_pred_eval_rm_idx_shift_mode.json'.format(data['short_name']), 'r') as file:
                    data['rm_ls'] = json.load(file)

            # remove test samples with unknown time      
            data['test_idx_ls'] = [idx for idx in data['test_idx_ls'] if (idx - data['num_samples_dist'][0] - data['num_samples_dist'][1]) not in data['rm_ls']]
        
        return


    def _prepare_stat_res(self, data, option, dataset_index):
        '''
        Prepare the statistics results for the dataset.

        Parameters:
            data: dict, store the data information
            option: Option, store the experiment settings
            dataset_index: int, index of the dataset

        Returns:
            None
        '''
        num_rel = data['num_rel']//2 if not option.shift else data['num_rel']  # known time range change

        data['pattern_ls'], data['ts_stat_ls'], data['te_stat_ls'], data['stat_res'] = self.processing_stat_res(data['short_name'], num_rel, 
                                                                                                                flag_time_shifting=option.shift,
                                                                                                                flag_interval=option.flag_interval)
        
        # relations w/o duration
        data['rel_ls_no_dur'] = [[4, 16, 17, 20], [0, 7], None, None, None][dataset_index]

        # We do not create the whole graph in advance which is time-consuming.
        data['connectivity_rel'], data['connectivity_TR'], data['TEKG_nodes'], data['TEKG_nodes_idx'] = None, None, None, None

        # Todo: use the duration information for learning
        if hasattr(option, 'flag_use_dur') and option.flag_use_dur:
            with open("../output/{}/dur_preds.json".format(data['short_name']), "r") as json_file:
                data['pred_dur'] = json.load(json_file)
        return
    

    def prepare_data(self, option, save_option=True, preprocess_walk_res=True):
        '''
        Prepare the data for the experiment.
        
        Parameters:
            option: Option, store the experiment settings
            save_option: bool, whether to save the option
            preprocess_walk_res: bool, whether to preprocess the random walk results to obtain query time probabilities
                                 This is only done for training data.

        Returns:
            data: dict, store the data information
        '''
        data = {}
        dataset_index = self._prepare_basic_info(data, option)
        option.flag_interval = True if dataset_index in [0, 1] else False

        self._prepare_nodes(data, option, dataset_index)
        self._prepare_stat_res(data, option, dataset_index)

        if save_option:
            option.this_expsdir = os.path.join(option.exps_dir, '{}_{}'.format(data['short_name'], option.tag))
            if not os.path.exists(option.this_expsdir):
                os.makedirs(option.this_expsdir)
            option.ckpt_dir = os.path.join(option.this_expsdir, "ckpt")
            if not os.path.exists(option.ckpt_dir):
                os.makedirs(option.ckpt_dir)
            option.model_path = os.path.join(option.ckpt_dir, "model")

            option.num_step = [4, 6, 4, 4, 4][dataset_index]
            option.num_rule = [1000, 1000, 1000, 1000, 1000][dataset_index]  # for each relation
            
            option.savetxt = '{}/intermediate_res.txt'.format(option.this_expsdir)
            option.save()
            print("Option saved.")


        data['random_walk_res'] = None
        if option.train and preprocess_walk_res:
            # Once this step is done, we recommend to save the results without rewriting unless the settings are changed.
            self.prepare_random_walk_res(option, data, 'Train', num_workers=20, show_tqdm=True, rewriting=True, flag_interval=option.flag_interval)
            print("Data prepared.")

        return data


    def obtain_all_data(self, dataset, shuffle_train_set=True, use_validation=True):
        '''
        Obtain the training, validation, and test data for the dataset.
        Note that nodes in TEKG are edges in TKG.

        Parameters:
            dataset: str, name of the dataset
            shuffle_train_set: bool, whether to shuffle the training set
            use_validation: bool, whether to use the validation set

        Returns:
            nodes1: np.array, training data
            nodes2: np.array, validation data
            nodes3: np.array, test data
        '''
        nodes1 = read_dataset_txt('../data/{}/train.txt'.format(dataset))
        nodes1 = np.array(nodes1)

        if shuffle_train_set:
            nodes1, _ = shuffle_data(nodes1)

        nodes2 = None
        if use_validation:
            nodes2 = read_dataset_txt('../data/{}/valid.txt'.format(dataset))
            nodes2 = np.array(nodes2)

        nodes3 = read_dataset_txt('../data/{}/test.txt'.format(dataset))
        nodes3 = np.array(nodes3)

        return nodes1, nodes2, nodes3


    def processing_stat_res(self, dataset, num_rel, flag_interval=True, flag_time_shifting=False):
        '''
        Process the statistics results for the dataset.

        Parameters:
            dataset: str, name of the dataset
            num_rel: int, number of relations
            flag_time_shifting: int, whether to use time shifting   

        Returns:
            pattern_ls: dict, list of patterns for each relation
            ts_stat_ls: dict, list of statistics for the query start time for each relation
            te_stat_ls: dict, list of statistics for the query end time for each relation
            stat_res: dict, statistics results for the dataset
        '''
        path_suffix = '' if not flag_time_shifting else '_time_shifting'
        pattern_ls, ts_stat_ls, te_stat_ls, stat_res = {}, {}, {}, {}
        for rel in range(num_rel):
            pattern_ls[rel], ts_stat_ls[rel], te_stat_ls[rel], stat_res[rel] = [], [], [], []

            # read results
            path = '../output/{}/stat_res/{}{}_rel_{}.json'.format(dataset, dataset, path_suffix, rel)
            if not os.path.exists(path):
                continue
            with open(path, "r") as f:
                res = json.load(f)
            
            for rule in res:
                if len(res[rule].keys()) == 0:
                    continue
                pattern_ls[rel].append(rule)

                for k in res[rule]:
                    res[rule][k] = np.array(res[rule][k])

                # remove invalid mean and std values
                res[rule]['mean'][res[rule]['num_samples'] == 0] = 9999
                res[rule]['std'][res[rule]['num_samples'] == 0] = 9999
                res[rule]['std'][res[rule]['std'] < 5] = 5

                if flag_interval:
                    # res: time_gap_ts_ref_ts + time_gap_ts_ref_te + time_gap_te_ref_ts + time_gap_te_ref_te
                    # cur_ts_stat_ls: 'ts_first_event_ts', 'ts_first_event_te', 'ts_last_event_ts', 'ts_last_event_te'
                    # cur_te_stat_ls: 'te_first_event_ts', 'te_first_event_te', 'te_last_event_ts', 'te_last_event_te'
                    cur_ts_stat_ls = [item for pair in zip(res[rule]['mean'][[0, 2, 1, 3]], res[rule]['std'][[0, 2, 1, 3]]) for item in pair]
                    cur_te_stat_ls = [item for pair in zip(res[rule]['mean'][[4, 6, 5, 7]], res[rule]['std'][[4, 6, 5, 7]]) for item in pair]

                    ts_stat_ls[rel].append(cur_ts_stat_ls)
                    te_stat_ls[rel].append(cur_te_stat_ls)
                else:
                    # res: time_gap_ts_ref_ts
                    # cur_ts_stat_ls: 'ts_first_event_ts', 'ts_last_event_ts'
                    cur_ts_stat_ls = [item for pair in zip(res[rule]['mean'][[0, 1]], res[rule]['std'][[0, 1]]) for item in pair]

                    ts_stat_ls[rel].append(cur_ts_stat_ls)
    
        return pattern_ls, ts_stat_ls, te_stat_ls, stat_res


    def _update_stat_res(self, walk, stat_res, samples_edges, mode, TR_ls, flag_interval):
        '''
        Given a walk, update the statistics results for the rule that the walk follows.

        Parameters:
            walk: dict, walk results
            stat_res: dict, statistics results
            samples_edges: dict, samples of edges
            mode: str, mode
            TR_ls: str, TR_ls
            flag_interval: bool, whether to use interval

        Returns:
            None
        '''
        e = [walk['entities'][0], walk["relations"][0], walk['entities'][1], walk['ts'][0]]
        if flag_interval:
            e.append(walk['te'][0])
        
        rule_pattern = '{} {}'.format(' '.join([str(cur_rel) for cur_rel in walk["relations"][1:]]), TR_ls)
        cur_ref_edge = [[walk['entities'][1], walk["relations"][1], walk['entities'][2], walk['ts'][1]], 
                        [walk['entities'][-2], walk["relations"][-1], walk['entities'][-1], walk['ts'][-1]]]

        if rule_pattern not in stat_res:
            stat_res[rule_pattern] = []

        if mode == 'Train':
            if flag_interval:
                cur_time_gap = list_subtract(walk['ts'][0:1], walk['ts'][-1:]) + list_subtract(walk['te'][0:1], walk['ts'][-1:])
            else:
                cur_time_gap = list_subtract(walk['ts'][0:1], walk['ts'][1:2]) + list_subtract(walk['ts'][0:1], walk['ts'][-1:])
            stat_res[rule_pattern].append(cur_time_gap)

        e = str(tuple(e))
        if e not in samples_edges:
            samples_edges[e] = {}
        if rule_pattern not in samples_edges[e]:
            samples_edges[e][rule_pattern] = []

        samples_edges[e][rule_pattern].append(cur_ref_edge)


    def _find_relation_specific_data(self, file_paths, rel):
        '''
        Given the file paths, find the relation-specific data.

        Parameters:
            file_paths: list, file paths
            rel: int, relation

        Returns:
            data: dict, relation-specific data
        '''
        data = {}
        for file_path in file_paths:
            if not '_rel_{}'.format(str(rel)) in file_path:
                continue
            with open(file_path) as f:
                data.update(json.load(f))
        return data


    def _format_stat_res(self, stat_res, rule_pattern, flag_plot):
        '''
        Formatting the statistics results for the rule pattern.

        Parameters:
            stat_res: dict, statistics results
            rule_pattern: str, rule pattern
            flag_plot: bool, whether to plot the results

        Returns:
            None
        '''
        res = np.vstack(stat_res[rule_pattern]).astype(float)
        stat_res[rule_pattern] = {'ts_first_event':{}, 'ts_last_event':{}}

        mean_ls, std_ls, prop_ls = adaptive_Gaussian_dist_estimate_new_ver(res[:, 0])
        stat_res[rule_pattern]['ts_first_event'] = {'mean_ls': mean_ls, 'std_ls': std_ls, 'prop_ls': prop_ls}
        mean_ls, std_ls, prop_ls = adaptive_Gaussian_dist_estimate_new_ver(res[:, 1])
        stat_res[rule_pattern]['ts_last_event'] = {'mean_ls': mean_ls, 'std_ls': std_ls, 'prop_ls': prop_ls}

        if flag_plot:
            for rule_length in [1,2,3,4,5]:
                plot_freq_hist(res[:, 0], 10, 'fig/len{}/{}'.format(str(rule_length), '_'.join([rule_pattern, 'ts'])))
                plot_freq_hist(res[:, 1], 10, 'fig/len{}/{}'.format(str(rule_length), '_'.join([rule_pattern, 'te'])))
        return


    def process_random_walk_results(self, dataset, rel_ls, file_paths, data=None, flag_interval=True, flag_plot=False, mode=None, flag_time_shifting=False,
                                    num_rules_per_rel=1000):
        '''
        Split the combined random walk results to obtain single samples and stat res.
        This is done for timestamp datasets only.

        Parameters:
            dataset: str, name of the dataset
            rel_ls: list, list of relations
            file_paths: list, file paths
            data: dict, data
            flag_interval: bool, whether to use interval
            flag_plot: bool, whether to plot the results
            mode: str, mode
            flag_time_shifting: bool, whether to use time shifting,
            num_rules_per_rel: int, number of preserved rules per relation (based on frequency)

        Returns:
            None
        '''
        path_dataset = dataset if not flag_time_shifting else '{}_time_shifting'.format(dataset)
        
        output_dir = '../output/{}'.format(path_dataset)
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)
            os.mkdir('{}/samples'.format(output_dir))

        for rel in rel_ls:
            samples_edges, stat_res = {}, {}
            if data is None:
                data = self._find_relation_specific_data(file_paths, rel)
       
            if len(data) == 0:
                continue

            for rule_length in [1,2,3,4,5]:
                if str(rule_length) not in data:
                    continue
                for TR_ls in data[str(rule_length)]:
                    for walk in data[str(rule_length)][TR_ls]:
                        self._update_stat_res(walk, stat_res, samples_edges, mode, TR_ls, flag_interval)
            
            # obtain stat information.
            if mode == 'Train':
                for rule_pattern in stat_res:
                    self._format_stat_res(stat_res, rule_pattern, flag_plot)
     
                with open('{}/stat_res_rel_{}_.json'.format(output_dir, rel), 'w') as f:
                    json.dump(stat_res, f)

                for e in samples_edges:
                    with open("{}/samples/{}_{}_sample_{}.json".format(output_dir, path_dataset, mode, e), "w") as f:
                        json.dump(convert_dict({e: samples_edges[e]}), f)

                rule_nums = rule_num_stat(samples_edges)

                if rel not in rule_nums:
                    pattern_ls = []
                else:
                    pattern_ls = sorted(rule_nums[rel], key=lambda p: rule_nums[rel][p], reverse=True)[:num_rules_per_rel]
                    random.shuffle(pattern_ls)

                with open("{}/{}_pattern_ls_rel_{}.json".format(output_dir, path_dataset, rel), 'w') as file:
                    json.dump(pattern_ls, file)

        return


    def process_random_walk_results_dist_ver(self, dataset, mode, num_rel, file_paths, num_workers=10):
        '''
        Process the random walk results for the dataset in a distributed way.

        Parameters:
            dataset: str, name of the dataset
            mode: str, mode
            num_rel: int, number of relations
            file_paths: list, file paths
            num_workers: int, number of workers

        Returns:
            None
        '''
        index_pieces = split_list_into_batches(range(num_rel), num_batches=num_workers)

        data = {}
        for file_path in file_paths:
            with open(file_path) as f:
                data.update(json.load(f))

        Parallel(n_jobs=num_workers)(delayed(self.process_random_walk_results)(dataset, piece, data=data, flag_interval=False, 
                                                                                flag_plot=False, mode=mode) for piece in index_pieces)
        return


    def _calculate_time_prob_dist(self, edge, timestamp_range, targ_interval, stat_ls, idx_ls, mode, flag_time_shift=False, probs_normalization=True, flag_interval=True):
        '''
        Calculate the probability distribution of the time gap.

        Parameters:
            edge: tuple, edge
            timestamp_range: np.array, timestamp range
            targ_interval: np.array, target interval
            stat_ls: list, statistics
            idx_ls: list, indices
            mode: str, mode
            flag_time_shift: bool, whether to use time shift
            probs_normalization: bool, whether to normalize the probabilities
            flag_interval: bool, whether to use interval

        Returns:
            flag_success: bool, whether the calculation is successful
            cur_probs: np.array, current probabilities
        '''
        idx_ts_or_te, idx_first_or_last_event, idx_ref_time_ts_or_te = idx_ls

        # Find the index of the statistics (which is a composite vector)
        idx_stat = 4*idx_ts_or_te + 2*idx_first_or_last_event + idx_ref_time_ts_or_te if flag_interval else idx_first_or_last_event

        # edge: (s, r, ts, te, o) or (s, r, ts, o)
        refTime = edge[2:4][idx_ref_time_ts_or_te] if flag_interval else edge[2:3][idx_ref_time_ts_or_te]   
        delta_t = timestamp_range - refTime
        targ_time = targ_interval[idx_ts_or_te]

        Gau_mean = stat_ls[idx_stat*2]
        Gau_std = stat_ls[idx_stat*2+1]
        
        if 9999 in [refTime, Gau_mean, Gau_std]:
            # invalid time exists
            return False, []

        cur_probs = gaussian_pdf(delta_t, Gau_mean, Gau_std)
        
        if flag_time_shift:
            cur_probs[delta_t < 0] = 0

        if len(cur_probs[np.isnan(cur_probs)])>0 or len(cur_probs[np.isinf(cur_probs)])>0 or sum(cur_probs) == 0:
            return False, []

        if probs_normalization:
            cur_probs /= np.sum(cur_probs)
        
        cur_probs = np.around(cur_probs, decimals=6)

        if mode == 'Train':
            # use ground truth
            cur_probs = cur_probs[delta_t.tolist().index(targ_time - refTime)]

        return True, cur_probs


    def _calculate_event_distribution(self, data, flag_interval):
        '''
        Given a rule, count the distribution of the first and last events. 
        Found that other positions are not so useful.

        Parameters:
            data: list, data

        Returns:
            first_position_distribution: dict, distribution of the first event
            last_position_distribution: dict, distribution of the last event
        '''
        first_position_counter = Counter()
        last_position_counter = Counter()

        for event_ls in data:
            first_event, last_event = tuple(event_ls[0]), tuple(event_ls[1])
            
            # remove unknown time events
            # 9999 is the placeholder for unknown time
            # Event format: (s, r, ts, te, o)
            if flag_interval:
                if first_event[2] != 9999 and first_event[3] != 9999:  
                    first_position_counter[first_event] += 1
    
                if last_event[2] != 9999 and last_event[3] != 9999:    
                    last_position_counter[last_event] += 1
            else:
                if first_event[2] != 9999:  
                    first_position_counter[first_event] += 1
    
                if last_event[2] != 9999:    
                    last_position_counter[last_event] += 1

        total_first = sum(first_position_counter.values())
        total_last = sum(last_position_counter.values())
        
        first_position_distribution = {k: v*1. / total_first for k, v in first_position_counter.items()}
        last_position_distribution = {k: v*1. / total_last for k, v in last_position_counter.items()}

        return [first_position_distribution, last_position_distribution]


    def _create_probs_dict(self, walk_res, query_time, pattern_ls, ts_stat_ls, te_stat_ls, timestamp_range, 
                           flag_rule_split, mode, flag_time_shift, probs_normalization, flag_interval):
        '''
        Given a sample, create a dictionary for storing the probabilities of the time gap.

        Parameters:
            walk_res: walk_res for the current sample
            (global) pattern_ls: list of all possible rules given the query relation
            (global) ts_stat_ls: list of statistics for the query start time given the query relation
            (global) te_stat_ls: list of statistics for the query end time given the query relation
            timestamp_range: np.array, timestamp range
            flag_rule_split: bool, whether to split the rules
            mode: str, mode
            flag_time_shift: bool, whether to use time shift
            probs_normalization: bool, whether to normalize the probabilities
            flag_interval: bool, whether to use interval

        Returns:
            probs: dict, probabilities of the time gap
        '''
        valid_rules = [p for p in walk_res if p in pattern_ls]
        if len(valid_rules) == 0:
            return None
        
        probs = {'0':{}, '1':{}}  # [first event, last event]
        for p in valid_rules:
            # for each rule, calculate the probability of the time gap
            p_idx = pattern_ls.index(p)
            ruleLen = len(p.split(' '))//2
            cur_stat_ls = ts_stat_ls[p_idx]
            if flag_interval: 
                cur_stat_ls += te_stat_ls[p_idx]

            events_for_cur_rule = []
            for walk in walk_res[p]['edge_ls']:
                events_for_cur_rule.append([walk[idx] for idx in [0, -1]])  # we only consider the first and last event
            
            events_for_cur_rule = self._calculate_event_distribution(events_for_cur_rule, flag_interval)

            for idx_event_pos in [0, 1]:
                for edge in events_for_cur_rule[idx_event_pos]:
                    alpha = events_for_cur_rule[idx_event_pos][edge]            
                    if str_tuple(edge) not in probs[str(idx_event_pos)]:
                        probs[str(idx_event_pos)][str_tuple(edge)] = {str(i): [] for i in range(4)} if not flag_rule_split else \
                                                                     {str(i): {str(rLen): [] for rLen in range(1, 6)} for i in range(4)}

                    for idx_query_time in [0, 1]:
                        for idx_ref_time in [0, 1]:
                            if (not flag_interval) and ((idx_query_time !=0) or (idx_ref_time != 0)):
                                continue
                            flag_success, cur_probs = self._calculate_time_prob_dist(edge, timestamp_range, query_time, cur_stat_ls, 
                                                                                        [idx_query_time, idx_event_pos, idx_ref_time], 
                                                                                        mode, flag_time_shift, probs_normalization, flag_interval)
                            if not flag_success:
                                cur_probs = np.array(1./len(timestamp_range)) if mode == 'Train' else \
                                            np.array([1./len(timestamp_range) for _ in range(len(timestamp_range))])
                            
                            cur_probs = (cur_probs * alpha).tolist()
                            cur_probs = [cur_probs, p_idx]  # Add rule idx for tracking.

                            idx_composite = 2*idx_query_time + idx_ref_time
                            if flag_rule_split:
                                probs[str(idx_event_pos)][str_tuple(edge)][str(idx_composite)][str(ruleLen)].append(cur_probs)                                    
                            else:
                                probs[str(idx_event_pos)][str_tuple(edge)][str(idx_composite)].append(cur_probs)
        return probs


    def prepare_inputs(self, res_path, dataset, nodes, idx_ls, targ_rel, pattern_ls, timestamp_range, 
                       ts_stat_ls, te_stat_ls, mode=None, flag_time_shift=False, write_to_file=False, show_tqdm=False,
                       flag_rule_split=False, rewriting=False, probs_normalization=True, flag_interval=True):            
        '''
        Prepare the input probabilities.

        Parameters:
            res_path: str, path of the results
            dataset: str, name of the dataset
            nodes: np.array, nodes
            idx_ls: list, indices
            targ_rel: int, target relation
            pattern_ls: dict, list of patterns
            timestamp_range: np.array, timestamp range
            ts_stat_ls: dict, list of statistics for the query start time
            te_stat_ls: dict, list of statistics for the query end time
            mode: str, mode
            flag_time_shift: bool, whether to use time shift
            write_to_file: bool, whether to write the results to file
            show_tqdm: bool, whether to show the progress bar
            flag_rule_split: bool, whether to split the rules
            rewriting: bool, whether to rewrite the results
            probs_normalization: bool, whether to normalize the probabilities
            flag_interval: bool, whether to use interval

        Returns:
            query_time: dict, query time
            probs: dict, probabilities of query time
        '''
        query_time, probs = {}, {}
        if show_tqdm:
            idx_ls = tqdm(idx_ls, desc='Prepare input probs')
        
        path_folder = '../output/{}/process_res/'.format(dataset)
        if write_to_file:
            if not os.path.exists(path_folder):
                os.mkdir(path_folder)

        for data_idx in idx_ls:
            # Check if the file exists
            if write_to_file:
                query_time, probs = {}, {}
                output_filename = '{}_idx_{}_input.json'.format(dataset, data_idx)
                if os.path.exists(path_folder + output_filename) and (not rewriting):
                    continue
                    
            # read walk res
            input_filename = '{}/{}_idx_{}.json'.format(res_path, dataset, data_idx)
            if not os.path.exists(input_filename):
                continue
            with open(input_filename, 'r') as f:
                data = f.read()
                json_data = json.loads(data)

            # check the query relation if specified
            if targ_rel != None and json_data['query'][1] != targ_rel:
                continue
            
            # extract query info
            cur_rel = json_data['query'][1]
            cur_output = self.process_TEILP_results(json_data, flag_interval=flag_interval)
            if mode == 'Train':
                cur_interval = [int(json_data['query'][3]), int(json_data['query'][4])] if flag_interval else [int(json_data['query'][3])]
            else:
                cur_interval = nodes[data_idx - len(nodes), 3:] if data_idx >= len(nodes) else nodes[data_idx, 3:]
            
            # Prepare the output of the function
            query_time[data_idx] = cur_interval

            # Check if the output is empty
            if len(cur_output[cur_rel]) == 0:
                continue
            
            # Calculate the probabilities
            cur_probs = self._create_probs_dict(cur_output[cur_rel], cur_interval, pattern_ls[cur_rel], ts_stat_ls[cur_rel], te_stat_ls[cur_rel], 
                                                timestamp_range, flag_rule_split, mode, flag_time_shift, probs_normalization, flag_interval)
            
            # Check if the probabilities are empty
            if cur_probs is None:
                continue
            
            # Prepare the output of the function
            probs[data_idx] = cur_probs
        
            if write_to_file:
                output = [query_time[data_idx], cur_probs]
                with open(path_folder + output_filename, 'w') as f:
                    json.dump(output, f)

        return [query_time, probs]


    def prepare_random_walk_res(self, option, data, mode, num_workers, show_tqdm, rewriting, flag_interval):
        '''
        Prepare the random walk results.

        Parameters:
            option: dict, options
            data: dict, data
            mode: str, mode
            num_workers: int, number of workers
            show_tqdm: bool, whether to show the progress bar
            rewriting: bool, whether to rewrite the results
            flag_interval: bool, whether to use interval

        Returns:
            output_probs: dict, output probabilities with reference edges
        '''
        idx_ls = data['{}_idx_ls'.format(mode.lower())]
        idx_pieces = split_list_into_batches(idx_ls, num_batches=num_workers)
        outputs = Parallel(n_jobs=num_workers)(delayed(self.create_TEKG_in_batch)(option, data, one_piece, mode, show_tqdm, rewriting, flag_interval) 
                                               for one_piece in idx_pieces)

        output_probs = {}
        for output in outputs:
            output_probs.update(output)
        return output_probs


    def create_TEKG_in_batch(self, option, data, idx_ls, mode, show_tqdm, rewriting, flag_interval):
        '''
        Create the TEKG in batch.

        Parameters:
            option: dict, options
            data: dict, data
            idx_ls: list, indices
            mode: str, mode
            show_tqdm: bool, whether to show the progress bar
            rewriting: bool, whether to rewrite the results
            flag_interval: bool, whether to use interval

        Returns:
            output: dict, output probabilities with reference edges
        '''
        path = data['walk_res_path']
        dataset_name = data['short_name']
        nodes = np.vstack((data['train_nodes'], data['valid_nodes'], data['test_nodes']))
        timestamp_range = data['timestamp_range']

        pattern_ls = data['pattern_ls']
        ts_stat_ls = data['ts_stat_ls']
        te_stat_ls = data['te_stat_ls']

        write_to_file = True if mode == 'Train' else False

        output = self.prepare_inputs(nodes=nodes, res_path=path, 
                                     dataset=dataset_name, 
                                     idx_ls=idx_ls,
                                     pattern_ls=pattern_ls, 
                                     timestamp_range=timestamp_range, 
                                     ts_stat_ls=ts_stat_ls, 
                                     te_stat_ls=te_stat_ls,
                                     targ_rel=None,
                                     mode=mode,
                                     flag_rule_split=option.flag_ruleLen_split_ver,
                                     show_tqdm=show_tqdm, write_to_file=write_to_file,
                                     rewriting=rewriting,
                                     flag_interval=flag_interval)
        return output[-1]


    def process_TEILP_results(self, res_dict, capture_dur_only=False, selected_rel=None, known_edges=None, flag_interval=True):
        '''
        Read the results of the TEILP model and extract the relevant information.
        This is for interval dataset only.
        To accelerate, we can select path for the given rules.

        Parameters:
            res_dict: dict, results
            capture_dur_only: bool, whether to capture duration only
            selected_rel: int, selected relation
            known_edges: np.array, known edges

        Returns:
            output: dict, output probabilities with reference edges
        '''
        
        targ_rel = res_dict['query'][1]
        targ_time = res_dict['query'][3:]
        notation_invalid = 9999
        output = {}
        if selected_rel is not None and targ_rel != selected_rel:
            return output

        output[targ_rel] = {}
        for rlen in [k for k in res_dict.keys() if k != 'query']:
            for walk in res_dict[rlen]:
                if known_edges is not None:
                    # mask the time of the edges that are not in the known_edges
                    for i in range(int(rlen)):
                        if flag_interval:
                            if [walk[4*i:4*i+5][num] for num in [0,1,4,2,3]] not in known_edges.tolist():
                                walk[4*i+2:4*i+4] = [9999, 9999]
                        else:
                            if [walk[3*i:3*i+4][num] for num in [0,1,3,2]] not in known_edges.tolist():
                                walk[3*i+2:3*i+3] = 9999

                # collect the information of the walk
                rel_ls = [str(walk[4*i+1]) for i in range(int(rlen))] if flag_interval else [str(walk[3*i+1]) for i in range(int(rlen))]
                time_ls = [[9999, 9999]] + [walk[4*i+2:4*i+4] for i in range(int(rlen))] if flag_interval else [[9999]] + [walk[3*i+2:3*i+3] for i in range(int(rlen))]
                TR_ls = [calculate_TR(time_ls[i], time_ls[i+1]) for i in range(int(rlen))]
                edge_ls = [walk[4*i:4*i+5] for i in range(int(rlen))] if flag_interval else [walk[3*i:3*i+4] for i in range(int(rlen))]
                
                rPattern = ' '.join(rel_ls + TR_ls)
                if rPattern not in output[targ_rel]:
                    output[targ_rel][rPattern] = {}
                    if capture_dur_only:
                        output[targ_rel][rPattern]['dur_with_ref'] = []
                    else:                
                        output[targ_rel][rPattern].update({'time_gap':[], 'edge_ls':[]})

                if capture_dur_only:
                    dur_with_ref = [cal_timegap(targ_time[1], targ_time[0]), time_ls[1], time_ls[-1]]
                    output[targ_rel][rPattern]['dur_with_ref'].append(dur_with_ref)
                else:
                    if flag_interval:                
                        time_gap = [cal_timegap(targ_time[0], time_ls[1][0], notation_invalid), cal_timegap(targ_time[0], time_ls[-1][0], notation_invalid),
                                    cal_timegap(targ_time[0], time_ls[1][1], notation_invalid), cal_timegap(targ_time[0], time_ls[-1][1], notation_invalid),
                                    cal_timegap(targ_time[1], time_ls[1][0], notation_invalid), cal_timegap(targ_time[1], time_ls[-1][0], notation_invalid),
                                    cal_timegap(targ_time[1], time_ls[1][1], notation_invalid), cal_timegap(targ_time[1], time_ls[-1][1], notation_invalid)]
                    else:
                        time_gap = [cal_timegap(targ_time[0], time_ls[1][0], notation_invalid), cal_timegap(targ_time[0], time_ls[-1][0], notation_invalid)]
                    output[targ_rel][rPattern]['time_gap'].append(time_gap)
                    output[targ_rel][rPattern]['edge_ls'].append(edge_ls)


        if selected_rel is not None:
            output = output[selected_rel]
        return output


    def process_TEILP_results_in_batch(self, path, file_ls, flag_capture_dur_only, rel_batch, known_edges, stat_res_path, flag_interval=True, num_rules_preserved=1000):
        '''
        Find the most frequent rules for each relation. Calcluate the mean and std of the time gap for each rule.

        Parameters:
            path: str, path
            file_ls: list, list of files
            flag_capture_dur_only: bool, whether to capture duration only
            rel_batch: list, list of relations
            known_edges: np.array, known edges
            stat_res_path: str, path of the statistics results
            num_rules_preserved: int, number of rules preserved

        Returns:
            None
        '''
        for rel in rel_batch:
            output = {}
            for file in tqdm(file_ls, desc='Rule summary for relation ' + str(rel)):
                with open(path + '/' + file, 'r') as f:
                    data = json.loads(f.read())
                
                res = self.process_TEILP_results(data, capture_dur_only=flag_capture_dur_only, selected_rel=rel, known_edges=known_edges, flag_interval=flag_interval)
               
                for rule in res:
                    if rule not in output:
                        # using online algorithm to accelerate the calculation
                        output[rule] = OnlineStatsVector(8) if flag_interval else OnlineStatsVector(2) 
                    for time_gap in res[rule]['time_gap']:
                        time_gap = map(lambda x: np.nan if x == 9999 else x, time_gap)
                        time_gap = list(time_gap)
                        output[rule].update(time_gap)

            # preserve the most frequent k rules for each relation    
            num_samples_dict = []
            for rule in output:
                num_samples_dict.append([rule, max(output[rule].n)]) 
                num_samples_dict.sort(key=lambda x: x[1], reverse=True)
                num_samples_dict = num_samples_dict[:num_rules_preserved]
            
            # Todo: for duration calculation
            # if flag_capture_dur_only:
            #    pass

            output_stat = {}
            for rule in num_samples_dict:
                rule = rule[0]
                output_stat[rule] = {'num_samples': output[rule].n.astype(int).tolist(), 
                                     'mean': np.around(output[rule].get_mean(), decimals=6).tolist(), 
                                     'std': np.around(output[rule].get_std(), decimals=6).tolist()}

            # save the results
            with open(stat_res_path + "_rel_" + str(rel) + ".json", "w") as f:
                json.dump(output_stat, f)

        return






class Option(object):
    def __init__(self, d):
        self.__dict__ = d
    def save(self):
        with open(os.path.join(self.this_expsdir, "option.txt"), "w") as f:
            for key, value in sorted(self.__dict__.items(), key=lambda x: x[0]):
                f.write("%s, %s\n" % (key, str(value)))




class OnlineStatsVector:
    '''
    Class to calculate the mean and variance of a vector of values using the online algorithm.
    '''
    def __init__(self, vector_length):
        self.n = np.zeros(vector_length)
        self.mean = np.zeros(vector_length)
        self.M2 = np.zeros(vector_length)
        # self.hist = []

    def update(self, x):
        x = np.array(x, dtype=float)
        mask = ~np.isnan(x)
        delta = np.zeros_like(x)
        delta2 = np.zeros_like(x)
        # self.hist.append(x)
        for i in range(len(x)):
            if mask[i]:
                self.n[i] += 1
                delta[i] = x[i] - self.mean[i]
                self.mean[i] += delta[i] / self.n[i]
                delta2[i] = x[i] - self.mean[i]
                self.M2[i] += delta[i] * delta2[i]
    
    def get_mean(self):
        return self.mean
    
    def get_variance(self):
        variance = np.zeros_like(self.mean)
        for i in range(len(self.n)):
            if self.n[i] > 1:
                variance[i] = self.M2[i] / self.n[i]
        return variance
    
    def get_std(self):
        return np.sqrt(self.get_variance())




class Rule_summarizer(Data_Processor):
    def convert_walks_into_rules(self, path, dataset, idx_ls=None, flag_time_shift=False, 
                                flag_capture_dur_only=False, rel=None, known_edges=None, flag_few_training=False,
                                ratio=None, imbalanced_rel=None, flag_biased=False, exp_idx=None, targ_rel_ls=None,
                                num_processes=20, stat_res_path='', flag_interval=True):
        '''
        This function is for interval dataset only.
        '''
        file_ls, rel = obtain_walk_file_ls(path, dataset, idx_ls, ratio, imbalanced_rel, flag_biased, exp_idx)
        targ_rel_ls = rel if rel is not None else targ_rel_ls

        if targ_rel_ls is None:
            return
        
        num_processes = min(num_processes, len(targ_rel_ls))
        rel_batch_ls = split_list_into_batches(targ_rel_ls, num_batches=num_processes)

        Parallel(n_jobs=num_processes)(
            delayed(self.process_TEILP_results_in_batch)(path, file_ls, flag_capture_dur_only, rel_batch, known_edges, stat_res_path, flag_interval) 
                                                        for rel_batch in rel_batch_ls)
        
        return