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




class Data_preprocessor():
    def _prepare_basic_info(self, data, option):
        dataset_index = ['wiki', 'YAGO', 'icews14', 'icews05-15', 'gdelt100'].index(option.dataset)

        data['path'] = '../output/walk_res/' if not option.shift else '../output/walk_res_time_shift/'
        
        data['dataset'] = ['WIKIDATA12k', 'YAGO11k', 'icews14', 'icews05-15', 'gdelt100'][dataset_index]
        if option.shift:
            data['dataset'] = 'difficult_settings/{}_time_shifting'.format(data['dataset'])

        data['dataset_name'] = ['wiki', 'YAGO', 'icews14', 'icews05-15', 'gdelt100'][dataset_index]

        data['num_rel'] = [48, 20, 460, 502, 40][dataset_index]
        data['num_TR'] = 4

        # Note that entity in TEKG is event in TKG.
        # Originally we use 40000 as num_entity for global graph which is time-consuming.
        # Now we use dynamic value for local graph.
        # data['num_entity'] = [40000, 40000, 40000, 40000, 40000][dataset_index]
         
        return dataset_index
    

    def _trainig_data_sampling(self, train_edges, num_rel, num_sample_per_rel=-1):
        pos_examples_idx = np.array(list(range(len(train_edges))))
        if num_sample_per_rel > 0:
            pos_examples_idx_sample = []
            for rel_idx in range(num_rel):
                cur_pos_examples_idx = pos_examples_idx[train_edges[:, 1] == rel_idx]
                np.random.shuffle(cur_pos_examples_idx)
                pos_examples_idx_sample.append(cur_pos_examples_idx[:num_sample_per_rel])

            pos_examples_idx_sample = np.hstack(pos_examples_idx_sample)
            pos_examples_idx = pos_examples_idx_sample

        pos_examples_idx = pos_examples_idx.tolist()
        return pos_examples_idx


    def _prepare_edges(self, data, option, dataset_index):
        data['train_edges'], data['valid_edges'], data['test_edges'] = self.obtain_all_data(data['dataset'], shuffle_train_set=False)
        data['timestamp_range'] = [np.arange(-3, 2024, 1), np.arange(-431, 2024, 1), np.arange(0, 366, 1), np.arange(0, 4017, 1), np.arange(90, 456, 1)][dataset_index]
        data['num_samples_dist'] = [[32497, 4062, 4062], [16408, 2050, 2051], [72826, 8941, 8963], [368962, 46275, 46092], [390045, 48756, 48756]][dataset_index]


        if option.dataset == 'gdelt100' and option.shift:
            with open('../data/gdelt100_sparse_edges.json') as json_file:
                edges = json.load(json_file)

            num_train = [16000, 2000]
            edges = np.array(edges)

            data['train_edges'] = edges[:num_train[0]]
            data['valid_edges'] = edges[num_train[0]:num_train[0] + num_train[1]]
            data['test_edges'] = edges[num_train[0] + num_train[1]:]

            data['num_samples_dist'] = [16000, 2000, 2000]

        data['train_idx_ls'] = self._trainig_data_sampling(data['train_edges'], data['num_rel'], num_sample_per_rel=option.num_samples_per_rel)
        data['valid_idx_ls'] = list(range(data['num_samples_dist'][0], data['num_samples_dist'][0] + data['num_samples_dist'][1]))
        data['test_idx_ls'] = list(range(data['num_samples_dist'][0] + data['num_samples_dist'][1], np.sum(data['num_samples_dist'])))

        
        if option.shift:
            data['train_idx_ls'] = [idx + np.sum(data['num_samples_dist']) for idx in data['train_idx_ls']]
            data['valid_idx_ls'] = [idx + np.sum(data['num_samples_dist']) for idx in data['valid_idx_ls']]
            data['test_idx_ls'] = [idx + np.sum(data['num_samples_dist']) for idx in data['test_idx_ls']]

        if option.dataset in ['wiki', 'YAGO']:
            with open('../data/{}_time_pred_eval_rm_idx.json'.format(data['dataset_name']), 'r') as file:
                data['rm_ls'] = json.load(file)
            if option.shift:
                with open('../data/{}_time_pred_eval_rm_idx_shift_mode.json'.format(data['dataset_name']), 'r') as file:
                    data['rm_ls'] = json.load(file)

        # remove test samples with unknown time      
        data['test_idx_ls'] = [idx for idx in data['test_idx_ls'] if (idx - data['num_samples_dist'][0] - data['num_samples_dist'][1]) not in data['rm_ls']]


    def _prepare_stat_res(self, data, option, dataset_index):
        num_rel = data['num_rel']//2 if not option.shift else data['num_rel']  # known time range change

        data['pattern_ls'], data['ts_stat_ls'], data['te_stat_ls'], data['stat_res'] = self.processing_stat_res(data['dataset_name'], num_rel, flag_time_shifting=option.shift)
        
        # relations w/o duration
        data['rel_ls_no_dur'] = [[4, 16, 17, 20], [0, 7], None, None, None][dataset_index]

        # We do not create the whole graph in advance which is time-consuming.
        data['connectivity_rel'], data['connectivity_TR'], data['TEKG_nodes'] = None, None, None

        # Todo: use the duration information for learning
        if hasattr(option, 'flag_use_dur') and option.flag_use_dur:
            with open("../output/{}_dur_preds.json".format(data['dataset_name']), "r") as json_file:
                data['pred_dur'] = json.load(json_file)


    def prepare_data(self, option, save_option=True, process_walk_res=True):
        data = {}
        dataset_index = self._prepare_basic_info(data, option)
        self._prepare_edges(data, option, dataset_index)
        self._prepare_stat_res(data, option, dataset_index)

        if save_option:
            option.this_expsdir = os.path.join(option.exps_dir, '{}_{}'.format(data['dataset_name'], option.tag))
            if not os.path.exists(option.this_expsdir):
                os.makedirs(option.this_expsdir)
            option.ckpt_dir = os.path.join(option.this_expsdir, "ckpt")
            if not os.path.exists(option.ckpt_dir):
                os.makedirs(option.ckpt_dir)
            option.model_path = os.path.join(option.ckpt_dir, "model")

            option.num_step = [4, 6, 4, 4, 4][dataset_index]
            option.num_rule = [1000, 1000, 1000, 1000, 1000][dataset_index]  # for each relation
            option.flag_interval = True if dataset_index in [0, 1] else False

            option.savetxt = '{}/intermediate_res.txt'.format(option.this_expsdir)
            option.save()
            print("Option saved.")


        data['random_walk_res'] = None
        if process_walk_res:
            if dataset_index in [0, 1] and option.train:
                # Once this step is done, we recommend to save the results without rewriting unless the settings are changed.
                self.prepare_graph_random_walk_res(option, data, 'Train', num_workers=20, show_tqdm=True, rewriting=True)
            print("Data prepared.")

        return data


    def obtain_all_data(self, dataset, shuffle_train_set=True, flag_use_valid=1):
        edges1 = read_dataset_txt('../data/{}/train.txt'.format(dataset))
        edges1 = np.array(edges1)

        if shuffle_train_set:
            edges1 = my_shuffle(edges1)

        edges2 = None
        if flag_use_valid:
            edges2 = read_dataset_txt('../data/{}/valid.txt'.format(dataset))
            edges2 = np.array(edges2)

        edges3 = read_dataset_txt('../data/{}/test.txt'.format(dataset))
        edges3 = np.array(edges3)

        return edges1, edges2, edges3


    def processing_stat_res(self, dataset, num_rel, flag_time_shifting=0):
        path_suffix = '_' if not flag_time_shifting else '_time_shifting_'
        pattern_ls, ts_stat_ls, te_stat_ls, stat_res = {}, {}, {}, {}
        for rel in range(num_rel):
            pattern_ls[rel], ts_stat_ls[rel], te_stat_ls[rel], stat_res[rel] = [], [], [], []

            # read results
            path = '../output/{}{}/{}{}stat_res_rel_{}.json'.format(dataset, path_suffix[:-1], dataset, path_suffix, rel)
            if not os.path.exists(path):
                continue
            with open(path, "r") as f:
                res = json.load(f)

            if dataset in ['icews14', 'icews05-15', 'gdelt100']:
                stat_res[rel] = res
                pattern_path = '../output/{}{}/{}{}pattern_ls_rel_{}.json'.format(dataset, path_suffix[:-1], dataset, path_suffix, rel)
                if os.path.exists(pattern_path):
                    with open(pattern_path, "r") as f:
                        pattern_ls[rel] = json.load(f)
                continue
            
            # Below is for interval datasets
            for rule in res:
                if len(res[rule].keys()) == 0:
                    continue

                pattern_ls[rel].append(rule)

                # res: time_gap_ts_ref_ts + time_gap_ts_ref_te + time_gap_te_ref_ts + time_gap_te_ref_te
                # cur_ts_stat_ls: 'ts_first_event_ts', 'ts_first_event_te', 'ts_last_event_ts', 'ts_last_event_te'
                # cur_te_stat_ls: 'te_first_event_ts', 'te_first_event_te', 'te_last_event_ts', 'te_last_event_te'
                for k in res[rule]:
                    res[rule][k] = np.array(res[rule][k])

                # remove invalid mean and std values
                res[rule]['mean'][res[rule]['num_samples'] == 0] = 9999
                res[rule]['std'][res[rule]['num_samples'] == 0] = 9999
                res[rule]['std'][res[rule]['std'] < 5] = 5

                cur_ts_stat_ls = [item for pair in zip(res[rule]['mean'][[0, 2, 1, 3]], res[rule]['std'][[0, 2, 1, 3]]) for item in pair]
                cur_te_stat_ls = [item for pair in zip(res[rule]['mean'][[4, 6, 5, 7]], res[rule]['std'][[4, 6, 5, 7]]) for item in pair]

                ts_stat_ls[rel].append(cur_ts_stat_ls)
                te_stat_ls[rel].append(cur_te_stat_ls)

        return pattern_ls, ts_stat_ls, te_stat_ls, stat_res


    def _update_stat_res(self, walk, stat_res, samples_edges, mode, TR_ls, flag_interval):
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

        e = tuple(e)
        if e not in samples_edges:
            samples_edges[e] = {}
        if rule_pattern not in samples_edges[e]:
            samples_edges[e][rule_pattern] = []

        samples_edges[e][rule_pattern].append(cur_ref_edge)


    def _prepare_data(self, data1, file_paths, rel):
        if data1 is None:
            data = {}
            for file_path in file_paths:
                if not '_rel_{}'.format(str(rel)) in file_path:
                    continue
                with open(file_path) as f:
                    data.update(json.load(f))
        else:
            data = data1
        return data


    def _format_stat_res(self, stat_res, rule_pattern, flag_plot):
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


    def read_random_walk_results(self, dataset, rel_ls, file_paths, file_suffix, data1=None, flag_interval=True, flag_plot=False, mode=None, flag_time_shifting=False):
        output_dir = '../output/{}/'.format(dataset) if not flag_time_shifting else '../output/{}_time_shifting/'.format(dataset)
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)
            os.mkdir('{}samples'.format(output_dir))

        for rel in rel_ls:
            samples_edges, stat_res = {}, {}
            data = self._prepare_data(data1, file_paths, rel)

            if len(data) == 0:
                continue

            for rule_length in [1,2,3,4,5]:
                if str(rule_length) not in data:
                    continue
                for TR_ls in data[str(rule_length)]:
                    for walk in data[str(rule_length)][TR_ls]:
                        self._update_stat_res(walk, stat_res, samples_edges, mode, TR_ls, flag_interval)
            
            if mode == 'Train':
                for rule_pattern in stat_res:
                    self._format_stat_res(stat_res, rule_pattern, flag_plot)
     
                with open('{}{}{}stat_res_rel_{}_.json'.format(output_dir, dataset, file_suffix, rel, mode), 'w') as f:
                    json.dump(stat_res, f)

                for e in samples_edges:
                    with open("{}samples/{}{}{}_sample_{}.json".format(output_dir, dataset, file_suffix, mode, e), "w") as f:
                        json.dump(convert_dict({e: samples_edges[e]}), f)

                rule_nums = rule_num_stat(samples_edges)

                if rel not in rule_nums:
                    pattern_ls = []
                else:
                    pattern_ls = sorted(rule_nums[rel], key=lambda p: rule_nums[rel][p], reverse=True)[:1000]
                    random.shuffle(pattern_ls)

                with open("{}{}{}pattern_ls_rel_{}.json".format(output_dir, dataset, file_suffix, rel), 'w') as file:
                    json.dump(pattern_ls, file)

        return


    def process_random_walk_results_dist_ver(self, dataset, mode, num_rel, file_paths, num_workers=10):
        index_pieces = split_list_into_batches(range(num_rel), num_batches=num_workers)

        data = {}
        for file_path in file_paths:
            with open(file_path) as f:
                data.update(json.load(f))

        Parallel(n_jobs=num_workers)(delayed(self.read_random_walk_results)(dataset, piece, data=data, flag_interval=False, 
                                                        flag_plot=False, mode=mode) for piece in index_pieces)
        return


    def _calculate_time_prob_dist(self, edge, timestamp_range, targ_interval, stat_ls, idx_ls, mode, flag_time_shift=False, probs_normalization=True):
        idx_ts_or_te, idx_first_or_last_event, idx_ref_time_ts_or_te = idx_ls
        idx_prob_ls = 4*idx_ts_or_te + 2*idx_first_or_last_event + idx_ref_time_ts_or_te

        refTime = edge[2:4][idx_ref_time_ts_or_te]   # edge: (s, r, ts, te, o)
        delta_t = timestamp_range - refTime
        targ_time = targ_interval[idx_ts_or_te]

        Gau_mean = stat_ls[idx_prob_ls*2]
        Gau_std = stat_ls[idx_prob_ls*2+1]
        
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


    def _calculate_distribution(self, data):
        '''
        Given a rule, count the distribution of the first and last events.
        '''
        first_position_counter = Counter()
        second_position_counter = Counter()

        for event_ls in data:
            first_event, last_event = tuple(event_ls[0]), tuple(event_ls[1])
            # remove unknown time events
            if first_event[2] != 9999 and first_event[3] != 9999:  # (s, r, ts, te, o)
                first_position_counter[first_event] += 1
    
            if last_event[2] != 9999 and last_event[3] != 9999:    
                second_position_counter[last_event] += 1
        
        total_first = sum(first_position_counter.values())
        total_second = sum(second_position_counter.values())
        
        first_position_distribution = {k: v*1. / total_first for k, v in first_position_counter.items()}
        second_position_distribution = {k: v*1. / total_second for k, v in second_position_counter.items()}

        return [first_position_distribution, second_position_distribution]


    def prepare_inputs_interval_ver(self, path, dataset, edges, idx_ls, targ_rel, num_samples_dist, pattern_ls, timestamp_range, num_rel, 
                                    ts_stat_ls, te_stat_ls, mode=None, rm_ls=None, with_ref_end_time=True, 
                                    only_find_samples_with_empty_rules=False, flag_output_probs_with_ref_edges=False,
                                    flag_acceleration=False, flag_time_shift=False, write_to_file=False, show_tqdm=False,
                                    flag_rule_split=False, rewriting=False, probs_normalization=True):            
        
        input_intervals, probs = {}, {}
        sample_with_empty_rules_ls = []

        if show_tqdm:
            idx_ls = tqdm(idx_ls, desc='Prepare input probs')
        for data_idx in idx_ls:
            if write_to_file:
                input_intervals, probs = {}, {}
                filename = dataset + '_idx_' + str(data_idx) + '_input.json'
                if os.path.exists('../output/process_res/' + filename) and not rewriting:
                    continue
                    
            # read data
            filename = dataset + '_idx_' + str(data_idx) + '.json'
            if not os.path.exists(path + filename):
                sample_with_empty_rules_ls.append(data_idx)
                continue
            with open(path + filename, 'r') as f:
                data = f.read()
                json_data = json.loads(data)

            if mode == 'Test' and rm_ls !=None and (data_idx - num_samples_dist[0] - num_samples_dist[1]) in rm_ls:
                # print(json_data['query'])
                continue
            if targ_rel != None and json_data['query'][1] != targ_rel:
                continue
            

            # process data
            cur_rel = json_data['query'][1]
            cur_output = self.process_TEILP_results(json_data)
            if mode == 'Train':
                cur_interval = [int(json_data['query'][3]), int(json_data['query'][4])]
            else:
                # data_idx -= num_samples_dist[1]
                cur_interval = edges[data_idx - len(edges), 3:] if data_idx >= len(edges) else edges[data_idx, 3:]


            input_intervals[data_idx] = cur_interval


            if len(cur_output[cur_rel]) == 0:
                sample_with_empty_rules_ls.append(data_idx)
                continue

            cur_valid_rules = [p for p in cur_output[cur_rel] if p in pattern_ls[cur_rel]]
            if len(cur_valid_rules) == 0:
                sample_with_empty_rules_ls.append(data_idx)
                continue

            if only_find_samples_with_empty_rules or not flag_output_probs_with_ref_edges:
                continue

            probs[data_idx] = {0:{}, 1:{}}  # [first event, last event]
            for p in cur_valid_rules:
                # for each rule, calculate the probability of the time gap
                p_idx = pattern_ls[cur_rel].index(p)
                ruleLen = len(p.split(' '))//2
                cur_stat_ls = ts_stat_ls[cur_rel][p_idx] + te_stat_ls[cur_rel][p_idx]

                events_for_cur_rule = []
                for walk in cur_output[cur_rel][p]['edge_ls']:
                    events_for_cur_rule.append([walk[idx] for idx in [0, -1]])  # we only consider the first and last event
                
                events_for_cur_rule = self._calculate_distribution(events_for_cur_rule)

                for idx_event_pos in [0, 1]:
                    for edge in events_for_cur_rule[idx_event_pos]:
                        alpha_edge = events_for_cur_rule[idx_event_pos][edge]            
                        if str_tuple(edge) not in probs[data_idx][idx_event_pos]:
                            probs[data_idx][idx_event_pos][str_tuple(edge)] = {i: [] for i in range(4)} if not flag_rule_split else \
                                                                              {i: {rLen: [] for rLen in range(1, 6)} for i in range(4)}

                        for idx_query_time in [0, 1]:
                            for idx_ref_time in [0, 1]:                                
                                flag_success, cur_probs = self._calculate_time_prob_dist(edge, timestamp_range, cur_interval, cur_stat_ls, 
                                                                                         [idx_query_time, idx_event_pos, idx_ref_time], 
                                                                                          mode, flag_time_shift, probs_normalization)
                                if not flag_success:
                                    cur_probs = np.array(1./len(timestamp_range)) if mode == 'Train' else np.array([1./len(timestamp_range) for _ in range(len(timestamp_range))])
                                
                                cur_probs = (cur_probs*alpha_edge).tolist()

                                if flag_acceleration:
                                    cur_probs = [cur_probs, p_idx]

                                if flag_rule_split:
                                    probs[data_idx][idx_event_pos][str_tuple(edge)][2*idx_query_time + idx_ref_time][ruleLen].append(cur_probs)                                    
                                else:
                                    probs[data_idx][idx_event_pos][str_tuple(edge)][2*idx_query_time + idx_ref_time].append(cur_probs)
            # Loop end for one rule pattern

            if len(cur_valid_rules) == 0:
                continue

            if write_to_file:
                output = [input_intervals[data_idx]]
                if flag_output_probs_with_ref_edges:
                    output.append(probs[data_idx])

                filename = dataset + '_idx_' + str(data_idx) + '_input.json'
                with open('../output/process_res/' + filename, 'w') as f:
                    json.dump(output, f)

        # Loop end for one sample

        output_ls = [input_intervals]
        if flag_output_probs_with_ref_edges:
            output_ls.append(probs)

        return output_ls


    def prepare_inputs_timestamp_ver(self, samples, samples_inv, targ_rels, pattern_ls, timestamp_range, num_rel, stat_res, mode=None,
                     flag_only_find_samples_with_empty_rules=False, dataset=None, file_suffix='', 
                     flag_compression=True, flag_write=False, flag_time_shifting=False, pattern_ls_fkt=None, stat_res_fkt=None,
                     shift_ref_time=None, flag_rm_seen_timestamp=False):
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
            
            dim = 1 if mode == 'Train' else len(timestamp_range)
            type_ref_time = ['ts_first_event', 'ts_last_event']

            cur_edge_probs_first_ref, cur_edge_probs_last_ref = {}, {}
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

                            prob_ls[idx] = cur_probs
                            prob_ls[idx] = cut_vector(prob_ls[idx], len(samples[e_str][p]))
                            if flag_time_shifting and flag_rm_seen_timestamp:
                                prob_ls[idx] = rm_seen_timestamp(prob_ls[idx], timestamp_range, max_ref_time, [time_gap, mean_ls, std_ls, prop_ls, relaxed_std])

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

            cnt += 1
            cur_edge_probs = {'input_edge_probs_first_ref': convert_dict(cur_edge_probs_first_ref), 
                              'input_edge_probs_last_ref': convert_dict(cur_edge_probs_last_ref),
                              'input_edge_probs_first_ref_inv_rel': convert_dict(cur_edge_probs_first_ref_inv_rel), 
                              'input_edge_probs_last_ref_inv_rel': convert_dict(cur_edge_probs_last_ref_inv_rel)}

            edge_probs.append(cur_edge_probs)

            if flag_write:
                with open("output/"+ dataset + '/' + dataset +"_"+ mode +"_input_edge_probs_edge_"+ str(e) + file_suffix + ".json", "w") as json_file:
                    json.dump(cur_edge_probs, json_file)

            # Loop end for one sample
        # Loop end for all samples

        return edge_probs


    def prepare_graph_random_walk_res_int_ver(self, option, data, mode, num_workers, file_suffix, show_tqdm, rewriting):
        idx_ls = data['train_idx_ls'] if mode == 'Train' else data['test_idx_ls']
        idx_pieces = split_list_into_batches(idx_ls, num_batches=num_workers)
        outputs = Parallel(n_jobs=num_workers)(delayed(self.create_TEKG_in_batch)(option, data, one_piece, mode, show_tqdm, rewriting) for one_piece in idx_pieces)

        output_probs_with_ref_edges = {}
        for output in outputs:
            output_probs_with_ref_edges.update(output)
        return output_probs_with_ref_edges


    def prepare_graph_random_walk_res_timestamp_ver(self, option, data, mode, num_workers, file_suffix, show_tqdm, rewriting):
        timestamp_range = data['timestamp_range']
        num_rel = data['num_rel']
        dataset = data['dataset_name']

        path1 = 'output/' + dataset + '/' + dataset + "_"+ mode + "_samples_edge"+ file_suffix + ".json"
        path2 = 'output/' + dataset + '/' + dataset + "_"+ mode + "_samples_edge_rel_0"+ file_suffix + ".json"
        if (not os.path.exists(path1)) and (not os.path.exists(path2)):
            if dataset in ['icews14', 'icews05-15']:
                if mode == 'Train':
                    file_paths = ['../output/'+ dataset + '/'+ dataset + '_train_walks_suc_with_TR.json']
                else:
                    file_paths = []
                    k_ls = [0,1,2,3] if dataset == 'icews14' else [0,1,2] + ['3_split'+str(i) for i in range(10)]
                    for k in k_ls:
                        file_paths.append('../output/'+ dataset + '/'+ dataset + '_test_samples_part'+ str(k) +'.json')
            elif dataset in ['gdelt']:
                if mode == 'Train':
                    file_paths = ['../output/gdelt_train_walks_suc_with_TR.json']
                else:
                    file_paths = ['../output/gdelt_test_samples.json']

            self.process_random_walk_results_dist_ver(dataset, mode = mode, num_rel=num_rel, file_paths=file_paths, file_suffix=file_suffix, num_workers=num_workers)


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
        idx_pieces = split_list_into_batches(rel_ls, num_batches=num_workers)
        Parallel(n_jobs=num_workers)(delayed(self.prepare_inputs_timestamp_ver)\
                                            (samples, one_piece, pattern_ls, timestamp_range, 
                                                num_rel//2, stat_res, mode=mode,
                                                flag_only_find_samples_with_empty_rules=False, dataset=dataset) for one_piece in idx_pieces)


    def prepare_graph_random_walk_res(self, option, data, mode, num_workers=20, file_suffix='', show_tqdm=True, rewriting=False):
        if data['dataset_name'] in ['wiki', 'YAGO']:
            output_probs_with_ref_edges = self.prepare_graph_random_walk_res_int_ver(option, data, mode, num_workers, file_suffix, show_tqdm, rewriting)
            return output_probs_with_ref_edges
        else:
            self.prepare_graph_random_walk_res_timestamp_ver(option, data, mode, num_workers, file_suffix, show_tqdm, rewriting)
            return


    def create_TEKG_in_batch(self, option, data, idx_ls, mode, show_tqdm, rewriting):
        path = data['path']
        dataset = data['dataset']
        dataset_name = data['dataset_name']

        edges = np.vstack((data['train_edges'], data['test_edges']))
        num_samples_dist = data['num_samples_dist']
        timestamp_range = data['timestamp_range']

        num_rel = data['num_rel']
        # num_entity = data['num_entity']

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

        write_to_file = True if mode == 'Train' else False

        output = self.prepare_inputs_interval_ver(edges=edges, path=path, 
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
                                                    flag_acceleration=flag_acceleration,
                                                    flag_rule_split=option.flag_ruleLen_split_ver,
                                                    show_tqdm=show_tqdm, write_to_file=write_to_file,
                                                    rewriting=rewriting)
        return output[-1]


    def process_TEILP_results(self, res_dict, with_ref_end_time=False, capture_dur_only=False, selected_rel=None, known_edges=None):
        '''
        Read the results of the TEILP model and extract the relevant information.
        This is for interval dataset only.
        To accelerate, we can select path for the given rules.
        To simplify the code, we set with_ref_end_time=True by default.
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
                        if [walk[4*i:4*i+5][num] for num in [0,1,4,2,3]] not in known_edges.tolist():
                            walk[4*i+2:4*i+4] = [9999, 9999]

                # collect the information of the walk
                rel_ls = [str(walk[4*i+1]) for i in range(int(rlen))]
                time_ls = [[9999, 9999]] + [walk[4*i+2:4*i+4] for i in range(int(rlen))]
                TR_ls = [calculate_TR(time_ls[i], time_ls[i+1]) for i in range(int(rlen))]
                edge_ls = [walk[4*i:4*i+5] for i in range(int(rlen))]
                
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
                    time_gap = [cal_timegap(targ_time[0], time_ls[1][0], notation_invalid), cal_timegap(targ_time[0], time_ls[-1][0], notation_invalid),
                                cal_timegap(targ_time[0], time_ls[1][1], notation_invalid), cal_timegap(targ_time[0], time_ls[-1][1], notation_invalid),
                                cal_timegap(targ_time[1], time_ls[1][0], notation_invalid), cal_timegap(targ_time[1], time_ls[-1][0], notation_invalid),
                                cal_timegap(targ_time[1], time_ls[1][1], notation_invalid), cal_timegap(targ_time[1], time_ls[-1][1], notation_invalid)]
                    output[targ_rel][rPattern]['time_gap'].append(time_gap)
                    output[targ_rel][rPattern]['edge_ls'].append(edge_ls)


        if selected_rel is not None:
            output = output[selected_rel]
        return output


    def process_TEILP_results_in_batch(self, path, file_ls, with_ref_end_time, flag_capture_dur_only, rel_batch, known_edges, stat_res_path, num_rules_preserved=1000):
        '''
        Find the most frequent rules for each relation. Calcluate the mean and std of the time gap for each rule.
        '''
        for rel in rel_batch:
            output = {}
            for file in tqdm(file_ls, desc='Rule summary for relation ' + str(rel)):
                with open(path + '/' + file, 'r') as f:
                    data = json.loads(f.read())
                
                res = self.process_TEILP_results(data, with_ref_end_time=with_ref_end_time, capture_dur_only=flag_capture_dur_only, selected_rel=rel, known_edges=known_edges)
               
                for rule in res:
                    if rule not in output:
                        output[rule] = OnlineStatsVector(8)  # using online algorithm to accelerate the calculation
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
            #    ....

            output_stat = {}
            for rule in num_samples_dict:
                rule = rule[0]
                output_stat[rule] = {'num_samples': output[rule].n.astype(int).tolist(), 'mean': np.around(output[rule].get_mean(), decimals=6).tolist(), 'std': np.around(output[rule].get_std(), decimals=6).tolist()}

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




class Rule_summarizer(Data_preprocessor):
    def convert_walks_into_rules(self, path, dataset, idx_ls=None, with_ref_end_time=True, flag_time_shift=0, 
                                flag_capture_dur_only=False, rel=None, known_edges=None, flag_few_training=0,
                                ratio=None, imbalanced_rel=None, flag_biased=0, exp_idx=None, targ_rel_ls=None,
                                num_processes=20, stat_res_path=''):

        file_ls, rel = obtain_walk_file_ls(path, dataset, idx_ls, ratio, imbalanced_rel, flag_biased, exp_idx)
        targ_rel_ls = rel if rel is not None else targ_rel_ls

        if targ_rel_ls is None:
            return
        
        num_processes = min(num_processes, len(targ_rel_ls))
        rel_batch_ls = split_list_into_batches(targ_rel_ls, num_batches=num_processes)

        Parallel(n_jobs=num_processes)(
            delayed(self.process_TEILP_results_in_batch)(path, file_ls, with_ref_end_time, flag_capture_dur_only, rel_batch, known_edges, stat_res_path) for rel_batch in rel_batch_ls
            )
        
        return