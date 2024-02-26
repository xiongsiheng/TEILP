import sys
import os
import time
import pickle
from collections import Counter
import numpy as np
import itertools
from utils import *


class Experiment():
    def __init__(self, sess, saver, option, learner, data):
        self.sess = sess
        self.saver = saver
        self.option = option
        self.learner = learner
        self.data = data

        self.msg_with_time = lambda msg: \
                "%s Time elapsed %0.2f hrs (%0.1f mins)" \
                % (msg, (time.time() - self.start) / 3600., 
                        (time.time() - self.start) / 60.)

        self.start = time.time()
        self.epoch = 0
        self.best_valid_loss = np.inf
        self.train_stats = []
        self.valid_stats = []
        self.test_stats = []
        self.early_stopped = False
        self.log_file = open(os.path.join(self.option.this_expsdir, "log.txt"), "w")

        if self.data['dataset_name'] in ['wiki', 'YAGO']:
            self.metrics = ['aeIOU', 'TAC']
        else:
            self.metrics = ['MAE']


    def one_epoch(self, mode, total_idx=None):
        epoch_loss = []
        epoch_eval_aeIOU = []
        epoch_eval_TAC = []
        epoch_eval_MAE = []

        batch_size = 32

        if mode == "Train":
            random.shuffle(self.data['train_idx_ls'])
            idx_ls = split_list_into_batches(self.data['train_idx_ls'], batch_size)
        elif mode == "Test":
            test_idx_ls = total_idx
            if test_idx_ls is None:
                test_idx_ls = self.data['test_idx_ls']
            idx_ls = split_list_into_batches(test_idx_ls, batch_size)

        save_data(self.option.savetxt, len(idx_ls))

        timestamp_range = self.data['timestamp_range']

        flag_rm_seen_ts = False # rm seen ts in training set
        if self.data['dataset_name'] in ['icews14', 'icews05-15', 'gdelt'] and not self.option.shift:
            flag_rm_seen_ts = True
            train_edges = self.data['train_edges']

        if self.option.flag_use_dur:
            pred_dur_dict = self.data['pred_dur']

        for (i, batch_idx_ls) in enumerate(idx_ls):
            if self.option.flag_acceleration:
                if mode == "Train":
                    run_fn = self.learner.update_acc
                else:
                    run_fn = self.learner.predict_acc
            else:
                if mode == "Train":
                    run_fn = self.learner.update
                else:
                    run_fn = self.learner.predict


            if self.option.flag_acceleration:
                qq, query_rels, refNode_source, res_random_walk, probs, valid_sample_idx, input_intervals, input_samples, ref_time_ls = self.create_TEKG_in_batch(batch_idx_ls, mode)
            else:
                qq, hh, tt, mdb, connectivity, probs, valid_sample_idx, valid_ref_event_idx, input_intervals, input_samples = self.create_TEKG_in_batch(batch_idx_ls, mode)


            save_data(self.option.savetxt, 'TEKG_prepared!')

            input_intervals = np.array(input_intervals)

            # print(len(valid_sample_idx))

            if len(valid_sample_idx) == 0:
                continue

            # print(probs)
            # print(ref_time_ls)


            if self.option.flag_acceleration:
                output = run_fn(self.sess, query_rels, refNode_source, res_random_walk, probs)
            else:
                output = run_fn(self.sess, qq, hh, tt, mdb, connectivity, probs, valid_sample_idx, valid_ref_event_idx)

            save_data(self.option.savetxt, 'model processed!')

            if mode == "Train":
                epoch_loss += list(output)
                save_data(self.option.savetxt, i)
                save_data(self.option.savetxt, output)

            else:
                prob_ts = output[0].reshape(-1)
                prob_ts = np.array(split_list_into_batches(prob_ts, len(timestamp_range)))

                if flag_rm_seen_ts:
                    input_samples = input_samples[valid_sample_idx]
                    preds = []
                    for idx in range(len(prob_ts)):
                        cur_prob_ts = prob_ts[idx]
                        seen_ts = train_edges[np.all(train_edges[:, :3] == input_samples[idx, :3], axis=1), 3]
                        seen_ts = [timestamp_range.tolist().index(ts) for ts in seen_ts if ts in timestamp_range.tolist()]
                        cur_prob_ts[seen_ts] = 0
                        pred_ts = timestamp_range[np.argmax(cur_prob_ts)]
                        preds.append(pred_ts)

                    preds = np.array(preds).reshape((-1, 1))

                else:
                    pred_ts = timestamp_range[np.argmax(prob_ts, axis=1)].reshape((-1, 1))
                    preds = pred_ts.copy()



                if self.option.flag_interval:
                    prob_te = output[1].reshape(-1)
                    prob_te = np.array(split_list_into_batches(prob_te, len(timestamp_range)))

                    pred_te = timestamp_range[np.argmax(prob_te, axis=1)].reshape((-1, 1))

                    if self.option.flag_use_dur:
                        pred_dur = []
                        for data_idx in np.array(batch_idx_ls)[valid_sample_idx]:
                            pred_dur1 = pred_dur_dict[str(data_idx - self.data['num_samples_dist'][1])]
                            pred_dur.append(abs(pred_dur1[1] - pred_dur1[0]))

                        pred_te = pred_ts + np.array(pred_dur).reshape((-1, 1))

                    preds = np.hstack([pred_ts, pred_te])


                    if self.data['rel_ls_no_dur'] is not None:
                        qq = np.array(qq)[valid_sample_idx]
                        x_tmp = preds[np.isin(qq, self.data['rel_ls_no_dur'])]
                        x_tmp = np.mean(x_tmp, axis=1).reshape((-1,1))
                        preds[np.isin(qq, self.data['rel_ls_no_dur'])] = np.hstack((x_tmp, x_tmp))


                save_data(self.option.savetxt, i)
                save_data(self.option.savetxt, preds)
                save_data(self.option.savetxt, input_intervals[valid_sample_idx])


                if 'aeIOU' in self.metrics:
                    epoch_eval_aeIOU += obtain_aeIoU(preds, input_intervals[valid_sample_idx])
                    print(obtain_aeIoU(preds, input_intervals[valid_sample_idx]))

                if 'TAC' in self.metrics:
                    epoch_eval_TAC += obtain_TAC(preds, input_intervals[valid_sample_idx])
                    print(obtain_TAC(preds, input_intervals[valid_sample_idx]))

                if 'MAE' in self.metrics:
                    # preds = np.array(ref_time_ls)
                    epoch_eval_MAE += np.abs(np.array(preds).reshape(-1) - input_intervals[valid_sample_idx].reshape(-1)).tolist()
                    print(np.abs(np.array(preds).reshape(-1) - input_intervals[valid_sample_idx].reshape(-1)).tolist())
                    print(preds.reshape(-1), input_intervals[valid_sample_idx].reshape(-1))
                    print(np.abs(np.array(ref_time_ls).reshape(-1) - input_intervals[valid_sample_idx].reshape(-1)).tolist())
                    print(np.array(ref_time_ls).reshape(-1), input_intervals[valid_sample_idx].reshape(-1))
            print('----------------------------')

        if mode == "Train":
            if len(epoch_loss) == 0:
                epoch_loss = [100]
            msg = self.msg_with_time(
                    "Epoch %d mode %s Loss %0.4f " 
                    % (self.epoch+1, mode, np.mean(epoch_loss)))
            save_data(self.option.savetxt, msg)
            self.log_file.write(msg + "\n")

            return epoch_loss

        else:
            epoch_eval = [epoch_eval_aeIOU, epoch_eval_TAC, epoch_eval_MAE]

            return epoch_eval


    def one_epoch_train(self):
        loss = self.one_epoch("Train")
        self.train_stats.append([loss])


    def one_epoch_valid(self):
        eval1 = self.one_epoch("Valid")
        self.valid_stats.append([eval1])
        self.best_valid_eval1 = max(self.best_valid_eval1, np.mean(eval1[0]))

    def one_epoch_test(self, total_idx=None):
        eval1 = self.one_epoch("Test", total_idx=total_idx)
        # self.test_stats.append(eval1)
        return eval1



    def early_stop(self):
        loss_improve = self.best_valid_loss == np.mean(self.valid_stats[-1][0])
        in_top_improve = self.best_valid_in_top == np.mean(self.valid_stats[-1][1])
        if loss_improve or in_top_improve:
            return False
        else:
            if self.epoch < self.option.min_epoch:
                return False
            else:
                return True

    def train(self):
        while (self.epoch < self.option.max_epoch and not self.early_stopped):
            self.one_epoch_train()

            self.epoch += 1
            model_path = self.saver.save(self.sess, 
                                         self.option.model_path,
                                         global_step=self.epoch)
            print("Model saved at %s" % model_path)

            # if self.early_stop():
            #     self.early_stopped = True
            #     print("Early stopped at epoch %d" % (self.epoch))


    def test(self, total_idx=None):
        eval1 = self.one_epoch_test(total_idx=total_idx)
        return eval1


    def get_rule_scores(self):
        if self.option.flag_acceleration:
            rule_scores = self.learner.get_rule_scores_acc(self.sess)
            return rule_scores

    def close_log_file(self):
        self.log_file.close()


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
        flag_time_shift = self.option.shift
        num_rule = self.option.num_rule

        mdb = self.data['mdb']
        connectivity = self.data['connectivity']
        TEKG_nodes = self.data['TEKG_nodes']
        file_suffix = '_'
        if self.option.shift:
            file_suffix = '_time_shifting_'

        dataset_index = ['wiki', 'YAGO', 'icews14', 'icews05-15', 'gdelt100'].index(dataset_name)
        weight_exp_dist = [None, None, 0.01, 0.01, 0.05][dataset_index]
        scale_exp_dist = [None, None, 5, 10, 100][dataset_index]
        offset_exp_dist = [None, None, 0, 0, 0][dataset_index]

        if self.data['dataset_name'] in ['wiki', 'YAGO']:
            path = self.data['path']
            edges = np.vstack((self.data['train_edges'], self.data['valid_edges'], self.data['test_edges']))
            pattern_ls = self.data['pattern_ls']
            ts_stat_ls = self.data['ts_stat_ls']
            te_stat_ls = self.data['te_stat_ls']
            rm_ls = self.data['rm_ls']
            output_probs_with_ref_edges = self.data['random_walk_res']
        else:
            edges = np.vstack((self.data['train_edges'], self.data['valid_edges'], self.data['test_edges']))
            pattern_ls = self.data['pattern_ls']
            stat_res = self.data['stat_res']
            pattern_ls_fkt = None
            stat_res_fkt = None
            if self.option.shift:
                pattern_ls_fkt = self.data['pattern_ls_fkt']
                stat_res_fkt = self.data['stat_res_fkt']

            if self.data['random_walk_res'] is not None:
                input_edge_probs = self.data['random_walk_res']

        # print(output_probs_with_ref_edges)
        if flag_acceleration:
            if self.data['dataset_name'] in ['wiki', 'YAGO']:
                query_edges = []
                input_intervals = []
                for data_idx in idx_ls:
                    file = dataset_name + '_idx_' + str(data_idx) + '.json'
                    if not os.path.exists(path + file):
                        continue
                    # print(path + file)
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
                                                 flag_acceleration=flag_acceleration,
                                                 flag_time_shift=flag_time_shift)


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

            # print(input_intervals)

            dim = 1
            if mode == 'Test':
                dim = len(timestamp_range)


            if self.data['dataset_name'] in ['wiki', 'YAGO']:
                if flag_ruleLen_split_ver:
                    # ruleLen_embedding = {}
                    # for rel in pattern_ls:
                    #     ruleLen = np.array([len(rule.split(' '))//2 for rule in pattern_ls[rel]])
                    #     ruleLen = np.hstack((ruleLen, np.zeros((num_rule - len(pattern_ls[rel]),))))
                    #     ruleLen_embedding[int(rel)] = ruleLen.copy()
                    print('Todo')


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

                ref_time_ls = []

                for (i, data_idx) in enumerate(idx_ls):
                    flag_valid = 0
                    if mode == 'Test':
                        # data_idx -= num_samples_dist[1]

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
                                refNode_probs[edge] = {0: {0: {0:[], 1:[]}, 1: {0:[], 1:[]}}, 1: {0: {0:[], 1:[]}, 1: {0:[], 1:[]}}}
                                refNode_res_rw[edge] = []

                            for idx_query_ts_or_te in [0,1]:
                                for idx_ref_ts_or_te in [0,1]:
                                    x = output_probs_with_ref_edges[data_idx][event_type][edge][idx_query_ts_or_te][idx_ref_ts_or_te]

                                    if len(x) > 0:
                                        flag_valid = 1
                                    for x1 in x:
                                        refNode_probs[edge][idx_first_or_last][idx_query_ts_or_te][idx_ref_ts_or_te].append(x1[0])
                                        refNode_res_rw[edge].append(x1[1])

                        # print('------------------------')


                    if flag_valid:
                        num_valid_edge = 0
                        for edge in refNode_res_rw:
                            if len(refNode_res_rw[edge]) == 0:
                                continue

                            num_valid_edge += 1
                            res_random_walk.append([[len(res_random_walk), idx_rule] for idx_rule in refNode_res_rw[edge]])
                            # print(refNode_probs[edge])

                            x = []
                            for idx_first_or_last in [0, 1]:
                                for idx_query_ts_or_te in [0, 1]:
                                    for idx_ref_ts_or_te in [0, 1]:
                                        if flag_ruleLen_split_ver:
                                            # x = [np.mean(refNode_probs[edge][(refNode_res_rw[edge] == 1) & (ruleLen_embedding[qq[i]] == l)], axis=0) for l in range(num_step-1)]
                                            print('Todo')
                                        else:
                                            cur_refNode_probs_ls = refNode_probs[edge][idx_first_or_last][idx_query_ts_or_te][idx_ref_ts_or_te]
                                            if mode == 'Train':
                                                cur_refNode_probs_ls = [[prob] for prob in cur_refNode_probs_ls]
                                            cur_refNode_probs_ls += [1./len(timestamp_range) * np.ones((dim,))] * (len(refNode_res_rw[edge]) - len(cur_refNode_probs_ls))

                                        if mode == 'Train':
                                            if prob_type_for_training == 'max':
                                                if flag_ruleLen_split_ver:
                                                    # x = [np.max(refNode_probs[edge][(refNode_res_rw[edge] == 1) & (ruleLen_embedding[qq[i]] == l)], axis=0) for l in range(num_step-1)]
                                                    print('Todo')
                                                else:
                                                    x.append(np.max(cur_refNode_probs_ls, axis=0))
                                            else:
                                                x.append(np.mean(cur_refNode_probs_ls, axis=0))
                                        else:
                                            x.append(np.mean(cur_refNode_probs_ls, axis=0))


                            if flag_ruleLen_split_ver:
                                # ts_probs_last_event_ts.append([x[l][1, 0, 0, :] for l in range(num_step-1)])
                                # ts_probs_last_event_te.append([x[l][1, 0, 1, :] for l in range(num_step-1)])
                                # ts_probs_first_event_ts.append([x[l][0, 0, 0, :] for l in range(num_step-1)])
                                # ts_probs_first_event_te.append([x[l][0, 0, 1, :] for l in range(num_step-1)])

                                # te_probs_last_event_ts.append([x[l][1, 1, 0, :] for l in range(num_step-1)])
                                # te_probs_last_event_te.append([x[l][1, 1, 1, :] for l in range(num_step-1)])
                                # te_probs_first_event_ts.append([x[l][0, 1, 0, :] for l in range(num_step-1)])
                                # te_probs_first_event_te.append([x[l][0, 1, 1, :] for l in range(num_step-1)])
                                print('Todo')
                            else:
                                ts_probs_last_event_ts.append(x[4])
                                ts_probs_last_event_te.append(x[5])
                                ts_probs_first_event_ts.append(x[0])
                                ts_probs_first_event_te.append(x[1])

                                te_probs_last_event_ts.append(x[6])
                                te_probs_last_event_te.append(x[7])
                                te_probs_first_event_ts.append(x[2])
                                te_probs_first_event_te.append(x[3])


                        if num_valid_edge > 0:
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

                x = []
                for i in range(len(res_random_walk)):
                    x += res_random_walk[i]
                res_random_walk = np.array(x)

                probs = [ts_probs_first_event_ts, ts_probs_first_event_te, ts_probs_last_event_ts, ts_probs_last_event_te,
                         te_probs_first_event_ts, te_probs_first_event_te, te_probs_last_event_ts, te_probs_last_event_te ]

                return qq, query_rels, refNode_source, res_random_walk, probs, valid_sample_idx, input_intervals, input_samples, ref_time_ls
            
            
            else:
                valid_sample_idx = []
                query_rels = []
                refNode_source = []
                res_random_walk = []

                ts_probs_first_event = []
                ts_probs_last_event = []
                ts_probs_first_event_inv_rel = []
                ts_probs_last_event_inv_rel = []

                ref_time_ls = []
                for (i, sample) in enumerate(input_samples):
                    flag_valid = 0
                    refNode_probs = {}
                    refNode_res_rw = {}

                    if self.data['random_walk_res'] is None:
                        ref_time = None
                        cur_path = '../output/' + dataset 
                        if self.option.shift:
                            cur_path += '_time_shifting'
                        cur_path += '/samples/' + dataset + file_suffix + mode + '_sample_'+ str(tuple(sample)) +'.json'
                        
                        samples_dict = {str(tuple(sample)): []}
                        if os.path.exists(cur_path):
                            with open(cur_path, 'r') as file:
                                samples_dict = json.load(file)

                        # print(sample)
                        # print(cur_path)

                        sample_inv = obtain_inv_edge(np.array([sample]), num_rel//2)[0]
                        samples_inv_dict = {str(tuple(sample_inv)): []}
                        cur_path = '../output/' + dataset 
                        if self.option.shift:
                            cur_path += '_time_shifting'
                        cur_path += '/samples/' + dataset + file_suffix
                        if self.option.shift:
                            cur_path += 'fix_ref_time_' 
                        cur_path += mode + '_sample_'+ str(tuple(sample_inv)) +'.json'
                        if os.path.exists(cur_path):
                            with open(cur_path, 'r') as file:
                                samples_inv_dict = json.load(file)

                        if mode == 'Test' and str(tuple(sample_inv)) in samples_dict:
                            ref_time = samples_dict['ref_time']
                            samples_inv_dict[str(tuple(sample_inv))] = samples_dict[str(tuple(sample_inv))]
                            samples_dict = {str(tuple(sample)): samples_dict[str(tuple(sample))]}
                            if ref_time is None:
                                continue
                            # print(ref_time)
                        # print(sample_inv)
                        # print(cur_path)
                        # print(samples_dict.keys())
                        # print(samples_inv_dict.keys())
                        # print(samples_dict)
                        # print(samples_inv_dict)
                        # print('-------------------')
                        # continue

                        input_edge_probs1 = create_inputs_v4(samples_dict, samples_inv_dict, None, pattern_ls, timestamp_range, num_rel//2, stat_res, mode=mode,
                                                             dataset=dataset, file_suffix=file_suffix, 
                                                             flag_compression = False, flag_write = False, 
                                                             flag_time_shifting = self.option.shift,
                                                             pattern_ls_fkt = pattern_ls_fkt,
                                                             stat_res_fkt = stat_res_fkt, shift_ref_time=ref_time, flag_rm_seen_timestamp=True)
                        if len(input_edge_probs1) == 0:
                            continue

                        input_edge_probs1 = input_edge_probs1[0]
                        # sys.exit()
                        # for k in input_edge_probs1:
                        #     print(ref_time, k, input_edge_probs1[k])
                        #     if len(input_edge_probs1[k]) > 0:
                        #         print(input_edge_probs1[k][:ref_time])
                        #     print('-----------------')


                        input_edge_probs = []
                        for j_k in ['input_edge_probs_first_ref', 'input_edge_probs_last_ref', 'input_edge_probs_first_ref_inv_rel', 'input_edge_probs_last_ref_inv_rel']:
                            input_edge_probs.append({tuple(sample): inv_convert_dict(input_edge_probs1[j_k])})

                    type_ref_time = ['ts_first_event', 'ts_last_event', 'ts_first_event', 'ts_last_event']
                    cur_rel_ls = [qq[i], qq[i], qq[i] + num_rel//2, qq[i] + num_rel//2]

                    for j in range(4):
                        if tuple(sample) not in input_edge_probs[j]:
                            continue

                        for edge in input_edge_probs[j][tuple(sample)]:
                            if edge not in refNode_probs:
                                refNode_probs[edge] = {0:[], 1:[], 2: [], 3:[]}
                                refNode_res_rw[edge] = []

                                x = input_edge_probs[j][tuple(sample)][edge]
                                if len(x)>0:
                                    flag_valid = 1
                                for x1 in x:
                                    refNode_probs[edge][j].append(x1[0])
                                    refNode_res_rw[edge].append(x1[1])


                    if flag_valid:
                        num_valid_edge = 0
                        for edge in refNode_res_rw:
                            if len(refNode_res_rw[edge]) == 0:
                                continue

                            num_valid_edge += 1
                            res_random_walk.append([[len(res_random_walk), idx_rule] for idx_rule in refNode_res_rw[edge]])

                            x = []
                            for j in range(4):
                                cur_refNode_probs_ls = refNode_probs[edge][j]
                                if mode == 'Train':
                                    cur_refNode_probs_ls = [[prob] for prob in cur_refNode_probs_ls]

                                dummy_cur_refNode_probs = [1./len(timestamp_range) * np.ones((dim,))]
                                if self.option.shift:
                                    dummy_cur_refNode_probs = rm_seen_timestamp(dummy_cur_refNode_probs, timestamp_range, ref_time)
                                
                                if self.option.shift:
                                    dummy_cur_refNode_probs = [generate_exp_dist(weight_exp_dist, scale_exp_dist, offset_exp_dist, timestamp_range, ref_time)]

                                cur_refNode_probs_ls += dummy_cur_refNode_probs * (len(refNode_res_rw[edge])-len(refNode_probs[edge][j]))
                                
                                if self.option.shift and 0:
                                    cur_refNode_probs_ls = [rm_seen_timestamp([np.array(cur_refNode_probs)], timestamp_range, ref_time)[0].tolist() for cur_refNode_probs in cur_refNode_probs_ls]
                                
                                if mode == 'Train':
                                    if prob_type_for_training == 'max':
                                        x.append(np.max(cur_refNode_probs_ls, axis=0))
                                    else:
                                        x.append(np.mean(cur_refNode_probs_ls, axis=0))
                                else:
                                    # x.append(np.mean(cur_refNode_probs_ls, axis=0) + generate_exp_dist(weight_exp_dist, scale_exp_dist, offset_exp_dist, timestamp_range, ref_time))
                                    x.append(np.mean(cur_refNode_probs_ls, axis=0))


                            ts_probs_first_event.append(x[0])
                            ts_probs_last_event.append(x[1])
                            ts_probs_first_event_inv_rel.append(x[2])
                            ts_probs_last_event_inv_rel.append(x[3])


                        if num_valid_edge>0:
                            valid_sample_idx.append(i)
                            refNode_source.append(num_valid_edge)
                            query_rels += [qq[i]] * num_valid_edge
                            ref_time_ls.append(ref_time)


                refNode_num = 0
                for i in range(len(refNode_source)):
                    x = np.zeros((len(res_random_walk), ))
                    x[refNode_num: refNode_num + refNode_source[i]] = 1
                    refNode_num += refNode_source[i]
                    refNode_source[i] = x.copy()

                x = []
                for i in range(len(res_random_walk)):
                    x += res_random_walk[i]
                res_random_walk = np.array(x)
                probs = [ts_probs_first_event, ts_probs_last_event, ts_probs_first_event_inv_rel, ts_probs_last_event_inv_rel]

            return qq, query_rels, refNode_source, res_random_walk, probs, valid_sample_idx, input_intervals, input_samples, ref_time_ls




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


            batch_edges = np.vstack((batch_edges, batch_edges_inv))
            batch_edges = np.unique(batch_edges, axis=0)
            # batch_edges_inv = batch_edges[batch_edges[:,1] >= num_rel//2]
            batch_edges = batch_edges[batch_edges[:,1] < num_rel//2]
            batch_edges_ori = batch_edges.copy()


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


            if flag_valid:
                valid_sample_idx.append(i)
                valid_ref_event_idx_ts_first.append(ref_event_idx_ts_first)
                valid_ref_event_idx_ts_last.append(ref_event_idx_ts_last)
                valid_ref_event_idx_te_first.append(ref_event_idx_te_first)
                valid_ref_event_idx_te_last.append(ref_event_idx_te_last)


        probs = [ts_probs_first_event, ts_probs_last_event, te_probs_first_event, te_probs_last_event]
        valid_ref_event_idx = [np.array(valid_ref_event_idx_ts_first), np.array(valid_ref_event_idx_ts_last), 
                               np.array(valid_ref_event_idx_te_first), np.array(valid_ref_event_idx_te_last)]
        input_samples = []

        return qq, hh, tt, mdb, connectivity, probs, valid_sample_idx, valid_ref_event_idx, input_intervals, input_samples