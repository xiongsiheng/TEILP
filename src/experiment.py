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


        if self.option.flag_acceleration:
            if self.data['dataset_name'] in ['icews14', 'icews05-15', 'gdelt']:
                TEKG = TEKG_creator_timestamp_acc_ver(self.option, self.data)
            else:
                TEKG = TEKG_creator_interval_acc_ver(self.option, self.data)
        else:
            TEKG = TEKG_creator(self.option, self.data)


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
                qq, query_rels, refNode_source, res_random_walk, probs, valid_sample_idx, input_intervals, input_samples, ref_time_ls = TEKG.create_TEKG_in_batch(batch_idx_ls, mode)
            else:
                qq, hh, tt, mdb, connectivity, probs, valid_sample_idx, valid_ref_event_idx, input_intervals, input_samples = TEKG.create_TEKG_in_batch(batch_idx_ls, mode)


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
                    # print(obtain_aeIoU(preds, input_intervals[valid_sample_idx]))

                if 'TAC' in self.metrics:
                    epoch_eval_TAC += obtain_TAC(preds, input_intervals[valid_sample_idx])
                    # print(obtain_TAC(preds, input_intervals[valid_sample_idx]))

                if 'MAE' in self.metrics:
                    # preds = np.array(ref_time_ls)
                    epoch_eval_MAE += np.abs(np.array(preds).reshape(-1) - input_intervals[valid_sample_idx].reshape(-1)).tolist()
                    # print(np.abs(np.array(preds).reshape(-1) - input_intervals[valid_sample_idx].reshape(-1)).tolist())
                    # print(preds.reshape(-1), input_intervals[valid_sample_idx].reshape(-1))
                    # print(np.abs(np.array(ref_time_ls).reshape(-1) - input_intervals[valid_sample_idx].reshape(-1)).tolist())
                    # print(np.array(ref_time_ls).reshape(-1), input_intervals[valid_sample_idx].reshape(-1))
            # print('----------------------------')

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





class TEKG_creator_base():
    def __init__(self, option, data):
        self.option = option
        self.data = data

        self.file_suffix = '_'
        if self.option.shift:
            self.file_suffix = '_time_shifting_'

        dataset_index = ['wiki', 'YAGO', 'icews14', 'icews05-15', 'gdelt100'].index(self.data['dataset_name'])
        self.weight_exp_dist = [None, None, 0.01, 0.01, 0.05][dataset_index]
        self.scale_exp_dist = [None, None, 5, 10, 100][dataset_index]
        self.offset_exp_dist = [None, None, 0, 0, 0][dataset_index]

        self.edges = np.vstack((self.data['train_edges'], self.data['valid_edges'], self.data['test_edges']))



class TEKG_creator_interval_acc_ver(TEKG_creator_base):
    def create_TEKG_in_batch(self, idx_ls, mode):
        random_walk_res, input_intervals_dict = self._initialize_walk_res(idx_ls, mode)

        query_edges, query_intervals = self._load_queries(idx_ls)
        query_relations = [query[1] for query in query_edges]

        if mode == 'Test':
            self._adjust_intervals_for_test_mode(idx_ls, query_intervals, input_intervals_dict)

        valid_sample_idxs, valid_query_relations, refNode_sources, valid_rule_idx, probabilities = self._process_edges(idx_ls, random_walk_res, mode, query_relations)

        return query_relations, valid_query_relations, refNode_sources, valid_rule_idx, probabilities, valid_sample_idxs, query_intervals, [], []


    def _adjust_intervals_for_test_mode(self, idx_ls, query_intervals, input_intervals_dict):
        for i, data_idx in enumerate(idx_ls):
            if data_idx in input_intervals_dict:
                query_intervals[i] = input_intervals_dict[data_idx].tolist()



    def _finalize_refNode_sources(self, refNode_sources, rule_idx):
        # Finalizes the structure of refNode_sources based on rule_idx
        refNode_num = 0
        for i in range(len(refNode_sources)):
            # print(refNode_num)
            x = np.zeros((len(rule_idx), ))
            x[refNode_num: refNode_num + refNode_sources[i]] = 1
            refNode_num += refNode_sources[i]
            refNode_sources[i] = x.copy()

        x = []
        for i in range(len(rule_idx)):
            x += rule_idx[i]

        return np.array(x)



    def _initialize_probabilities_lists(self):
        # Initializes and returns empty lists for each probability type
        return [[], [], [], [], [], [], [], []]

    def _initialize_refNode_structures(self):
        # Initializes structures to hold probabilities and random walk results for reference nodes
        return {}, {}


    def _initialize_walk_res(self, idx_ls, mode):
        if self.data.get('random_walk_res') is None:
            # if there are no existing res
            output = create_inputs_v3(edges=self.edges, path=self.data['path'], 
                                         dataset=self.data['dataset_name'], 
                                         idx_ls=idx_ls,
                                         pattern_ls=self.data['pattern_ls'], 
                                         timestamp_range=self.data['timestamp_range'], 
                                         num_rel=self.data['num_rel']//2,
                                         ts_stat_ls=self.data['ts_stat_ls'], 
                                         te_stat_ls=self.data['te_stat_ls'],
                                         with_ref_end_time=True,
                                         targ_rel=None, num_samples_dist=self.data['num_samples_dist'], 
                                         mode=mode, rm_ls=self.data['rm_ls'],
                                         flag_output_probs_with_ref_edges=True,
                                         flag_acceleration=self.option.flag_acceleration,
                                         flag_time_shift=self.option.shift)

            return output[-1], output[-2] if mode == 'Test' else {} # only test mode needs targ_interval
        else:
            return self.data['random_walk_res'], self._prepare_test_interval(idx_ls) if mode == 'Test' else {}



    def _load_queries(self, idx_ls):
        query_edges, query_intervals = [], []
        for data_idx in idx_ls:
            file_path = os.path.join(self.data['path'], "%s_idx_%s.json" % (self.data['dataset_name'], data_idx))
            if os.path.exists(file_path):
                with open(file_path, 'r') as file:
                    json_data = json.load(file)
                cur_query = json_data['query']
                query_edges.append(cur_query)
                query_intervals.append(cur_query[3:])
        return query_edges, query_intervals



    def _prepare_test_interval(self, idx_ls):
        input_intervals_dict = {}
        for data_idx in idx_ls:
            data_idx -= self.data['num_samples_dist'][1] # idx alignment
            cur_interval = self.edges[data_idx, 3:]
            input_intervals_dict[data_idx] = cur_interval
        return input_intervals_dict



    def _process_edges(self, idx_ls, random_walk_res, mode, query_relations):
        valid_sample_idxs, valid_query_relations, refNode_sources, valid_rule_idx, probabilities = [], [], [], [], self._initialize_probabilities_lists()
        for i, data_idx in enumerate(idx_ls):
            if data_idx not in random_walk_res:
                continue

            refNode_probs, refNode_res_rw = self._initialize_refNode_structures()
            flag_valid = self._populate_refNode_structures(data_idx, refNode_probs, refNode_res_rw, random_walk_res)

            if flag_valid:
                self._update_output_structures(refNode_probs, refNode_res_rw, probabilities, query_relations, i, valid_sample_idxs, refNode_sources, valid_query_relations, valid_rule_idx, mode)

        valid_rule_idx = self._finalize_refNode_sources(refNode_sources, valid_rule_idx)

        return valid_sample_idxs, valid_query_relations, refNode_sources, valid_rule_idx, probabilities




    def _populate_refNode_structures(self, data_idx, refNode_probs, refNode_res_rw, random_walk_res):
        # Populates refNode_probs and refNode_res_rw based on random_walk_res
        # This is a placeholder for the actual logic to populate these structures
        flag_valid = False
        # Assuming some logic here that updates flag_valid to True if valid data is found
        for idx_first_or_last in [0,1]:
            event_type = ['first_event', 'last_event'][idx_first_or_last]

            for edge in random_walk_res[data_idx][event_type]:
                if edge not in refNode_probs:
                    refNode_probs[edge] = {0: {0: {0:[], 1:[]}, 1: {0:[], 1:[]}}, 1: {0: {0:[], 1:[]}, 1: {0:[], 1:[]}}}
                    refNode_res_rw[edge] = []

                for idx_query_ts_or_te in [0,1]:
                    for idx_ref_ts_or_te in [0,1]:
                        x = random_walk_res[data_idx][event_type][edge][idx_query_ts_or_te][idx_ref_ts_or_te]

                        if len(x) > 0:
                            flag_valid = True
                        for x1 in x:
                            refNode_probs[edge][idx_first_or_last][idx_query_ts_or_te][idx_ref_ts_or_te].append(x1[0])
                            refNode_res_rw[edge].append(x1[1])
        return flag_valid


    def _update_output_structures(self, refNode_probs, refNode_res_rw, probabilities, query_relations, i, valid_sample_idxs, refNode_sources, valid_query_relations, rule_idx, mode):
        # Updates the output structures with processed data from refNode_probs and refNode_res_rw
        # This is a placeholder for the actual logic
        if self.option.flag_ruleLen_split_ver:
            print('Todo')
            pass

        dim = len(self.data['timestamp_range']) if mode == 'Test' else 1

        num_valid_edge = 0
        for edge in refNode_res_rw:
            if len(refNode_res_rw[edge]) == 0:
                continue

            num_valid_edge += 1
            rule_idx.append([[len(rule_idx), idx_rule] for idx_rule in refNode_res_rw[edge]])
            # print(refNode_probs[edge])

            x = []
            for idx_first_or_last in [0, 1]:
                for idx_query_ts_or_te in [0, 1]:
                    for idx_ref_ts_or_te in [0, 1]:
                        cur_refNode_probs_ls = refNode_probs[edge][idx_first_or_last][idx_query_ts_or_te][idx_ref_ts_or_te]
                        if mode == 'Train':
                            cur_refNode_probs_ls = [[prob] for prob in cur_refNode_probs_ls]
                        cur_refNode_probs_ls += [1./len(self.data['timestamp_range']) * np.ones((dim,))] * (len(refNode_res_rw[edge]) - len(cur_refNode_probs_ls))

                        if mode == 'Train' and self.option.prob_type_for_training == 'max':
                            x.append(np.max(cur_refNode_probs_ls, axis=0))
                        else:
                            x.append(np.mean(cur_refNode_probs_ls, axis=0))

            # Indices mapping: defines which element of x goes into which sublist of probabilities
            indices_mapping = [0, 1, 4, 5, 2, 3, 6, 7]

            # Loop through the mapping to append items to the corresponding sublist
            for i, index in enumerate(indices_mapping):
                probabilities[i].append(x[index])


        if num_valid_edge > 0:
            valid_sample_idxs.append(i)
            refNode_sources.append(num_valid_edge)
            valid_query_relations += [query_relations[i]] * num_valid_edge



class TEKG_creator_timestamp_acc_ver(TEKG_creator_base):
    def create_TEKG_in_batch(self, idx_ls, mode):
        """
        Create Temporal Edge Knowledge Graphs (TEKG) in batch with timestamp accuracy verification.
        
        Parameters:
        - idx_ls: List of indices for which TEKG needs to be generated.
        - mode: The mode of operation, e.g., 'Test' or 'Train'.
        
        Returns:
        A tuple containing processed TEKG components.
        """
        pattern_ls_fkt, stat_res_fkt = self._handle_option_shift()
        input_edge_probs = self.data.get('random_walk_res')
        
        input_samples, input_intervals, qq = self._extract_edge_info(idx_ls)
        
        # Initialize variables for results
        valid_sample_idx, query_rels, refNode_source, res_random_walk, probs, ref_time_ls = [], [], [], [], [], self._initialize_probabilities_lists(), []

        for i, sample in enumerate(input_samples):
            sample_results = self._process_sample(sample, mode, pattern_ls_fkt, stat_res_fkt, qq[i])
            if sample_results:
                valid_sample_idx.append(i)
                query_rels.extend(sample_results['query_rels'])
                refNode_source.append(sample_results['num_valid_edge'])
                res_random_walk.extend(sample_results['res_random_walk'])
                for j in range(len(probs))
                    probs[j].extend(sample_results['probs'][j])
                ref_time_ls.append(sample_results['ref_time'])

        refNode_source = self._process_refNode_source(refNode_source)
        res_random_walk = np.array([item for sublist in res_random_walk for item in sublist])

        return qq, query_rels, refNode_source, res_random_walk, probs, valid_sample_idx, input_intervals, input_samples, ref_time_ls



    def _aggregate_probs_for_events(self, probs, mode):
        """
        Aggregates probabilities for an event type based on the mode.
        
        Parameters:
        - probs: List of probabilities for an event.
        - mode: The mode of operation, e.g., 'Test' or 'Train'.
        
        Returns:
        Aggregated probabilities for the event. This implementation simply takes the mean of the probabilities,
        but you might choose a different strategy based on your application's specifics.
        """
        if mode == 'Train' and self.option.prob_type_for_training == 'max':
            # For training, you might prefer the mean to smooth out the data
            return np.max(probs)
        else:
            # For testing or other modes, you might choose the maximum probability to focus on the most likely event
            return np.mean(probs)




    def _build_sample_path(self, sample, mode, inverse=False):
        """
        Constructs the file path for a sample based on mode and whether it's an inverse sample.
        
        Parameters:
        - sample: The sample to construct the path for.
        - mode: The mode of operation, e.g., 'Test' or 'Train'.
        - inverse: Boolean indicating if the path should be for the inverse sample.
        
        Returns:
        Constructed file path as a string.
        """
        # This method should construct the file path similarly to how it's done in the original code.
        cur_path = '../output/' + self.data['dataset'] 

        if self.option.shift:
            cur_path += '_time_shifting'

        cur_path += '/samples/' + self.data['dataset'] + self.file_suffix

        if inverse and self.option.shift:
            cur_path += 'fix_ref_time_'

        cur_path += mode + '_sample_'+ str(tuple(sample)) +'.json'

        return cur_path



    def _convert_format_edge_probs(input_edge_probs1, sample):
        input_edge_probs1 = input_edge_probs1[0]

        input_edge_probs = []
        for j_k in ['input_edge_probs_first_ref', 'input_edge_probs_last_ref', 'input_edge_probs_first_ref_inv_rel', 'input_edge_probs_last_ref_inv_rel']:
            input_edge_probs.append({tuple(sample): inv_convert_dict(input_edge_probs1[j_k])})

        return input_edge_probs


    def _calculate_probs_and_walks(self, input_edge_probs, sample, rel_type, mode):
        """
        Calculate probabilities and random walks for a given sample and its inputs.
        
        Parameters:
        - input_edge_probs: The probabilities for input edges.
        - sample: The current sample being processed.
        - rel_type: The type of relationship for the current sample.
        - mode: The mode of operation, e.g., 'Test' or 'Train'.
        
        Returns:
        A tuple containing a flag indicating if the sample is valid, lists of probabilities,
        list of random walks, and the number of valid edges.
        """
        flag_valid = False
        ts_probs_first_event, ts_probs_last_event, ts_probs_first_event_inv_rel, ts_probs_last_event_inv_rel = [], [], [], []
        res_random_walk = []
        num_valid_edge = 0

        # Process each edge in the input_edge_probs structure
        for j, edge_prob_group in enumerate(input_edge_probs):
            # Each edge_prob_group corresponds to probabilities for different types of events (first, last, inv first, inv last)
            for edge, probs in edge_prob_group.items():
                # Check if the edge has any associated probabilities (thus valid)
                if probs:
                    flag_valid = True
                    random_walks_for_edge = self._generate_random_walks_for_edge(probs, sample, edge)
                    res_random_walk.extend(random_walks_for_edge)

                    # Aggregate probabilities for each event type
                    event_probs = self._aggregate_probs_for_events(probs, mode)
                    if j == 0:
                        ts_probs_first_event.append(event_probs)
                    elif j == 1:
                        ts_probs_last_event.append(event_probs)
                    elif j == 2:
                        ts_probs_first_event_inv_rel.append(event_probs)
                    elif j == 3:
                        ts_probs_last_event_inv_rel.append(event_probs)

                    num_valid_edge += 1

        # Combine all the probability lists into a single list for easier handling downstream
        probs_combined = [ts_probs_first_event, ts_probs_last_event, ts_probs_first_event_inv_rel, ts_probs_last_event_inv_rel]

        return flag_valid, probs_combined, res_random_walk, num_valid_edge



    def _extract_edge_info(self, idx_ls):
        """
        Extracts edge information from the provided indices.
        
        Parameters:
        - idx_ls: List of indices for which to extract edge information.
        
        Returns:
        A tuple of input samples, intervals, and relationship types.
        """
        input_samples = self.edges[idx_ls]
        input_intervals = self.edges[idx_ls, 3]
        qq = self.edges[idx_ls, 1]
        return input_samples, input_intervals, qq


    def _generate_random_walks_for_edge(self, probs, sample, edge):
        """
        Generates random walks for a given edge based on probabilities.
        
        Each walk is represented as a tuple of (sample index, probability, edge), 
        illustrating a simple approach to random walk generation where each edge is considered once.
        
        Parameters:
        - probs: Probabilities associated with the edge.
        - sample: The current sample being processed.
        - edge: The edge for which to generate random walks.
        
        Returns:
        A list of generated random walks for the edge, with each walk including the edge and its probability.
        """
        random_walks = []
        for prob in probs:
            # Here, we simply pair each probability with its edge to represent a "walk"
            # This is a simplistic representation and should be adapted to your specific logic and requirements
            random_walks.append((edge, prob))
        return random_walks



    def _handle_option_shift(self):
        """
        Handles the 'shift' option in the TEKG data preparation.
        
        Returns:
        A tuple containing pattern and statistics results if option 'shift' is enabled.
        """
        if self.option.shift:
            return self.data['pattern_ls_fkt'], self.data['stat_res_fkt']
        return None, None



    def _load_sample_data(self, sample, mode, inverse=False):
        """
        Load data for a sample or its inverse from the filesystem.
        
        Parameters:
        - sample: The sample to load data for.
        - mode: The mode of operation, e.g., 'Test' or 'Train'.
        - inverse: Boolean indicating if the inverse sample data should be loaded.
        
        Returns:
        Loaded data as a dictionary or None if the file does not exist.
        """
        cur_path = self._build_sample_path(sample, mode, inverse)
        if os.path.exists(cur_path):
            with open(cur_path, 'r') as file:
                return json.load(file)
        return {str(tuple(sample)): []}


    def _obtain_inv_edge(self, sample):
        """
        Obtain the inverse edge for a given sample.
        
        Parameters:
        - sample: The sample for which to find the inverse edge.
        
        Returns:
        Inverse edge.
        """
        # Assuming the logic to obtain the inverse edge is implemented here.

        sample_inv = obtain_inv_edge(np.array([sample]), self.data['num_rel']//2)[0]

        return sample_inv



    def _process_refNode_source(self, refNode_source):
        """
        Transforms refNode_source information into a structured binary matrix format.
        
        Parameters:
        - refNode_source: List containing the number of reference nodes associated with each sample.
        
        Returns:
        A binary matrix indicating the association between samples and reference nodes.
        """
        # Calculate the total number of reference nodes across all samples
        total_refNodes = sum(refNode_source)
        
        # Initialize a binary matrix of shape [number of samples, total number of reference nodes]
        # Each row corresponds to a sample, each column to a reference node
        association_matrix = np.zeros((len(refNode_source), total_refNodes))
        
        # Current start index for setting ones in the matrix
        current_start_idx = 0
        
        for i, num_refNodes in enumerate(refNode_source):
            # Set ones for the current sample's reference nodes
            association_matrix[i, current_start_idx:current_start_idx + num_refNodes] = 1
            
            # Update the start index for the next sample
            current_start_idx += num_refNodes
        
        return association_matrix



    def _process_sample(self, sample, mode, pattern_ls_fkt, stat_res_fkt, rel_type):
        """
        Processes a single sample for TEKG generation, including loading data,
        handling inversions, and calculating probabilities.
        
        Parameters:
        - sample: The current sample to process.
        - mode: The mode of operation, e.g., 'Test' or 'Train'.
        - pattern_ls_fkt: Pre-computed pattern list function if option 'shift' is enabled.
        - stat_res_fkt: Pre-computed statistics results function if option 'shift' is enabled.
        - rel_type: The type of relationship for the current sample.
        
        Returns:
        A dictionary containing processed information for the sample or None if not valid.
        """
        
        sample_dict = self._load_sample_data(sample, mode)
        sample_inv_dict = self._load_sample_data(self._obtain_inv_edge(sample), mode, inverse=True)

        # if not sample_dict or not sample_inv_dict:
        #     return None

        # Assuming create_inputs_v4 and other required methods are defined elsewhere within the class
        input_edge_probs = create_inputs_v4(sample_dict, sample_inv_dict, None, self.data['pattern_ls'], self.data['timestamp_range'], 
                                             self.data['num_rel']//2, self.data['stat_res'], mode=mode,
                                             dataset=self.data['dataset'], file_suffix=self.file_suffix, 
                                             flag_compression = False, flag_write = False, 
                                             flag_time_shifting = self.option.shift,
                                             pattern_ls_fkt = pattern_ls_fkt,
                                             stat_res_fkt = stat_res_fkt, shift_ref_time=None, flag_rm_seen_timestamp=True)

        if len(input_edge_probs) == 0:
            return None

        input_edge_probs = self._convert_format_edge_probs(input_edge_probs, sample)

        flag_valid, probs, res_random_walk, num_valid_edge = self._calculate_probs_and_walks(input_edge_probs, sample, rel_type, mode)

        if not flag_valid:
            return None

        ref_time = sample_dict.get('ref_time')
        return {
            'query_rels': [rel_type] * num_valid_edge,
            'num_valid_edge': num_valid_edge,
            'res_random_walk': res_random_walk,
            'probs': probs,
            'ref_time': ref_time
        }



class TEKG_creator(TEKG_creator_base):
    def create_TEKG_in_batch(self, idx_ls, mode):
        num_step = self.option.num_step-1
        mdb = copy.copy(self.data['mdb'])
        connectivity = copy.copy(self.data['connectivity'])
        TEKG_nodes = copy.copy(self.data['TEKG_nodes'])
        output_probs_with_ref_edges = copy.copy(self.data['random_walk_res'])

        flag_use_batch_graph = False
        if (mdb is None) or (connectivity is None) or (TEKG_nodes is None):
            flag_use_batch_graph = True

        query_edges, input_intervals = [], []
        batch_edges = [] if flag_use_batch_graph else None

        for data_idx in idx_ls:
            file_path = "{}{}_train_query_{}.json".format(self.data['path'], self.data['dataset_name'], data_idx)
            with open(file_path, 'r') as f:
                json_data = json.load(f)
            self._process_json_data(json_data, query_edges, input_intervals, batch_edges)


        if flag_use_batch_graph:
            batch_edges = self._unique_edges(batch_edges)
            batch_edges_inv = self._inverse_edges(batch_edges, self.data['num_rel'])
            batch_edges = self._unique_edges(np.vstack((batch_edges, batch_edges_inv)))  # Combine and get unique edges again
            assert len(batch_edges) <= self.data['num_entity'] // 2, 'Increase num_entity or reduce edges.'
            mdb = self._build_mdb(batch_edges, self.data['num_entity'], self.data['num_rel'])
            connectivity = self._calculate_connectivity(batch_edges, self.data['num_entity'])


        if output_probs_with_ref_edges is None:
            if self.option.flag_ruleLen_split_ver:
                output = create_inputs_v3_ruleLen_split_ver(edges=edges, path=self.data['path'], 
                                                             dataset=self.data['dataset_name'], 
                                                             idx_ls=idx_ls,
                                                             pattern_ls=self.data['pattern_ls'], 
                                                             timestamp_range=self.data['timestamp_range'], 
                                                             num_rel=self.data['num_rel']//2,
                                                             ts_stat_ls=self.data['ts_stat_ls'], 
                                                             te_stat_ls=self.data['te_stat_ls'],
                                                             with_ref_end_time=True,
                                                             targ_rel=None, num_samples_dist=self.data['num_samples_dist'], 
                                                             mode=mode, rm_ls=self.data['rm_ls'],
                                                             flag_output_probs_with_ref_edges=True)
            else:
                output = create_inputs_v3(edges=edges, path=self.data['path'], 
                                             dataset=self.data['dataset_name'], 
                                             idx_ls=idx_ls,
                                             pattern_ls=self.data['pattern_ls'], 
                                             timestamp_range=self.data['timestamp_range'], 
                                             num_rel=self.data['num_rel']//2,
                                             ts_stat_ls=self.data['ts_stat_ls'], 
                                             te_stat_ls=self.data['te_stat_ls'],
                                             with_ref_end_time=True,
                                             targ_rel=None, num_samples_dist=self.data['num_samples_dist'], 
                                             mode=mode, rm_ls=self.data['rm_ls'],
                                             flag_output_probs_with_ref_edges=True)

            output_probs_with_ref_edges = output[-1]

            if mode == 'Test':
                input_intervals_dict = output[-2]

                for i, data_idx in enumerate(idx_ls):
                    data_idx -= self.data['num_samples_dist'][1]
                    if data_idx in input_intervals_dict:
                        input_intervals[i] = input_intervals_dict[data_idx].tolist()



        qq = [query[1] for query in query_edges]
        if flag_use_batch_graph:
            hh = [batch_edges_ori.tolist().index(query) for query in query_edges]
        else:
            hh = [TEKG_nodes.tolist().index(query) for query in query_edges]

        tt = [h + self.data['num_entity']//2 for h in hh]


        ts_probs_first_event, te_probs_first_event, ts_probs_last_event, te_probs_last_event = self._initialize_probabilities(
            len(idx_ls), self.data['num_entity'], num_step, self.data['timestamp_range'], mode
        )


        valid_sample_idx, valid_ref_event_idx_ts_first, valid_ref_event_idx_ts_last, valid_ref_event_idx_te_first, valid_ref_event_idx_te_last = [], [], [], [], []

        for (i, data_idx) in enumerate(idx_ls):
            flag_valid = 0
            if mode == 'Test':
                data_idx -= self.data['num_samples_dist'][1]

            if data_idx not in output_probs_with_ref_edges:
                continue

            ref_event_idx_ts_first, ref_event_idx_ts_last, ref_event_idx_te_first, ref_event_idx_te_last = np.zeros((self.data['num_entity'],)), np.zeros((self.data['num_entity'],)), \
                                                                                        np.zeros((self.data['num_entity'],)), np.zeros((self.data['num_entity'],))

            flag_valid |= self._update_event_probabilities(output_probs_with_ref_edges[data_idx]['first_event'], flag_use_batch_graph, batch_edges, 
                                                            batch_edges_idx_cmp, TEKG_nodes, num_step, mode, 
                                                            ts_probs_first_event, te_probs_first_event, ref_event_idx_ts_first, ref_event_idx_te_first)

            flag_valid |= self._update_event_probabilities(output_probs_with_ref_edges[data_idx]['last_event'], flag_use_batch_graph, batch_edges, 
                                                            batch_edges_idx_cmp, TEKG_nodes, num_step, mode, 
                                                            ts_probs_last_event, te_probs_last_event, ref_event_idx_ts_last, ref_event_idx_te_last)

            if flag_valid:
                valid_sample_idx.append(i)
                valid_ref_event_idx_ts_first.append(ref_event_idx_ts_first)
                valid_ref_event_idx_ts_last.append(ref_event_idx_ts_last)
                valid_ref_event_idx_te_first.append(ref_event_idx_te_first)
                valid_ref_event_idx_te_last.append(ref_event_idx_te_last)


        probs = [ts_probs_first_event, ts_probs_last_event, te_probs_first_event, te_probs_last_event]
        valid_ref_event_idx = [np.array(valid_ref_event_idx_ts_first), np.array(valid_ref_event_idx_ts_last), 
                               np.array(valid_ref_event_idx_te_first), np.array(valid_ref_event_idx_te_last)]

        return qq, hh, tt, mdb, connectivity, probs, valid_sample_idx, valid_ref_event_idx, input_intervals, []



    def _assign_probability(self, prob, prob_type, j, cur_edge_idx, step, ts_probs, te_probs):
        if prob_type == 'ts':
            if step is not None:
                ts_probs[j][i][cur_edge_idx][step] = prob
            else:
                ts_probs[j][i][cur_edge_idx] = prob
        elif prob_type == 'te':
            if step is not None:
                te_probs[j][i][cur_edge_idx][step] = prob
            else:
                te_probs[j][i][cur_edge_idx] = prob

    def _build_mdb(self, edges, num_entity, num_rel):
        """Build the mdb dictionary based on current relations and entities."""
        mdb = {}
        edges_idx = np.arange(len(edges))
        for r in range(num_rel // 2):
            idx_cur_rel = edges_idx[edges[:, 1] == r].reshape((-1, 1))
            if idx_cur_rel.size == 0:
                mdb[r] = [[[0, 0]], [0.0], [num_entity, num_entity]]
                mdb[r + num_rel // 2] = [[[0, 0]], [0.0], [num_entity, num_entity]]
            else:
                mdb[r] = [np.hstack([idx_cur_rel, idx_cur_rel]).tolist(), [1.0] * len(idx_cur_rel), [num_entity, num_entity]]
                mdb[r + num_rel // 2] = [np.hstack([idx_cur_rel + num_entity // 2, idx_cur_rel + num_entity // 2]).tolist(), [1.0] * len(idx_cur_rel), [num_entity, num_entity]]
        return mdb


    def _calculate_connectivity(self, batch_edges, num_entity):
        """Calculate connectivity matrix."""
        connectivity = {key: [] for key in range(4)}
        batch_edges_idx_cmp = np.hstack((np.arange(len(batch_edges)), np.arange(len(batch_edges)) + num_entity // 2))
        
        for ent in np.unique(np.hstack([batch_edges[:, 0], batch_edges[:, 2]])):
            b = batch_edges_idx_cmp[batch_edges[:, 2] == ent]
            a = batch_edges_idx_cmp[batch_edges[:, 0] == ent]
            combinations = np.array(list(itertools.product(a, b)))

            combinations_TR = combinations.copy()
            combinations_TR[combinations_TR>=num_entity//2] -= num_entity//2
            TRs = calculate_TR_mat_ver(batch_edges[combinations_TR[:,0], 3:], batch_edges[combinations_TR[:,1], 3:])

            for i in range(4):
                mask = TRs == i if i else slice(None)
                connectivity[i] += combinations[mask].tolist()

        for TR in range(4):
            if connectivity[TR]:
                A = np.array(connectivity[TR])
                B = A.copy()
                B[B >= num_entity // 2] -= num_entity // 2
                A = A[~(B[:, 0] == B[:, 1])]
                if A.size == 0:
                    connectivity[TR] = [[[0, 0]], [0.0], [num_entity, num_entity]]
                else:
                    A = np.unique(A, axis=0)
                    connectivity[TR] = [A.tolist(), [1.0] * len(A), [num_entity, num_entity]]
            else:
                connectivity[TR] = [[[0, 0]], [0.0], [num_entity, num_entity]]

        return connectivity


    def _get_current_edge_index(self, edge, flag_use_batch_graph, batch_edges, batch_edges_idx_cmp, TEKG_nodes):
        edge_transform = [edge[j] for j in [0,1,4,2,3]]
        if flag_use_batch_graph:
            return batch_edges_idx_cmp[batch_edges.tolist().index(edge_transform)]
        else:
            return TEKG_nodes.tolist().index(edge_transform)


    def _initialize_probabilities(self, num_lists, num_entities, num_steps, timestamp_range, mode):
        """
        Initialize probabilities for events.

        Parameters:
        - num_lists: The number of lists or batches.
        - num_entities: The number of entities per list.
        - num_steps: The number of steps per entity.
        - timestamp_range: The range of timestamps.
        - flag_ruleLen_split_ver: A flag indicating which version to use for Train mode.
        - mode: The mode of operation ('Train' or 'Test').

        Returns:
        A tuple of four lists representing the probabilities for the first and last events' start and end times.
        """
        if mode == 'Train':
            if self.option.flag_ruleLen_split_ver:
                base_array = np.ones((len(idx_ls), num_entities, num_steps)) / len(timestamp_range)
            else:
                base_array = np.tile(np.ones(num_entities) / len(timestamp_range), (len(idx_ls), 1, num_steps))
        elif mode == 'Test':
            base_array = np.tile(np.ones((len(timestamp_range),)) / len(timestamp_range), (len(idx_ls), num_entities, num_steps))
        else:
            raise ValueError("Invalid mode. Choose either 'Train' or 'Test'.")

        return ([base_array.tolist()] * 2, [base_array.tolist()] * 2, 
                [base_array.tolist()] * 2, [base_array.tolist()] * 2)


    def _inverse_edges(self, edges, num_rel):
        """Inverse the edges and adjust relation IDs for symmetry."""
        inv_edges = edges[:, [2, 1, 0, 3, 4]]
        mask = edges[:, 1] < num_rel // 2
        inv_edges[mask, 1] += num_rel // 2
        inv_edges[~mask, 1] -= num_rel // 2
        return inv_edges


    def _process_json_data(self, json_data, query_edges, input_intervals, batch_edges=None):
        cur_query = json_data['query']
        query_edges.append(cur_query)
        input_intervals.append(cur_query[3:])

        if batch_edges is not None:
            batch_edges.append(cur_query)
            for rule_len in range(1, 6):
                walks = json_data.get(str(rule_len), [])
                for walk in walks:
                    for i in range(rule_len):
                        x = walk[4*i:4*i+5]
                        batch_edges.append([x[j] for j in [0, 1, 4, 2, 3]])


    def _select_probability(self, probs, mode, aggregate=False):
        if mode == 'Train':
            if self.option.prob_type_for_training == 'max':
                return max(probs)
            else:
                return np.mean(probs)
        elif mode == 'Test':
            if aggregate:
                return np.mean(probs, axis=0)
            else:
                return np.mean(probs)



    def _unique_edges(self, edges):
        """Return unique edges, ensuring no duplicates."""
        return np.unique(edges, axis=0)



    def _update_event_probabilities(self, events, flag_use_batch_graph, batch_edges, batch_edges_idx_cmp, TEKG_nodes, num_step, mode, ts_probs, te_probs, ref_event_idx_ts, ref_event_idx_te):
        flag_valid = 0  # Initialize flag_valid
        for event_key, event_value in events.items():
            for edge in event_value:
                cur_edge_idx = self._get_current_edge_index(edge, flag_use_batch_graph, batch_edges, batch_edges_idx_cmp, TEKG_nodes)
                flag_valid |= self._update_probabilities_for_edge(edge, event_value, cur_edge_idx, num_step, mode, ts_probs, te_probs, ref_event_idx_ts, ref_event_idx_te)
        return flag_valid


    def _update_probabilities_for_edge(self, edge, event, cur_edge_idx, num_step, mode, ts_probs, te_probs, ref_event_idx_ts, ref_event_idx_te):
        flag_valid = 0
        for j in [0, 1]:
            if self.option.flag_ruleLen_split_ver:
                for l in range(num_step):
                    flag_valid |= self._update_probability_by_step(edge, j, l+1, cur_edge_idx, event, mode, ts_probs, te_probs)
            else:
                flag_valid |= self._update_probability_without_step(edge, j, cur_edge_idx, event, mode, ts_probs, te_probs)
            ref_event_idx_ts[cur_edge_idx] = 1
            ref_event_idx_te[cur_edge_idx] = 1
        return flag_valid


    def _update_probability_by_step(self, edge, j, step, cur_edge_idx, event, mode, ts_probs, te_probs):
        flag_valid = 0
        for prob_type_key, probs_event in zip(['ts', 'te'], [0, 1]):
            if len(event[edge][probs_event][j][step]) > 0:
                selected_prob = self._select_probability(event[edge][probs_event][j][step], mode)
                self._assign_probability(selected_prob, prob_type_key, j, cur_edge_idx, step, ts_probs, te_probs)
                flag_valid = 1
        return flag_valid


    def _update_probability_without_step(self, edge, j, cur_edge_idx, event, mode, ts_probs, te_probs):
        flag_valid = 0
        for prob_type_key, probs_event in zip(['ts', 'te'], [0, 1]):
            if len(event[edge][probs_event][j]) > 0:
                selected_prob = self._select_probability(event[edge][probs_event][j], mode, aggregate=True)
                self._assign_probability(selected_prob, prob_type_key, j, cur_edge_idx, None, ts_probs, te_probs)
                flag_valid = 1
        return flag_valid