import sys
import os
import numpy as np
import itertools
import json
from utlis import *
from gadgets import *



class TEKG_params():
    def __init__(self, option, data):
        self.option = option
        self.data = data
        self.file_suffix = '_time_shifting_' if self.option.shift else '_'

        dataset_index = ['wiki', 'YAGO', 'icews14', 'icews05-15', 'gdelt100'].index(self.data['dataset_name'])
        self.weight_exp_dist = [None, None, 0.01, 0.01, 0.05][dataset_index]
        self.scale_exp_dist = [None, None, 5, 10, 100][dataset_index]
        self.offset_exp_dist = [None, None, 0, 0, 0][dataset_index]

        self.edges = np.vstack((self.data['train_edges'], self.data['valid_edges'], self.data['test_edges']))


class TEKG():
    def __init__(self, option, data):
        if option.flag_acceleration:
            if data['dataset_name'] in ['icews14', 'icews05-15', 'gdelt']:
                self.graph = TEKG_timestamp_acc_ver(option, data)
            else:
                self.graph = TEKG_int_acc_ver(option, data)
        else:
            self.graph = TEKG_normal_ver(option, data)



class TEKG_int_acc_ver(TEKG_params):
    def _initialize_walk_res(self, idx_ls, mode):
        '''
        Given the idx_ls, either read the pre-processing results or do the pre-processing right now

        Parameters:
        - idx_ls: The list of indices for which to generate the TEKG.
        - mode: The mode of operation ('Train' or 'Test').

        Returns:
        - prob_dict: A dictionary containing the probabilities for each sample. 
        - target_interval: tarerget interval for each sample (test mode only)
        '''
        assert mode in ['Train', 'Test']
        
        if mode == 'Train':
            # instead of reading all the results, we read the results for the current batch
            prob_dict = {}
            target_interval = {}
            for data_idx in idx_ls:
                path_probs = '../output/process_res/' + self.data['dataset_name'] + '_idx_' + str(data_idx) + '_input.json'
                if not os.path.exists(path_probs):
                    # print(path_probs)
                    continue
                with open(path_probs) as f:
                    prob_dict[data_idx] = json.load(f)[1]
        else:
            preprocessor = Data_preprocessor()
            # for test mode, there is no pre-processing
            output = preprocessor.prepare_inputs_interval_ver(edges=self.edges, path=self.data['path'], 
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
                                         flag_time_shift=self.option.shift, show_tqdm=False, probs_normalization=False)
            prob_dict, target_interval = output[-1], output[-2] # only test mode needs targ_interval

        return prob_dict, target_interval


    def _load_queries(self, idx_ls):
        '''
        Load the queries for the given idx_ls
        '''
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


    def _create_rule_used_ls(self, walk_res, query_rel, mode, max_rule_num=20):
        '''
        For each sample, we choose a fixed number of rules to use according to their scores.
        Previous: 100
        '''
        if mode == 'Train':
            return None, None, None

        rule_used_ls = []
        for idx_event_pos in [0,1]:
            for edge in walk_res[idx_event_pos]:
                for idx_query_time in [0,1]:
                    for idx_ref_time in [0,1]:
                        idx_complete = idx_query_time*2 + idx_ref_time
                        probs = walk_res[idx_event_pos][edge][idx_complete]
                        for prob_dict in probs:
                            if prob_dict[1] not in rule_used_ls:
                                rule_used_ls.append(prob_dict[1])

        
        # read rule score and sort the rules
        path_suffix = '_' if not self.option.shift else '_time_shifting_'
        cur_path = '../output/' + self.option.dataset + path_suffix[:-1] + '/' + self.option.dataset + path_suffix
        cur_path += 'rule_scores_rel_' + str(query_rel) + '.json'
        rule_scores, refType_scores = None, None
        if os.path.exists(cur_path):
            with open(cur_path) as file:
                data = json.load(file)

            rule_scores = data['rule_scores']
            refType_scores = data["refType_scores"]

            # select the top k rules
            rule_used_ls = sorted(rule_used_ls, key=lambda x: rule_scores[x], reverse=True)
            rule_used_ls = rule_used_ls[:max_rule_num]
            
            # normalize the rule scores
            rule_scores = [0 if idx not in rule_used_ls else score for idx, score in enumerate(rule_scores)]
            rule_scores = [score/(sum(rule_scores) + 1e-20) for score in rule_scores]
        
        return rule_used_ls, rule_scores, refType_scores


    def _populate_refNode_structures(self, walk_res, mode, rule_used_ls=None, rule_scores=None, refType_scores=None):
        '''
        For each sample, populates refNode_probs and refNode_rule_idx based on random_walk_res
        '''
        flag_valid = False
        refNode_probs, refNode_rule_idx, preds = {}, {}, {}
        for idx_event_pos in [0,1]:
            # change the format
            idx_event_pos = str(idx_event_pos) if mode == 'Train' else idx_event_pos
            for edge in walk_res[idx_event_pos]:
                if eval(edge)[2] == 9999 or eval(edge)[3] == 9999:
                    # we don't need unknown time events as reference events
                    continue
                if edge not in refNode_probs:
                    refNode_probs[edge] = {0: {0: {0:[], 1:[]}, 1: {0:[], 1:[]}}, 1: {0: {0:[], 1:[]}, 1: {0:[], 1:[]}}}  # [idx_event_pos][idx_query_time][idx_ref_time]
                    refNode_rule_idx[edge] = []
                    if mode == 'Test':
                        preds[edge] = {0: 0, 1: 0}  # query ts or te

                # for each event, we have a list of probabilities
                for idx_query_time in [0,1]:
                    for idx_ref_time in [0,1]:
                        idx_complete = idx_query_time*2 + idx_ref_time
                        idx_complete = str(idx_complete) if mode == 'Train' else idx_complete
                        probs = walk_res[idx_event_pos][edge][idx_complete]

                        # Given the query, the event pos and the reference event, more than one rules are satisfied.
                        for prob_dict in probs:
                            if (rule_used_ls is not None) and (prob_dict[1] not in rule_used_ls):
                                continue

                            refNode_probs[edge][int(idx_event_pos)][idx_query_time][idx_ref_time].append(prob_dict)
                            refNode_rule_idx[edge].append(prob_dict[1])
                            
                            if mode == 'Test' and (rule_scores is not None) and (refType_scores is not None):
                                # refType_scores: [(first_event_ts, first_event_te), (last_event_ts, last_event_te), (first_event, last_event)]
                                prob_event_pos = refType_scores[3*idx_query_time + 2][idx_event_pos]
                                prob_ref_time = refType_scores[3*idx_query_time + idx_event_pos][idx_ref_time]
                                preds[edge][idx_query_time] += prob_event_pos * prob_ref_time * rule_scores[prob_dict[1]] * np.array(prob_dict[0])
                            
                            flag_valid = True
        
        return flag_valid, refNode_probs, refNode_rule_idx, preds


    def _update_outputs(self, refNode_probs, refNode_rule_idx, query_relations, idx, valid_sample_idxs, 
                                  refNode_sources, valid_query_relations, rule_idx, mode, preds):
        '''
        Updates the output structures with processed data from refNode_probs and refNode_rule_idx
        '''
        # This is a placeholder for the actual logic
        if self.option.flag_ruleLen_split_ver:
            print('Todo')
            pass
        
        if mode == 'Test' and len(preds) > 0:
            # Given the sample, merge probs from different events.
            final_preds = {0: 0, 1: 0}
            for edge in preds:
                for idx_query_time in [0,1]:
                    final_preds[idx_query_time] += preds[edge][idx_query_time]
            valid_sample_idxs.append(idx)     
            return final_preds
 
        num_valid_edge = 0
        for edge in refNode_rule_idx:
            # refNode_rule_idx[edge]: num of different rules satisfied for the current edge (idx_event_pos can be 0 or 1)
            if len(refNode_rule_idx[edge]) == 0:
                continue

            num_valid_edge += 1
            rule_idx_with_probs = []
            for idx_query_time in [0, 1]:
                for idx_event_pos in [0, 1]:
                    for idx_ref_time in [0, 1]:
                        probs = refNode_probs[edge][idx_event_pos][idx_query_time][idx_ref_time]
                        rule_idx_with_probs.append([[len(rule_idx), prob_dict[1], prob_dict[0]] for prob_dict in probs])  # [idx_event_in_batch, idx_rule, prob]
                        
            rule_idx.append(rule_idx_with_probs)  # [idx_event_in_batch, idx_rule, prob] * 4 * (1+flag_int)

        if num_valid_edge > 0:
            valid_sample_idxs.append(idx)
            refNode_sources.append(num_valid_edge)
            valid_query_relations += [query_relations[idx]] * num_valid_edge

        return None


    def _process_graph(self, idx_ls, random_walk_res, mode, query_relations):
        valid_sample_idxs, valid_query_relations, refNode_sources, valid_rule_idx = [], [], [], []
        merged_valid_rule_idx = []
        final_preds = [[] for _ in range(1+ int(self.option.flag_interval))]
        for idx, query_idx in enumerate(idx_ls):
            if query_idx not in random_walk_res:
                continue
            
            walk_res = random_walk_res[query_idx]
            rule_used_ls, rule_scores, refType_scores = self._create_rule_used_ls(walk_res, query_relations[idx], mode)
            flag_valid, refNode_probs, refNode_rule_idx, preds = self._populate_refNode_structures(walk_res, mode, rule_used_ls, rule_scores, refType_scores)

            if not flag_valid:
                continue
            
            preds = self._update_outputs(refNode_probs, refNode_rule_idx, query_relations, idx, 
                                         valid_sample_idxs, refNode_sources, valid_query_relations, valid_rule_idx, mode, preds)
            if mode == 'Test':
                for i in range(1+ int(self.option.flag_interval)):
                    final_preds[i].append(preds[i])
    

        # For training, we create all matrixes; for test, we directly calculate the probs.
        if mode == 'Train':    
            # Finalizes the structure of refNode_sources based on rule_idx
            # refNode_sources: edges from which sample
            refNode_num = 0
            for i in range(len(refNode_sources)): # len(refNode_sources): batch_size
                sources = np.zeros((len(valid_rule_idx), ))
                sources[refNode_num: refNode_num + refNode_sources[i]] = 1
                
                refNode_num += refNode_sources[i]
                refNode_sources[i] = sources.copy()

            for j in range(len(valid_rule_idx[i])):
                output = []
                for i in range(len(valid_rule_idx)):
                    output += valid_rule_idx[i][j]
                merged_valid_rule_idx.append(np.array(output))
        else:
            for i in range(1+ int(self.option.flag_interval)):
                if len(final_preds[i]) > 0:
                    final_preds[i] = np.vstack(final_preds[i])

        return valid_sample_idxs, valid_query_relations, refNode_sources, merged_valid_rule_idx, final_preds


    def create_graph(self, idx_ls, mode):
        random_walk_res, input_intervals_dict = self._initialize_walk_res(idx_ls, mode)
        query_edges, query_intervals = self._load_queries(idx_ls)
        query_relations = [query[1] for query in query_edges]

        if mode == 'Test':
            for i, data_idx in enumerate(idx_ls):
                if data_idx in input_intervals_dict:
                    query_intervals[i] = input_intervals_dict[data_idx].tolist()

        valid_sample_idxs, valid_query_relations, refNode_sources, merged_valid_rule_idx, final_preds = self._process_graph(idx_ls, random_walk_res, mode, query_relations)
        return query_relations, valid_query_relations, refNode_sources, merged_valid_rule_idx, valid_sample_idxs, query_intervals, [], [], final_preds




class TEKG_timestamp_acc_ver(TEKG_params):
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

        preprocessor = Data_preprocessor()

        # Assuming create_inputs_v4 and other required methods are defined elsewhere within the class
        input_edge_probs = preprocessor.create_inputs_timestamp_ver(sample_dict, sample_inv_dict, None, self.data['pattern_ls'], self.data['timestamp_range'], 
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


    def create_graph(self, idx_ls, mode):
        """
        Create Temporal Edge Knowledge Graphs (TEKG) in batch with timestamp accuracy verification.
        
        Parameters:
        - idx_ls: List of indices for which TEKG needs to be generated.
        - mode: The mode of operation, e.g., 'Test' or 'Train'.
        
        Returns:
        A tuple containing processed TEKG components.
        """
        pattern_ls_fkt, stat_res_fkt = self._handle_option_shift()        
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
                for j in range(len(probs)):
                    probs[j].extend(sample_results['probs'][j])
                ref_time_ls.append(sample_results['ref_time'])

        refNode_source = self._process_refNode_source(refNode_source)
        res_random_walk = np.array([item for sublist in res_random_walk for item in sublist])

        return qq, query_rels, refNode_source, res_random_walk, probs, valid_sample_idx, input_intervals, input_samples, ref_time_ls




class TEKG_normal_ver(TEKG_params):
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





    def _initialize_probabilities(self, num_samples, num_entities, num_steps, timestamp_range, mode):
        """
        Initialize probabilities for events.

        Parameters:
        - num_samples: The number of samples.
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
                base_array = np.ones((num_samples, num_entities, num_steps)) / len(timestamp_range)
            else:
                base_array = np.tile(np.ones(num_entities) / len(timestamp_range), (num_samples, 1, num_steps))
        elif mode == 'Test':
            base_array = np.tile(np.ones((len(timestamp_range),)) / len(timestamp_range), (num_samples, num_entities, num_steps))
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


    def _unique_edges(self, edges):
        """Return unique edges, ensuring no duplicates."""
        return np.unique(edges, axis=0)


    def _assign_probability(self, i, prob, prob_type, j, cur_edge_idx, step, ts_probs, te_probs):
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


    def _update_probability_by_step(self, i, edge, j, step, cur_edge_idx, event, mode, ts_probs, te_probs):
        flag_valid = 0
        for prob_type_key, probs_event in zip(['ts', 'te'], [0, 1]):
            if len(event[edge][probs_event][j][step]) > 0:
                selected_prob = self._select_probability(event[edge][probs_event][j][step], mode)
                self._assign_probability(i, selected_prob, prob_type_key, j, cur_edge_idx, step, ts_probs, te_probs)
                flag_valid = 1
        return flag_valid


    def _update_probability_without_step(self, i, edge, j, cur_edge_idx, event, mode, ts_probs, te_probs):
        flag_valid = 0
        for prob_type_key, probs_event in zip(['ts', 'te'], [0, 1]):
            if len(event[edge][probs_event][j]) > 0:
                selected_prob = self._select_probability(event[edge][probs_event][j], mode, aggregate=True)
                self._assign_probability(i, selected_prob, prob_type_key, j, cur_edge_idx, None, ts_probs, te_probs)
                flag_valid = 1
        return flag_valid


    def _update_probabilities_for_edge(self, i, edge, event, cur_edge_idx, num_step, mode, ts_probs, te_probs, ref_event_idx_ts, ref_event_idx_te):
        flag_valid = 0
        for j in [0, 1]:
            if self.option.flag_ruleLen_split_ver:
                for l in range(num_step):
                    flag_valid |= self._update_probability_by_step(i, edge, j, l+1, cur_edge_idx, event, mode, ts_probs, te_probs)
            else:
                flag_valid |= self._update_probability_without_step(i, edge, j, cur_edge_idx, event, mode, ts_probs, te_probs)
            ref_event_idx_ts[cur_edge_idx] = 1
            ref_event_idx_te[cur_edge_idx] = 1
        return flag_valid


    def _update_event_probabilities(self, idx_first_or_last_event, probs, flag_use_batch_graph, batch_edges, batch_edges_idx_cmp, TEKG_nodes, 
                                    num_step, mode, ts_probs, te_probs, ref_event_idx):
        # probs[data_idx][idx_first_or_last_event][str_tuple(edge)][2*idx_ts_or_te + idx_ref_time_ts_or_te]
        flag_valid = 0  # Initialize flag_valid
     
        for edge in probs[idx_first_or_last_event]:
            cur_edge_idx = batch_edges_idx_cmp[batch_edges.tolist().index([edge[j] for j in [0,1,4,2,3]])] if flag_use_batch_graph else \
                           TEKG_nodes.tolist().index([edge[j] for j in [0,1,4,2,3]])
            


        return flag_valid


    def create_graph(self, idx_ls, mode, edges, batch_edges_ori, batch_edges_idx_cmp):
        num_step = self.option.num_step-1
        mdb = copy.copy(self.data['mdb'])
        connectivity = copy.copy(self.data['connectivity'])
        TEKG_nodes = copy.copy(self.data['TEKG_nodes'])
        probs = copy.copy(self.data['random_walk_res'])

        flag_use_batch_graph = False
        if (mdb is None) or (connectivity is None) or (TEKG_nodes is None):
            flag_use_batch_graph = True

        query_edges, input_intervals = [], []
        batch_edges = [] if flag_use_batch_graph else None

        for data_idx in idx_ls:
            file_path = "{}{}_train_query_{}.json".format(self.data['path'], self.data['dataset_name'], data_idx)
            with open(file_path, 'r') as f:
                data = json.load(f)
            self._process_json_data(data, query_edges, input_intervals, batch_edges)


        if flag_use_batch_graph:
            batch_edges = self._unique_edges(batch_edges)
            batch_edges_inv = self._inverse_edges(batch_edges, self.data['num_rel'])
            batch_edges = self._unique_edges(np.vstack((batch_edges, batch_edges_inv)))  # Combine and get unique edges again
            
            assert len(batch_edges) <= self.data['num_entity'] // 2, 'Increase num_entity or reduce edges.'
            
            mdb = self._build_mdb(batch_edges, self.data['num_entity'], self.data['num_rel'])
            connectivity = self._calculate_connectivity(batch_edges, self.data['num_entity'])


        if probs is None:
            preprocessor = Data_preprocessor()
            output = preprocessor.create_inputs_interval_ver(edges=edges, path=self.data['path'], 
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
                                            flag_rule_split=self.option.flag_ruleLen_split_ver)

            probs = output[-1]

            if mode == 'Test':
                input_intervals_dict = output[-2]

                for i, data_idx in enumerate(idx_ls):
                    data_idx -= self.data['num_samples_dist'][1]
                    if data_idx in input_intervals_dict:
                        input_intervals[i] = input_intervals_dict[data_idx].tolist()



        qq = [query[1] for query in query_edges]
        hh = [batch_edges_ori.tolist().index(query) for query in query_edges] if flag_use_batch_graph else [TEKG_nodes.tolist().index(query) for query in query_edges]
        tt = [h + self.data['num_entity']//2 for h in hh]


        ts_probs_first_event, te_probs_first_event, ts_probs_last_event, te_probs_last_event = self._initialize_probabilities(
                                            len(idx_ls), self.data['num_entity'], num_step, self.data['timestamp_range'], mode
                                        )

        valid_sample_idx, valid_ref_event_idx_ts_first, valid_ref_event_idx_ts_last, valid_ref_event_idx_te_first, valid_ref_event_idx_te_last = [], [], [], [], []

        for (i, data_idx) in enumerate(idx_ls):
            flag_valid = 0
            if mode == 'Test':
                data_idx -= self.data['num_samples_dist'][1]

            if data_idx not in probs:
                continue

            ref_event_idx = np.zeros((4, self.data['num_entity']))

            flag_valid |= self._update_event_probabilities(i, probs[data_idx]['0'], flag_use_batch_graph, batch_edges, 
                                                            batch_edges_idx_cmp, TEKG_nodes, num_step, mode, 
                                                            ts_probs_first_event, te_probs_first_event, ref_event_idx_ts_first, ref_event_idx_te_first)

            flag_valid |= self._update_event_probabilities(probs[data_idx]['1'], flag_use_batch_graph, batch_edges, 
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