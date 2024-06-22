import sys
import os
import numpy as np
import itertools
import json
from utlis import *
from gadgets import *



class Base(object):
    def __init__(self, option, data):
        self.option = option
        self.data = data

        dataset_index = ['wiki', 'YAGO', 'icews14', 'icews05-15', 'gdelt100'].index(self.data['short_name'])

        # Parameters for the exponential distribution.
        # Only used in icews and gdelt datasets.
        self.weight_exp_dist = [None, None, 0.01, 0.01, 0.05][dataset_index]
        self.scale_exp_dist = [None, None, 5, 10, 100][dataset_index]
        self.offset_exp_dist = [None, None, 0, 0, 0][dataset_index]

        # Here we use all nodes to obtain the ground truth. The validation and test set are masked in random walk results.
        self.nodes = np.vstack((self.data['train_nodes'], self.data['valid_nodes'], self.data['test_nodes']))
    

    def _convert_walk_res_into_probs(self, idx_ls, mode):
        '''
        Given the idx_ls, either read the pre-processing results or do the pre-processing right now

        Parameters:
        - idx_ls: The list of sample indices for which to generate the TEKG.
        - mode: The mode of operation ('Train' or  'Valid' or 'Test').
        
        Returns:
        - prob_dict: Probabilities of the query time given reference nodes and rules. 
        - query_time: the query time for each sample.
        '''
        assert mode in ['Train', 'Valid', 'Test']
        
        if mode == 'Train':
            # instead of reading all the results, we read the results for the current batch
            prob_dict = {}
            query_time = {}
            for data_idx in idx_ls:
                filepath = '../output/{}/process_res/{}_idx_{}_input.json'.format(self.data['short_name'], self.data['short_name'], data_idx)
                if not os.path.exists(filepath):
                    continue
                with open(filepath) as f:
                    data = json.load(f)
 
                query_time[data_idx] = data[0]
                prob_dict[data_idx] = data[1]    
        else:
            # for test mode, there is no pre-processing
            preprocessor = Data_Processor()
            output = preprocessor.prepare_inputs(nodes=self.nodes, res_path=self.data['walk_res_path'], 
                                                 dataset=self.data['short_name'], 
                                                 idx_ls=idx_ls, pattern_ls=self.data['pattern_ls'], 
                                                 timestamp_range=self.data['timestamp_range'],
                                                 ts_stat_ls=self.data['ts_stat_ls'], 
                                                 te_stat_ls=self.data['te_stat_ls'],
                                                 targ_rel=None, mode=mode,
                                                 flag_time_shift=self.option.shift, 
                                                 show_tqdm=False, probs_normalization=True,
                                                 flag_interval=self.option.flag_interval)
            query_time, prob_dict = output[0], output[1]

        return prob_dict, query_time


    def _merge_list_inside(self, ori_ls):
        '''
        Given a list, where each element has the same number of lists, we want to merge the inside lists at the same position.
        E.g. [[ls1, ls2], [ls3, ls4]] -> [ls1 + ls3, ls2 + ls4]
        '''
        if len(ori_ls) == 0:
            return []
        
        merged_ls = []
        for j in range(len(ori_ls[0])):
            output = []
            for i in range(len(ori_ls)):
                output += ori_ls[i][j]
            merged_ls.append(np.array(output))
        return merged_ls


    def _merge_array_inside(self, ori_ls):
        '''
        Given a list, where each element has the same number of arrays, we want to merge the inside arrays at the same position.
        E.g. [[array1, array2], [array3, array4]] -> [np.vstack([array1, array3]), np.vstack([array2, array4])]
        '''
        if len(ori_ls) == 0:
            return []
        
        merged_ls = []
        for j in range(len(ori_ls[0])):
            output = []
            for i in range(len(ori_ls)):
                output.append(ori_ls[i][j])
            merged_ls.append(np.vstack(output))
        return merged_ls




class TEKG_family():
    def __init__(self, option, data):
        '''
        Select the appropriate TEKG version.
        '''
        if option.flag_acceleration:
            self.graph = TEKG_fast_ver(option, data)
        else:
            self.graph = TEKG(option, data)




class TEKG_fast_ver(Base):
    def __init__(self, option, data, call_by_TEKG=False):
        super(TEKG_fast_ver, self).__init__(option, data)
        # call_by_TEKG: 
        #   In TEKG, we need to distinguish one event at different event pos, i.e., 
        #   if an event is both the first and the last event, we consider it as two events.
        #   Also, we need to prepare all the data during training if call_by_TEKG is True.
        self.call_by_TEKG = call_by_TEKG
        

    def _load_queries(self, idx_ls):
        '''
        Load the queries for the given idx_ls
        '''
        query_edges, query_time = [], []
        for data_idx in idx_ls:
            file_path = os.path.join(self.data['walk_res_path'], "%s_idx_%s.json" % (self.data['short_name'], data_idx))
            if os.path.exists(file_path):
                with open(file_path, 'r') as file:
                    json_data = json.load(file)
                cur_query = json_data['query']
                query_edges.append(cur_query)
                query_time.append(cur_query[3:])

        query_rel = [query[1] for query in query_edges]
        return query_edges, query_rel, query_time


    def _create_selected_rule_ls(self, walk_res, query_rel, mode, max_rule_num=20):
        '''
        For each sample, we choose a fixed number of rules to use according to their scores.
        Previous max_rule_num: 100 (Found not necessary.)
        '''
        # No prior for training and no need when called by TEKG.
        if mode == 'Train' or self.call_by_TEKG:
            return None, None, None

        selected_rule_ls = []
        for idx_event_pos in [0,1]:
            for edge in walk_res[str(idx_event_pos)]:
                for idx_query_time in [0,1]:
                    for idx_ref_time in [0,1]:
                        if (not self.option.flag_interval) and ((idx_query_time != 0) or (idx_ref_time != 0)):
                            continue
                        idx_complete = idx_query_time*2 + idx_ref_time
                        probs = walk_res[str(idx_event_pos)][edge][str(idx_complete)]
                        for prob_dict in probs:
                            if prob_dict[1] not in selected_rule_ls:
                                selected_rule_ls.append(prob_dict[1])

   
        # read rule score and sort the rules
        path_suffix = '' if not self.option.shift else '_time_shifting'
        cur_path = '../output/{}/rule_scores/{}{}_rel_{}.json'.format(self.option.dataset, self.option.dataset, path_suffix, query_rel)
        rule_scores, refType_scores = None, None
        if os.path.exists(cur_path):
            with open(cur_path) as file:
                data = json.load(file)

            rule_scores = data['rule_scores']
            refType_scores = data["refType_scores"]

            # select the top k rules
            selected_rule_ls = sorted(selected_rule_ls, key=lambda x: rule_scores[x], reverse=True)
            selected_rule_ls = selected_rule_ls[:max_rule_num]
            
            # normalize the rule scores
            rule_scores = [0 if idx not in selected_rule_ls else score for idx, score in enumerate(rule_scores)]
            rule_scores = [score/(sum(rule_scores) + 1e-20) for score in rule_scores]
        
        return selected_rule_ls, rule_scores, refType_scores


    def _build_refNode_structures(self, walk_res, mode, selected_rule_ls=None, rule_scores=None, refType_scores=None):
        '''
        For each sample, populates refNode_probs and refNode_rule_idx based on random_walk_res
        '''
        flag_valid = False
        refNode_probs, refNode_rule_idx, preds = {}, {}, {}
        for idx_event_pos in [0,1]:
            for edge in walk_res[str(idx_event_pos)]:
                if (eval(edge)[2] == 9999) or (self.option.flag_interval and eval(edge)[3] == 9999):
                    # we don't need unknown time events as reference events
                    continue
                if edge not in refNode_probs:
                    refNode_probs[edge] = {0: {0: {0:[], 1:[]}, 1: {0:[], 1:[]}}, 1: {0: {0:[], 1:[]}, 1: {0:[], 1:[]}}}  # [idx_event_pos][idx_query_time][idx_ref_time]
                  
                    refNode_rule_idx[edge] = []
                    if mode == 'Test' and (not self.call_by_TEKG):
                        if self.option.flag_interval:
                            preds[edge] = {0: 0, 1: 0}  # query ts or te
                        else:
                            preds[edge] = {0: 0} # query ts

                # for each event, we have a list of probabilities
                for idx_query_time in [0,1]:
                    for idx_ref_time in [0,1]:
                        if (not self.option.flag_interval) and ((idx_query_time != 0) or (idx_ref_time != 0)):
                            continue

                        idx_complete = idx_query_time*2 + idx_ref_time
                        probs = walk_res[str(idx_event_pos)][edge][str(idx_complete)]

                        # Given the query, the event pos and the reference event, more than one rules are satisfied.
                        for prob_dict in probs:
                            if (selected_rule_ls is not None) and (prob_dict[1] not in selected_rule_ls):
                                continue
                            
                            refNode_probs[edge][idx_event_pos][idx_query_time][idx_ref_time].append(prob_dict)
                            refNode_rule_idx[edge].append(prob_dict[1])
                            
                            if mode == 'Test' and (rule_scores is not None) and (refType_scores is not None) and (not self.call_by_TEKG):
                                if self.option.flag_interval:
                                    # refType_scores: [(first_event_ts, first_event_te), (last_event_ts, last_event_te), (first_event, last_event)]
                                    prob_event_pos = refType_scores[3*idx_query_time + 2][idx_event_pos]
                                    prob_ref_time = refType_scores[3*idx_query_time + idx_event_pos][idx_ref_time]
                                    preds[edge][idx_query_time] += prob_event_pos * prob_ref_time * rule_scores[prob_dict[1]] * np.array(prob_dict[0])
                                else:
                                    prob_event_pos = refType_scores[0][idx_event_pos]
                                    preds[edge][idx_query_time] += prob_event_pos * rule_scores[prob_dict[1]] * np.array(prob_dict[0])

                            flag_valid = True

        return flag_valid, refNode_probs, refNode_rule_idx, preds


    def _update_outputs(self, refNode_probs, refNode_rule_idx, query_relations, idx, valid_sample_idxs, 
                              refNode_nums, query_rel_flatten, probs, refEdges, preds, mode):
        '''
        Updates the output structures with processed data from refNode_probs and refNode_rule_idx
        '''
        # This is a placeholder for the actual logic
        if self.option.flag_ruleLen_split_ver:
            print('Todo')
            pass
        
        if mode == 'Test' and (len(preds) > 0) and (not self.call_by_TEKG):
            # Given the sample, merge probs from different events.
            final_preds = {0: 0, 1: 0} if self.option.flag_interval else {0: 0}
            for edge in preds:
                for idx_query_time in [0,1]:
                    if not self.option.flag_interval and idx_query_time != 0:
                        continue
                    final_preds[idx_query_time] += preds[edge][idx_query_time]
              
            valid_sample_idxs.append(idx)     
            return final_preds
 

        num_valid_edge = 0
        for edge in refNode_rule_idx:
            # refNode_rule_idx[edge]: num of different rules satisfied for the current edge (idx_event_pos can be 0 or 1)
            if len(refNode_rule_idx[edge]) == 0:
                continue
            
            cnt_edge_num = 0
            probs_with_rule_idx = []
            for idx_query_time in [0, 1]:
                for idx_event_pos in [0, 1]:
                    num_rules = 0
                    for idx_ref_time in [0, 1]:
                        if (not self.option.flag_interval) and ((idx_query_time != 0) or (idx_ref_time != 0)):
                            continue
                        cur_probs = refNode_probs[edge][idx_event_pos][idx_query_time][idx_ref_time]
                        num_rules += len(cur_probs)
                        probs_with_rule_idx.append([[len(probs), prob_dict[1], prob_dict[0]] for prob_dict in cur_probs])  # [idx_event_in_batch, idx_rule, prob]

                    # If there are rules for the current event_pos, we add the edge to refEdges.
                    # We only consider it for tqs since we don't want to add the same edge multiple times.
                    if idx_query_time == 0 and num_rules > 0:
                        refEdges.append([edge, idx_event_pos])
                        cnt_edge_num += 1
            
            # If we do not want to distinguish the event pos, we only consider the edge once.
            cnt_edge_num = min(cnt_edge_num, 1) if not self.call_by_TEKG else cnt_edge_num
            num_valid_edge += cnt_edge_num

            probs.append(probs_with_rule_idx)  # [idx_event_in_batch, idx_rule, prob] * 8 if flag_interval else * 2


        if num_valid_edge > 0:
            # update these global variables.
            valid_sample_idxs.append(idx)
            query_rel_flatten += [query_relations[idx]] * num_valid_edge
            refNode_nums.append(num_valid_edge)
        
        return None


    def _obtain_one_hot_form(self, source_ls):
        '''
        Convert the source_ls into one-hot form.
        source_ls: [num of rules in each sample], e.g. [2, 13]
        one_hot: one hot form of source_ls, e.g. [[1, 1, 0, 0, 0, ...], [0, 0, 1, 1, 1, ...]]
        '''
        refNode_num = 0
        num_nodes = sum(source_ls)
        one_hot = []
        for i in range(len(source_ls)): # len(source_ls): batch_size
            sources = np.zeros((num_nodes, ))
            sources[refNode_num: refNode_num + source_ls[i]] = 1
            one_hot.append(sources)
            refNode_num += source_ls[i]
            
        return one_hot


    def _build_graph(self, idx_ls, random_walk_res, query_relations, mode):
        '''
        Process the walk res to create the structured data.
        '''
        valid_sample_idx, query_rel_flatten, refNode_nums, probs, refNodes = [], [], [], [], []
        final_preds = [[] for _ in range(1+ int(self.option.flag_interval))]
        for idx, query_idx in enumerate(idx_ls):
            if query_idx not in random_walk_res:
                continue
            
            walk_res = random_walk_res[query_idx]
            selected_rule_ls, rule_scores, refType_scores = self._create_selected_rule_ls(walk_res, query_relations[idx], mode)
            flag_valid, refNode_probs, refNode_rule_idx, preds = self._build_refNode_structures(walk_res, mode, selected_rule_ls, rule_scores, refType_scores)

            if not flag_valid:
                continue
            
            preds = self._update_outputs(refNode_probs, refNode_rule_idx, query_relations, idx, 
                                         valid_sample_idx, refNode_nums, query_rel_flatten, 
                                         probs, refNodes, preds, mode)
            if mode == 'Test' and (not self.call_by_TEKG):
                for i in range(1+ int(self.option.flag_interval)):
                    final_preds[i].append(preds[i])

        # For training, we create all matrixes; for test, we directly calculate the probs.
        if mode == 'Train' or self.call_by_TEKG:    
            # Finalizes the structure of refNode_sources based on rule_idx
            # refNode_sources: node from which sample
            refNode_sources = self._obtain_one_hot_form(refNode_nums)
            probs = self._merge_list_inside(probs)
        else:
            # RefNode_sources in test mode are too large.
            refNode_sources = []
            for i in range(1+ int(self.option.flag_interval)):
                if len(final_preds[i]) > 0:
                    final_preds[i] = np.vstack(final_preds[i])

        return valid_sample_idx, query_rel_flatten, refNode_sources, probs, refNodes, final_preds


    def create_graph(self, idx_ls, mode):
        '''
        Create the graph for the given idx_ls and mode.

        Parameters:
            idx_ls: The list of indices for which to generate the TEKG.
            mode: The mode of operation ('Train' or  'Valid' or 'Test').

        Returns:
            query_rel: The original query relations.
            query_rel_flatten: We duplicate the query relations for each valid node.
            refNode_sources: The sources of the reference nodes.
            probs: The probabilities for query time given the reference nodes.
            refEdges: The reference nodes.
            valid_sample_idx: The local indices of valid samples.
            query_time: The query time.
            query_samples: The query samples.
            final_preds: The final predictions.
        '''
        random_walk_res, input_intervals_dict = self._convert_walk_res_into_probs(idx_ls, mode)
        query_samples, query_rel, query_time = self._load_queries(idx_ls)

        if mode == 'Test' and (not self.call_by_TEKG):
            for i, data_idx in enumerate(idx_ls):
                if data_idx in input_intervals_dict:
                    query_time[i] = input_intervals_dict[data_idx].tolist()

        output = self._build_graph(idx_ls, random_walk_res, query_rel, mode)
        valid_sample_idx, query_rel_flatten, refNode_sources, probs, refNodes, final_preds = output

        return query_rel, query_rel_flatten, refNode_sources, probs, refNodes, valid_sample_idx, query_time, query_samples, final_preds




class TEKG(Base):
    def _build_connectivity_rel(self, nodes, nodes_idx, num_entity, num_rel):
        """
        Build the connectivity_rel mat based on current nodes.
        Nodes are augmented with inverse nodes.
        """
        connectivity_rel = {}
        for idx_rel in range(num_rel):
            # Find all the nodes that satisfy the relation.
            idx_nodes_cur_rel = nodes_idx[nodes[:, 1] == idx_rel].reshape((-1, 1))
            if idx_nodes_cur_rel.size == 0:
                connectivity_rel[idx_rel] = [[[0,0]], [0.], [num_entity, num_entity]]
            else:
                x, y = idx_nodes_cur_rel, idx_nodes_cur_rel
                connectivity_rel[idx_rel] = [np.hstack([x, y]).tolist(), [1.0 for _ in range(len(idx_nodes_cur_rel))], [num_entity, num_entity]]
        return connectivity_rel


    def _build_aug_nodes(self, nodes, num_entity):
        """
        Build the augmented nodes based on the current nodes.
        """
        nodes_inv = self._obtain_inverse_edges(nodes, self.data['num_rel'])
        nodes_aug = np.vstack((nodes, nodes_inv))
        nodes_idx = np.arange(len(nodes))
        nodes_idx_aug = np.hstack((nodes_idx, nodes_idx + num_entity//2))
        return nodes_aug, nodes_idx_aug


    def _build_connectivity_TR(self, nodes, nodes_idx, num_entity, num_TR):
        """
        Build the connectivity_TR mat based on current nodes.
        Nodes are augmented with inverse nodes.
        Different temporal relations (TRs) [0: ukn, 1: bf, 2: touch, 3: af]
        """
        connectivity_TR = {key: [] for key in range(num_TR)}    
        for entity in np.unique(np.hstack([nodes[:, 0], nodes[:, 2]])):
            # Find the node pairs that have the same entities as the subject and object.
            # Direction: b -> a
            b = nodes_idx[nodes[:, 2] == entity]
            a = nodes_idx[nodes[:, 0] == entity]
            combinations = np.array(list(itertools.product(a, b)))
            
            # Calculate the TRs for the node pairs.
            TRs = calculate_TR_mat_ver(nodes[combinations[:,0], 3:], nodes[combinations[:,1], 3:])

            # For different TRs, we store the corresponding node pairs with mask.
            for i in range(num_TR):
                mask = TRs == i if i else slice(None)
                connectivity_TR[i] += combinations[mask].tolist() 

        # Convert the output format.
        for idx_TR in range(num_TR):
            if connectivity_TR[idx_TR]:
                A = np.array(connectivity_TR[idx_TR])
                if A.size == 0:
                    connectivity_TR[idx_TR] = [[[0,0]], [0.], [num_entity, num_entity]]
                else:
                    A = np.unique(A, axis=0)
                    connectivity_TR[idx_TR] = [A.tolist(), [1.0 for _ in range(len(A))], [num_entity, num_entity]]
            else:
                connectivity_TR[idx_TR] = [[[0,0]], [0.], [num_entity, num_entity]]

        return connectivity_TR


    def _obtain_inverse_edges(self, edges, num_rel):
        """
        Inverse the edges and adjust relation IDs for symmetry.
        Edges: [entity1, relation, entity2, start_time, end_time]
        Inv edges: [entity2, inv_relation, entity1, start_time, end_time]
        """
        inv_edges = edges[:, [2, 1, 0, 3, 4]] if self.option.flag_interval else edges[:, [2, 1, 0, 3]]
        mask = edges[:, 1] < num_rel // 2
        inv_edges[mask, 1] += num_rel // 2
        inv_edges[~mask, 1] -= num_rel // 2
        return inv_edges


    def _process_walk_res(self, idx_ls, flag_use_batch_graph):
        query_nodes, query_time = [], []
        batch_nodes = [] if flag_use_batch_graph else None
        num_entity = None
        batch_nodes_idx = None
        for data_idx in idx_ls:
            file_path = "{}/{}_idx_{}.json".format(self.data['walk_res_path'], self.data['short_name'], data_idx)
            if not os.path.exists(file_path):
                continue
            with open(file_path, 'r') as f:
                data = json.load(f)

            cur_query = data['query']
            query_time.append(cur_query[3:])

            # We need to mask the query time during training.
            cur_query[3:] = [9999 for _ in range(len(cur_query)-3)]

            query_nodes.append(cur_query)
            if flag_use_batch_graph:
                batch_nodes.append(cur_query)
                for rule_len in range(1, 6):
                    walks = data.get(str(rule_len), [])
                    for walk in walks:
                        for t in range(rule_len):
                            node = walk[4*t:4*t+5] if self.option.flag_interval else walk[3*t:3*t+4]
                            node = [node[i] for i in [0, 1, 4, 2, 3]] if self.option.flag_interval else [node[i] for i in [0, 1, 3, 2]]
                            batch_nodes.append(node)
        
        if flag_use_batch_graph and len(batch_nodes)>0:
            # Convert into array and avoid repetition.
            batch_nodes = np.array(batch_nodes)
            batch_nodes = np.unique(batch_nodes, axis=0)
            num_entity = len(batch_nodes)*2
            batch_nodes, batch_nodes_idx = self._build_aug_nodes(batch_nodes, num_entity)

        return query_nodes, query_time, batch_nodes, batch_nodes_idx, num_entity


    def _select_probability(self, probs, mode, min_prob=1e-4):
        '''
        Select the probability for the current reference events (There might be multiple rules satisfied).
        '''
        if mode == 'Train':
            # min_prob: set minimum probability that make the training more stable.
            if self.option.prob_selection_for_training == 'max':
                return max(min_prob, max(probs))
            else:
                return max(min_prob, np.mean(probs))
        else:
            probs = np.array(probs) # shape: (num_rules, num_timestamp)
            return np.mean(probs, axis=0)
      

    def _calculate_distribution_score(self, extra_data, all_idx):
        '''
        Given all the related indices, we calculate the score for current distribution.
        '''
        data_idx, edge_idx, idx_query_time, idx_event_pos, idx_ref_time = all_idx
        final_state_vec, attn_refType = extra_data

        if final_state_vec is None or attn_refType is None:
            return 0.
        
        # We first select corresponding final state vector.
        selected_final_prob = final_state_vec[str((data_idx, idx_event_pos, edge_idx))]
        selected_attn_refType = attn_refType[str(data_idx)]
        
        if self.option.flag_interval:   
            # attn_refType: [tqs, tqe] X [(last_ts, last_te), (first_ts, first_te), (last_event, first_event)]
            score = selected_final_prob * selected_attn_refType[3*idx_query_time + 2][1-idx_event_pos] \
                                        * selected_attn_refType[3*idx_query_time + 1-idx_event_pos][idx_ref_time]
        else:
            # attn_refType: [(last_event, first_event)]
            score = selected_final_prob * selected_attn_refType[0][1-idx_event_pos]

        return score


    def _obtain_edge_idx_in_TEKG(self, edge, TEKG_nodes, TEKG_nodes_idx, num_entity, idx_event_pos):
        '''
        Given an edge, obtain its idex in TEKG.
        '''
        edge_idx = TEKG_nodes_idx[TEKG_nodes.tolist().index(edge)]

        # For the first event, we will do a reverse walk. Thus, we need to find the inv node in TEKG.
        if idx_event_pos == 0:
            edge_idx += num_entity//2
        return edge_idx


    def _update_event_probabilities(self, probs, idx_sample, data_idx, walk_res, num_entity, TEKG_nodes_aug, TEKG_nodes_idx_aug, mode, 
                                    stage, final_preds, extra_data):
        '''
        Given the walk res, update the probs for the current sample.
        '''
        flag_valid = 0
        ref_event_idx = {}
        preds = [0. for _ in range(1 + int(self.option.flag_interval))]
        for idx_event_pos in [0, 1]:
            # Format: walk_res[idx_event_pos][str_tuple(edge)][2*idx_query_time + idx_ref_time]
            for edge in walk_res[str(idx_event_pos)]:
                # The format we used in walk res is different from the original one.
                edge_reformatted = [eval(edge)[j] for j in [0,1,4,2,3]] if self.option.flag_interval else [eval(edge)[j] for j in [0,1,3,2]]

                # Obtain the edge index in TEKG.
                edge_idx = self._obtain_edge_idx_in_TEKG(edge_reformatted, TEKG_nodes_aug, TEKG_nodes_idx_aug, num_entity, idx_event_pos)

                for idx_query_time in [0, 1]:
                    for idx_ref_time in [0, 1]:
                        if (not self.option.flag_interval) and ((idx_query_time != 0) or (idx_ref_time != 0)):
                            continue
                        cur_probs = [item[0] for item in walk_res[str(idx_event_pos)][edge][str(2*idx_query_time + idx_ref_time)]]
                        if len(cur_probs) == 0:
                            continue
 
                        selected_prob = self._select_probability(cur_probs, mode)

                        # Use a composite index here to simplify the coding.
                        # When building the model, we first calculate the last event and then the first event, first tqs and then tqe.
                        idx_complete = 4*idx_query_time + 2*(1-idx_event_pos) + idx_ref_time if self.option.flag_interval else 1-idx_event_pos
                        
                        if mode == 'Train':
                            probs[idx_sample][idx_complete][edge_idx] = selected_prob
                        else:
                            # No need to update the probabilities for inference. Instead we use an online algorithm.    
                            if stage == 'obtain state vec':
                                ref_event_idx[(data_idx, idx_event_pos, edge_idx)] = 0.
                            else:
                                # During inference, we calculate the time prediction in an online manner.
                                all_idx = [data_idx, edge_idx, idx_query_time, idx_event_pos, idx_ref_time]
                                preds[idx_query_time] += selected_prob * self._calculate_distribution_score(extra_data, all_idx)
                        flag_valid = 1
        
        if flag_valid and stage == 'time prediction':                
            final_preds.append(preds)

        return flag_valid, ref_event_idx


    def _build_graph(self, idx_ls, query_nodes, TEKG_nodes, TEKG_nodes_idx, num_entity, walk_res, mode, stage, extra_data):
        '''
        Format the outputs for the current batch.
        '''
        qq = [query[1] for query in query_nodes]

        # In TEKG, the nodes are the events in TKG, thus, we use the index of the event in the TEKG_nodes list as the entity index.
        hh = [TEKG_nodes.tolist()[:len(TEKG_nodes)].index(query) for query in query_nodes]

        # In TEKG, the tail entity is the inv event of the head entity.
        # Thus, we add half the number of entities to the head entity index to get the tail entity index.
        tt = [h + num_entity//2 for h in hh]
 
        if self.option.flag_ruleLen_split_ver:
            num_step = self.option.num_step-1
            pass
        
        # For initializing the probabilities, we use a uniform distribution. The shape of the probabilities is [num_samples, num_cases, num_entity].
        # Notice that entity in TEKG is event in TKG.
        # idx_complete = 4*idx_query_time + 2*(1-idx_event_pos) + idx_ref_time
        num_cases = 8 if self.option.flag_interval else 2
        if mode == 'Train':
            probs = [[[1./len(self.data['timestamp_range']) for _ in range(num_entity)] for _ in range(num_cases)] for _ in range(len(idx_ls))]
        else:
            probs = [[[0. for _ in range(num_entity)] for _ in range(num_cases)] for _ in range(len(idx_ls))]

        # For the stage of time prediction in test mode, we only need to calculate the time prediction for each sample.
        if stage == 'time prediction':
            final_preds = []
        else:
            final_preds = None

        valid_sample_idx, valid_ref_event_idx = [], {}
        for (idx, data_idx) in enumerate(idx_ls):
            if data_idx not in walk_res:
                continue

            flag_valid, ref_event_idx = self._update_event_probabilities(probs, idx, data_idx, walk_res[data_idx], num_entity, TEKG_nodes, TEKG_nodes_idx, mode, 
                                                                         stage, final_preds, extra_data)

            if flag_valid:
                valid_sample_idx.append(idx)
                valid_ref_event_idx.update(ref_event_idx)

        if stage == 'time prediction' and len(final_preds)>0:
            final_preds = self._merge_array_inside(final_preds)

        return qq, hh, tt, probs, valid_sample_idx, valid_ref_event_idx, final_preds


    def _obtain_refNode_index(self, refEdges, refNode_sources, batch_nodes, batch_nodes_idx, num_entity):
        '''
        Given the refEdges, refNode_sources and batch_nodes, we want to obtain the refNode_index.
        
        Parameters:
        refEdges format: [[refEvent, idx_event_pos] for each ref event]
        refNode_sources: one-hot form which shows the events come from which sample, 
                    e.g. [[1, 1, 0, 0, 0], [0, 0, 1, 1, 1]]
                         refEdge 0 and 1 come from sample 0, refEdge 2, 3, 4 come from sample 1.
        batch_nodes: the nodes in the current batch.

        Return:
            refNode_index: [[idx_sample, idx_node] for each ref event]
                      e.g. [[0, 125], [0, 126], [1, 8], [1, 16], [1, 13]]
        '''
        refNode_index = []
        idx_sample = 0

        for (i, node_dict) in enumerate(refEdges):
            # node_dict format: [node, idx_event_pos] (0: first event, 1: last event)
            node = node_dict[0]
            node = eval(node)

            # change the edge format.
            node = [node[j] for j in [0, 1, 4, 2, 3]] if self.option.flag_interval else [node[j] for j in [0, 1, 3, 2]]
            
            node_idx = batch_nodes_idx[batch_nodes.tolist().index(node)] # Find its idx in TEKG.
            if node_dict[1] == 0:
                # We need to use the inv node for first event.
                node_idx += num_entity//2
            
            # find the next sample.
            while refNode_sources[idx_sample][i] == 0:
                idx_sample += 1
            refNode_index.append([idx_sample, node_idx])
        return refNode_index


    def create_graph(self, idx_ls, mode, stage=None, extra_data=None):
        '''
        Create the TEKG in batch.
        For test mode, we have two steps: 1) calculate final state vec only; 2) predict time given the final state vec.

        We show an example TEKG here:
        YAGO dataset train_idx: 91
        Local TEKG:
        Node 0: [[ 216    7  388 2012 2012]
        Node 1: [ 217   10  216 9999 9999]
        Node 2: [ 388   10 1629 1975 1975]
        Node 3: [1514    0  217 1945 1945]
        Node 4: [1514    4 1629 9999 9999]
        Node 5: [1514   14 1629 9999 9999]

        Node 6:  [ 388   17  216 2012 2012]
        Node 7:  [ 216    0  217 9999 9999]
        Node 8:  [1629    0  388 1975 1975]
        Node 9:  [ 217   10 1514 1945 1945]
        Node 10:  [1629   14 1514 9999 9999]
        Node 11:  [1629    4 1514 9999 9999]]

        head node: 4   [1514    4 1629 9999 9999]
        tail node: 10  [1629   14 1514 9999 9999]

        (Valid) path: we learn the prob dist of the second to last node in the path.
        4 -> 8 -> 6 ->  7 -> (9) -> 4 
        10 -> 3 -> 1 -> 0 -> (2) -> 10
        
        Random walk res:
        first event: node 3 (inv: node 9)   last event: node 2
        Note: we need to find the inv node in TEKG for the first event.

            
        Parameters:
            idx_ls: The list of indices for which to generate the TEKG.
            mode: The mode of operation ('Train' or  'Valid' or 'Test').
            stage: The stage of operation ('obtain state vec' or 'time prediction').
            extra_data: The extra data for the current batch (final_state_vec, attn_refType).
        
        Returns:
            qq: The query relations.
            hh: The head nodes.
            tt: The tail nodes.
            connectivity_rel: The connectivity matrix for the relations.
            connectivity_TR: The connectivity matrix for the temporal relations.
            probs: The probabilities for query time given the reference nodes.
            valid_local_idx: The local indices of valid samples.
            valid_ref_idx: The indices of valid reference nodes.
            inputs_for_enhancement: The inputs for state vector enhancement.
            query_time: The query time.
            query_samples: The query samples.
            final_preds: The final predictions.
        '''
        assert mode in ['Train', 'Valid', 'Test']
        assert stage in [None, 'obtain state vec', 'time prediction']

        connectivity_rel = self.data['connectivity_rel']
        connectivity_TR = self.data['connectivity_TR']
        TEKG_nodes = self.data['TEKG_nodes']
        TEKG_nodes_idx = self.data['TEKG_nodes_idx']
        walk_res = self.data['random_walk_res']

        flag_use_batch_graph = False
        if (connectivity_rel is None) or (connectivity_TR is None) or (TEKG_nodes is None):
            # We can load the global TEKG in advance and use it for all batches. But it is too time-consuming.
            # Instead we use the batch graph for each batch.
            flag_use_batch_graph = True
        
        query_samples, query_time, batch_nodes, batch_nodes_idx, num_entity = self._process_walk_res(idx_ls, flag_use_batch_graph)

        batch_nodes = TEKG_nodes if batch_nodes is None else batch_nodes
        batch_nodes_idx = TEKG_nodes_idx if batch_nodes_idx is None else batch_nodes_idx
        num_entity = len(batch_nodes) if num_entity is None else num_entity

        if batch_nodes is None or len(batch_nodes) == 0:
            return None, None, None, None, None, None, [], None, None, None, None, None    


        if flag_use_batch_graph:
            # We use (augmented) batch_nodes to build (augmented) connectivity_rel and (augmented) connectivity_TR.
            connectivity_rel = self._build_connectivity_rel(batch_nodes, batch_nodes_idx, num_entity, self.data['num_rel'])
            connectivity_TR = self._build_connectivity_TR(batch_nodes, batch_nodes_idx, num_entity, self.data['num_TR'])

        
        if walk_res is None:
            walk_res, query_time_dict = self._convert_walk_res_into_probs(idx_ls, mode)

            if mode == 'Test':
                for i, data_idx in enumerate(idx_ls):
                    if data_idx in query_time_dict:
                        query_time[i] = query_time_dict[data_idx].tolist()


        if stage == 'time prediction':
            final_state_vec, attn_refType = extra_data
            # We do not have the final state vec and attn_refType for the current batch.
            if final_state_vec is None or attn_refType is None:
                return None, None, None, None, None, None, [], None, None, query_time, None, []


        # For probs and valid_refNode_idx, we use a composite index here to simplify the coding.
        # idx_complete = 4*idx_query_time + 2*(1-idx_event_pos) + idx_ref_time
        #           or = 1-idx_event_pos
        qq, hh, tt, probs, valid_sample_idx, valid_refNode_idx, final_preds = self._build_graph(idx_ls, query_samples, batch_nodes, batch_nodes_idx, num_entity, 
                                                                                                walk_res, mode, stage, extra_data)
 
        inputs_for_enhancement = []
        if self.option.flag_state_vec_enhancement:
            # Use fast ver graph to prepare data.
            graph = TEKG_fast_ver(self.option, self.data, call_by_TEKG=True)
            
            # In TEKG, we need to distinguish one event at different event pos, i.e., 
            # if an event is both the first and the last event, we consider it as two events.
            # Also, we need to prepare all the data during training if call_by_TEKG is True.
            _, query_rels, refNode_sources, random_walk_res, refEdges, _, _, _, _ = graph.create_graph(idx_ls, mode)

            refNode_index = self._obtain_refNode_index(refEdges, refNode_sources, batch_nodes, batch_nodes_idx, num_entity)

            # collect the inputs for enhancement.
            # query_rels: query relations for each ref event. 
            #       e.g. [4, 4, 4, 6, 6, 6, 6, 6] 
            #            Two samples (rel: [4, 6]): the first one has three ref events and the second has five ref events.
            # random_walk_res: random walk results for each ref event. 
            #            [[[idx_refEvent, idx_rule, prob_for_query_time] for _ in range(num_possible_comb)] for _ in range(num_cases)]
            #            num_cases = 8 if self.option.flag_interval else 2
            #       e.g. [[0, 154, 0.13], [0, 254, 0.17], [1, 17, 0.01], [2, 79, 0.04]] 
            # refNode_index: [[idx_sample, idx_node] for each ref event]
            #       e.g.  [[0, 125], [0, 126], [0, 49], [1, 13], [1, 52], [1, 75], [1, 148], [1, 150]]
            inputs_for_enhancement = [query_rels, random_walk_res, refNode_index]

        return qq, hh, tt, connectivity_rel, connectivity_TR, probs, valid_sample_idx, valid_refNode_idx, inputs_for_enhancement, query_time, query_samples, final_preds