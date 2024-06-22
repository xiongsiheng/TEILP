import numpy as np 
import tensorflow as tf
from utlis import *



class Learner(object):
    def __init__(self, option, data):
        self.seed = option.seed
        self.num_step = option.num_step
        self.num_layer = option.num_layer
        self.rnn_state_size = option.rnn_state_size
        
        self.norm = not option.no_norm
        self.thr = option.thr
        self.dropout = option.dropout

        self.num_relation = data['num_rel']
        self.num_TR = data['num_TR']
        self.num_rule = option.num_rule
        self.num_timestamp = len(data['timestamp_range'])
        self.num_query = data['num_query']
        self.query_embed_size = option.query_embed_size
        

        self.flag_int = option.flag_interval
        self.flag_rel_TR_split = option.different_states_for_rel_and_TR
        self.flag_ruleLen_split = option.flag_ruleLen_split_ver
        self.flag_state_vec_enhance = option.flag_state_vec_enhancement
        self.flag_acceleration = option.flag_acceleration

        # weight for the state vector enhancement (w/ shallow layers)
        # final state = gamma * state + (1-gamma) * enhance_state
        self.gamma = 0.7

        # To make the learning more stable, we scale the final prob dynamically.
        # If the prob is greater than self.prob_scaling_factor[0], we scale it by self.prob_scaling_factor[1].
        self.prob_scaling_factor = [0.1, 0.5]
 
        np.random.seed(self.seed)

        if option.flag_ruleLen_split_ver:
            print('Todo: ruleLen_split_ver')
            pass
        else:
            if self.flag_acceleration:
                self._build_comp_graph_fast_ver()
            else:
                self._build_comp_graph()


    def _random_uniform_unit(self, r, c):
        bound = 6./ np.sqrt(c)
        init_matrix = np.random.uniform(-bound, bound, (r, c))
        init_matrix = np.array(map(lambda row: row / np.linalg.norm(row), init_matrix))
        return init_matrix


    def _clip_if_not_None(self, g, v, low, high):
        if g is not None:
            return (tf.clip_by_value(g, low, high), v)
        else:
            return (g, v)


    def _init_ruleLen_embed(self, pattern_ls):
        self.ruleLen_embedding = [np.zeros((self.num_query, self.num_rule))] * (self.num_step-1)

        for rel in pattern_ls:
            ruleLen = np.array([len(rule.split(' '))//2 for rule in pattern_ls[rel]])
            ruleLen = np.hstack((ruleLen, np.zeros((self.num_rule - len(pattern_ls[rel]),))))
            for l in range(1, self.num_step):
                ruleLen_cp = ruleLen.copy()
                ruleLen_cp[~(ruleLen == l)] = 0
                ruleLen_cp[ruleLen == l] = 1
                self.ruleLen_embedding[l-1][int(rel)] = ruleLen_cp


    def _scale_final_prob(self, prob):
        # Calculate if any entry is greater than self.prob_scaling_factor[0]
        condition = tf.reduce_any(tf.greater(prob, self.prob_scaling_factor[0]))

        # Use tf.cond to set prob_scaling_factor
        prob_scaling_factor = tf.cond(
            condition,
            lambda: tf.constant(self.prob_scaling_factor[1]),
            lambda: tf.constant(self.prob_scaling_factor[0])
        )

        # scaling the loss to make the learning more stable.
        prob /= prob_scaling_factor
        return prob


    def _build_rnn_inputs(self, num_entity):
        '''
        Build RNN inputs for rule score learning.
        '''
        # Tail and head entities in TEKG graph which are used in random walk
        self.tails = tf.placeholder(tf.int32, [None])
        self.heads = tf.placeholder(tf.int32, [None])
        self.targets = [tf.one_hot(indices=self.tails, depth=num_entity), tf.one_hot(indices=self.heads, depth=num_entity)]

        # We distinguish different queries based on query relations.
        self.query_rels = tf.placeholder(tf.int32, [None])
        self.queries = tf.placeholder(tf.int32, [None, self.num_step])
        self.query_embedding_params = tf.Variable(self._random_uniform_unit(
                                                      self.num_query + 1, # <END> token 
                                                      self.query_embed_size), 
                                                      dtype=tf.float32, name="query_embedding_params")

        rnn_inputs = tf.nn.embedding_lookup(self.query_embedding_params, self.queries)        
        self.rnn_inputs = [tf.reshape(q, [-1, self.query_embed_size]) for q in tf.split(rnn_inputs, self.num_step, axis=1)]


    def _build_rnn_outputs(self, num_sample):
        # Instead of using multiple RNNs for different relations and TRs, we use a single RNN with a longer hidden state vector.
        # We need to distinguish the hidden states for first event and last event.
        # Also, if we distinguish the hidden states for relations and TRs, we need to double the hidden state vector.
        cellLen = self.rnn_state_size*4 if self.flag_rel_TR_split else self.rnn_state_size*2
        cell = tf.nn.rnn_cell.LSTMCell(cellLen, state_is_tuple=True)
        self.cell = tf.nn.rnn_cell.MultiRNNCell([cell] * self.num_layer, 
                                                    state_is_tuple=True)
        init_state = self.cell.zero_state(num_sample, tf.float32)
        self.rnn_outputs, _ = tf.contrib.rnn.static_rnn(self.cell, self.rnn_inputs, initial_state=init_state)


    def _build_attn_rel_and_TR(self, query_specific=False):
        # Different channels: [last_event, first_event]
        # We use different weights and biases for different query relations. We found it unstable to share the weights.
        dim = self.num_query if query_specific else 1
        self.W_rel = tf.Variable(np.random.randn(dim, self.rnn_state_size, 2, self.num_relation), dtype=tf.float32, name="W_rel")
        self.b_rel = tf.Variable(np.zeros((dim, 2, self.num_relation)), dtype=tf.float32, name="b_rel")
        
        self.W_TR = tf.Variable(np.random.randn(dim, self.rnn_state_size, 2, self.num_TR), dtype=tf.float32, name="W_TR")
        self.b_TR = tf.Variable(np.zeros((dim, 2, self.num_TR)), dtype=tf.float32, name="b_TR")

        if query_specific:
            # Embedding lookup for selected W and b
            W_rel_selected = tf.nn.embedding_lookup(self.W_rel, self.query_rels)
            b_rel_selected = tf.nn.embedding_lookup(self.b_rel, self.query_rels)

            W_TR_selected = tf.nn.embedding_lookup(self.W_TR, self.query_rels)
            b_TR_selected = tf.nn.embedding_lookup(self.b_TR, self.query_rels)
        else:
            W_rel_selected = self.W_rel[0, :, :, :]
            b_rel_selected = self.b_rel

            W_TR_selected = self.W_TR[0, :, :, :]
            b_TR_selected = self.b_TR


        # Different cases: [last_event, first_event]
        # each uses self.rnn_state_size entry in the rnn_output
        self.attn_rel, self.attn_TR = [], []
        for idx_event_pos in range(2):
            if query_specific:
                fn1 = lambda x, w, b, idx, pos: tf.einsum('bi, bij -> bj', x[:, pos:pos + self.rnn_state_size], w[:, :, idx, :]) + b[:, idx, :]
            else:
                fn1 = lambda x, w, b, idx, pos: tf.matmul(x[:, pos:pos + self.rnn_state_size], w[:, idx, :]) + b[:, idx, :]

            fn = lambda x, w, b, idx, pos, num: tf.split(tf.nn.softmax(fn1(x, w, b, idx, pos), axis=1), num, axis=1)

            state_start_pos = self.rnn_state_size*idx_event_pos if not self.flag_rel_TR_split else self.rnn_state_size*(2*idx_event_pos) 
            self.attn_rel.append([fn(rnn_output, W_rel_selected, b_rel_selected, idx_event_pos, state_start_pos, self.num_relation) for rnn_output in self.rnn_outputs])
     
            # if self.flag_rel_TR_split, the RNN output will be split into four parts: [last event rel, last event TR, first event rel, first event TR]
            # otherwise, the RNN output will be split into two parts: [last event, first event]
            state_start_pos += 0 if not self.flag_rel_TR_split else self.rnn_state_size
            self.attn_TR.append([fn(rnn_output, W_TR_selected, b_TR_selected, idx_event_pos, state_start_pos, self.num_TR) for rnn_output in self.rnn_outputs]) 

    
    def _build_attn_refType(self):
        num_cases = 6 if self.flag_int else 1
        # The probability we choose different cases: 
        # if self.flag_int:
        #       [tqs, tqe] X [(last_ts, last_te), (first_ts, first_te), (last_event, first_event)]
        # else:
        #       [last_event, first_event]
        self.attn_refType_embed = [tf.Variable(self._random_uniform_unit(self.num_query, 2), 
                                               dtype=tf.float32, name='attn_refType_embed_{}'.format(i)) 
                                               for i in range(num_cases)] 

        attn_refType = [tf.nn.embedding_lookup(embed, self.query_rels) for embed in self.attn_refType_embed]
        attn_refType = [tf.nn.softmax(x, axis=1) for x in attn_refType]
        self.attn_refType = attn_refType


    def _build_shallow_layers(self):
        self.query_rels_flatten = tf.placeholder(tf.int32, [None])   # (flatten_batch_size) num_relevant_events in a batch
        self.random_walk_ind = tf.placeholder(tf.float32, [None, 2, self.num_rule])  # (flatten_batch_size, 2, num_rule)
        self.refNode_index = tf.placeholder(tf.int32, [None, 2])    # show the refNode index from which query

        # Use shallow layers to enhance the state vector.
        # We distinguish the rule scores for last and first events.
        # 0: last_event, 1: first_event (different from fast ver)
        self.attn_rule_embed = tf.Variable(np.random.randn(self.num_query, 2, self.num_rule), dtype=tf.float32, name="attn_rule_embed")


    def _build_memories(self, num_entity):
        # attention_memories: (will be) a list of num_step tensors,
        # each of size (batch_size, t+1),
        # where t is the current step (zero indexed).
        # Each tensor represents the attention over currently populated memory cells. 
        # we will have a list of attention_memories: [last_event, first_event]
        self.attn_memories_ls = [[] for _ in range(2)]

        # memories: (will be) a tensor of size (batch_size, t+1, num_entity),
        # where t is the current step (zero indexed)
        # Then tensor represents currently populated memory cells.
        # we will have a list of memories: [last_event, first_event]
        self.memories_ls = [tf.expand_dims(tf.one_hot(indices=self.tails, depth=num_entity), 1),
                            tf.expand_dims(tf.one_hot(indices=self.heads, depth=num_entity), 1)]


    def _build_variables(self):
        # Consider the prob dist of the query time.
        # shape: self.probs (batch_size, num_cases, num_entity, num_timestamp)
        # Given a query, a case, a reference node, and a timestamp, what is the probability as the timestamp to be the query time.
        # For training, we know the target timestamp, so the shape becomes (batch_size, num_cases, num_entity)
        # num_cases: [tqs, tqe] X [last_event, first_event] X [ts, te] 
        # Order: 000, 001, 010, 011, 100, 101, 110, 111
        self.probs = tf.placeholder(tf.float32, [None, None, None])
        num_sample = tf.shape(self.probs)[0]   
        num_entity = tf.shape(self.probs)[2]
        
        # Connectivity matrix for differen relations (predicates)
        self.connectivity_rel = {idx_rel: tf.sparse_placeholder(
                                            dtype=tf.float32, 
                                            name="connectivity_rel_%d" % idx_rel)
                                            for idx_rel in range(self.num_relation)}

        # Connectivity matrix for differen TRs: [0: ukn, 1: bf, 2: touch, 3: af]
        self.connectivity_TR = {idx_TR: tf.sparse_placeholder(
                                        dtype=tf.float32,
                                        name="connectivity_TR_%d" % idx_TR)
                                        for idx_TR in range(self.num_TR)}


        self._build_rnn_inputs(num_entity)
        self._build_rnn_outputs(num_sample)
        self._build_attn_rel_and_TR()
        self._build_attn_refType()
        self._build_memories(num_entity)

        if self.flag_state_vec_enhance:
            self._build_shallow_layers()
        
        return num_entity, num_sample


    def _time_prediction(self, state_vec, query_time_dist, target_entities):
        '''
        Given the final state vector, we predict the query time.
        '''
        # The state vector is about the probability we arrive at different reference nodes. 
        # Given each reference node, we can calculate a probability for the query time.
        # shape change: (batch_size, num_nodes, 1) * (batch_size, num_nodes, num_timestamp) -> (batch_size, num_nodes, num_timestamp)
        # For training, since we know the target timestamp, we can simply the calculatation as:
        # shape change: (batch_size, num_nodes) * (batch_size, num_nodes) -> (batch_size, num_nodes)
        pred = state_vec * query_time_dist
        
        # We require the path to be cyclic. And the last TR can only be ukn since we don't know the query time.
        pred = tf.transpose(pred)
        pred = tf.sparse_tensor_dense_matmul(self.connectivity_TR[0], pred)
        pred = tf.transpose(pred)
        pred = tf.reduce_sum(target_entities * pred, axis=1, keep_dims=True)

        # To make the learning more stable, we scale the final prob.
        # All the operations are the same as above except without the multiplication of query_time_dist.
        norm = tf.identity(state_vec)
        norm = tf.transpose(norm)
        norm = tf.sparse_tensor_dense_matmul(self.connectivity_TR[0], norm)
        norm = tf.transpose(norm)
        norm = 1e-20 + tf.reduce_sum(target_entities * norm, axis=1, keep_dims=True)

        return pred/norm


    def _calculate_loss(self, pred):
        return - tf.reduce_sum(tf.log(tf.maximum(pred, self.thr)), 1)


    def _selection_block(self, state_vec, attn, trans_mat, choice_ls):
        '''
        Select elements (either relations or TRs) during transition via a weighted sum.
        '''
        state_vec = tf.transpose(state_vec)

        trans_results = []
        for idx in choice_ls:  # now connectivity_rel is a diagonal matrix which shows the predicate of nodes
            product = tf.sparse_tensor_dense_matmul(trans_mat[idx], state_vec)
            product = tf.transpose(product)
            trans_results.append(attn[idx] * product)
        
        added_trans_results = tf.add_n(trans_results)
        
        if self.norm:
            added_trans_results /= tf.maximum(self.thr, tf.reduce_sum(added_trans_results, axis=1, keep_dims=True))
                
        return added_trans_results


    def _obtain_state_vec_enhance(self, num_sample, num_entity):
        '''
        Use shallow layers to enhance the state vector.
        notation: 0: last_event, 1: first_event (different from fast ver)
        '''
        attn_rule = tf.nn.embedding_lookup(self.attn_rule_embed, self.query_rels_flatten)
        self.attn_rule = tf.nn.softmax(attn_rule, axis=2) # shape: (flatten_batch_size, 2, num_rule)

        # We calculate the probability we arrive at different reference nodes from shallow rule scores.
        #    self.random_walk_ind: the index of rule for each ref event
        #           shape:(num_ref_events, 2, num_rules) [2 channels: we distinguish the nodes as last or first events]
        #    res shape: (flatten_batch_size, )
        refNode_probs = [tf.reduce_sum(self.random_walk_ind[:, i, :] * self.attn_rule[:, i, :], axis=1, keep_dims=False) for i in range(2)]
        
        # We obtain the state vector enhancement by assigning the probability to certain index of the state vector.
        # res shape: (batch_size, num_entity)
        state_vec_enhance = [tf.scatter_nd(self.refNode_index, refNode_probs[i], [num_sample, num_entity]) for i in range(2)]
        
        # normalize the state vector enhancement
        state_vec_enhance = [x/(1e-20 + tf.reduce_sum(x, axis=1, keep_dims=True)) for x in state_vec_enhance]
        
        return state_vec_enhance


    def _param_optimization(self, loss):
        optimizer = tf.train.AdamOptimizer()
        gvs = optimizer.compute_gradients(tf.reduce_mean(loss))
 
        # Check for NaNs in gradients
        grads_are_nan = tf.reduce_any([tf.reduce_any(tf.is_nan(grad)) for grad, _ in gvs])

        # Define a conditional update operation
        update_op = tf.cond(
            grads_are_nan,
            lambda: tf.no_op(),  # Do nothing if NaNs are found
            lambda: optimizer.apply_gradients(gvs)  # Apply gradients if no NaNs
        )
        return update_op


    def _build_comp_graph(self):
        '''
        Define forward process. Use both RNN and shallow layers.
        Please look at the Neural-LP paper (Yang et al., 2017) for more details and explanations of the RNN structure.
        '''
        num_entity, num_sample = self._build_variables()
        
        if self.flag_state_vec_enhance:
            # Use shallow layers to enhance the state vector.
            # For first and last events, we use different state vectors.
            state_vec_enhance = self._obtain_state_vec_enhance(num_sample, num_entity)
        
        self.pred = []
        self.state_vec_recorded = [] # Record the final state vector for both tqs and tqe since they share it.
        for idx_event_pos in range(2):
            # Use last or first event on the path to predict the query time.
            # For last event it is the end node of our random walk, and for first event it is the end node of our inverse random walk.
            state_start_pos = self.rnn_state_size*idx_event_pos
            for t in range(self.num_step):
                # Each time we start from a state vector which is a weight sum of all the previous state vectors.
                # self.rnn_outputs[t] is current state vector (shape: (batch_size, rnn_state_size*2))
                # the first half is used for last event, and the second half is used for first event
                self.attn_memories_ls[idx_event_pos].append(tf.nn.softmax(tf.squeeze(tf.matmul(
                                                            tf.expand_dims(self.rnn_outputs[t][:, state_start_pos:state_start_pos + self.rnn_state_size], 1), 
                                                            tf.stack([rnn_output[:, state_start_pos:state_start_pos + self.rnn_state_size] 
                                                                        for rnn_output in self.rnn_outputs[0:t+1]], axis=2)), 
                                                                        squeeze_dims=[1])))
                # (batch_size, num_nodes)
                state_vec = tf.squeeze(
                                tf.matmul(
                                    tf.expand_dims(self.attn_memories_ls[idx_event_pos][t], 1), 
                                    self.memories_ls[idx_event_pos]), 
                                squeeze_dims=[1])

                if  t < self.num_step - 1:   
                    # We first consider the selection of different temporal relations (TR).
                    # for t = 0, TR is fixed as ukn since we don't know the query time.
                    # self.connectivity_TR[0] if for the TR of ukn.
                    TR_choice_ls = [0] if t == 0 else range(self.num_TR)
                    state_vec = self._selection_block(state_vec, self.attn_TR[idx_event_pos][t], self.connectivity_TR, TR_choice_ls)

                    # We then consider the selection of different relations (predicates).    
                    state_vec = self._selection_block(state_vec, self.attn_rel[idx_event_pos][t], self.connectivity_rel, range(self.num_relation))
                    
                    # As the end of each step, we apply dropout (if choosen) and add the current memory to the memory list.
                    if self.dropout > 0.:
                        state_vec = tf.nn.dropout(state_vec, keep_prob=1.-self.dropout)
                    
                    # Record the memory for the next step.
                    self.memories_ls[idx_event_pos] = tf.concat( 
                                                                [self.memories_ls[idx_event_pos], 
                                                                tf.expand_dims(state_vec, 1)],
                                                                axis=1)
                else:
                    # We use a mixed final state vector here.
                    if self.flag_state_vec_enhance:
                        state_vec = self.gamma * state_vec + (1-self.gamma) * state_vec_enhance[idx_event_pos]

                    # record the final state vector for tqe (if needed).
                    self.state_vec_recorded.append(tf.identity(state_vec))
                    
                    # We use a weighted sum of dist from difference cases.
                    if self.flag_int:
                        query_time_dist = self.attn_refType[idx_event_pos][:, 0:1] * self.probs[:, 2*idx_event_pos, :] + \
                                          self.attn_refType[idx_event_pos][:, 1:2] * self.probs[:, 2*idx_event_pos+1, :]
                    else:
                        query_time_dist = self.probs[:, idx_event_pos, :]

                    # For the last step, we use the state vector to predict the query time.
                    # For last event, we start from tail entity and should come back to the tail entity.
                    # For first event, we start from head entity and should come back to the head entity.
                    self.pred.append(self._time_prediction(state_vec, query_time_dist, self.targets[idx_event_pos]))
           
        # We merge the results from last and first events to get the final prediction.
        if self.flag_int:
            self.final_pred = [self.attn_refType[2][:, 0:1] * self.pred[0] + self.attn_refType[2][:, 1:2] * self.pred[1]]
        else:
            self.final_pred = [self.attn_refType[0][:, 0:1] * self.pred[0] + self.attn_refType[0][:, 1:2] * self.pred[1]]


        # scaling the final prob to make the learning more stable.
        if self.prob_scaling_factor is not None:
            self.final_pred[0] = self._scale_final_prob(self.final_pred[0])

        # calculate the loss
        self.final_loss = self._calculate_loss(self.final_pred[0])


        if self.flag_int:
            for idx_event_pos in range(2):
                # Use the same final state vector for both tqs and tqe.
                # All the operations are the same as above except the query_time_dist has changed.
                # We use a weighted sum of dist from difference cases.
                state_vec = self.state_vec_recorded[idx_event_pos]
                
                query_time_dist = self.attn_refType[3+idx_event_pos][:, 0:1] * self.probs[:, 2*idx_event_pos+4, :] + \
                                  self.attn_refType[3+idx_event_pos][:, 1:2] * self.probs[:, 2*idx_event_pos+5, :]
                                
                self.pred.append(self._time_prediction(state_vec, query_time_dist, self.targets[idx_event_pos]))

            # attn_refType changes for tqs and tqe.     
            self.final_pred.append(self.attn_refType[5][:, 0:1] * self.pred[2] + self.attn_refType[5][:, 1:2] * self.pred[3])
           
            if self.prob_scaling_factor is not None:
                # scaling the loss to make the learning more stable.
                self.final_pred[1] = self._scale_final_prob(self.final_pred[1])
            
            self.final_loss += self._calculate_loss(self.final_pred[1])
            self.final_loss *= 0.5

        # param optimization
        self.optimizer_step = self._param_optimization(self.final_loss)

        
    def _build_comp_graph_fast_ver(self):
        '''
        Use shallow layers only to accelerate the training process.
        '''
        self.query_rel_flatten = tf.placeholder(tf.int32, [None])   # (flatten_batch_size) num_relevant_events in a batch
        self.refNode_source = tf.placeholder(tf.float32, [None, None])  # show where the refNode comes from (batch_size, flatten_batch_size)
        
        if self.flag_ruleLen_split:
            # Todo: ruleLen_split_ver
            pass
        else:
            num_cases_probs = 8 if self.flag_int else 2   # [tqs, tqe] X [last_event, first_event] X [ts, te] or [last_event, first_event]
            self.random_walk_prob = tf.placeholder(tf.float32, [None, num_cases_probs, self.num_rule])  # (flatten_batch_size, num_cases, num_rule)
            self.random_walk_ind = tf.placeholder(tf.float32, [None, 2, self.num_rule])  # (flatten_batch_size, 2, num_rule)


        # attention vec for different rules given the query relation
        self.attn_rule_embed = tf.Variable(np.random.randn(self.num_query, self.num_rule), dtype=tf.float32, name="attn_rule_embed")
        attn_rule = tf.nn.embedding_lookup(self.attn_rule_embed, self.query_rel_flatten)
        attn_rule = tf.nn.softmax(attn_rule, axis=1)
        self.attn_rule = attn_rule  # shape: (flatten_batch_size, num_rule)
        
        # attention vec for different positions: first or last event, using ts or te
        # [tqs, tqe] X [(first_event_ts, first_event_te), (last_event_ts, last_event_te), (first_event, last_event)] or [first_event, last_event]
        num_cases_refType = 6 if self.flag_int else 1
        self.attn_refType_embed = [tf.Variable(self._random_uniform_unit(self.num_query, 2), dtype=tf.float32, name='attn_refType_embed_{}'.format(i)) 
                                   for i in range(num_cases_refType)]
        attn_refType = [tf.nn.embedding_lookup(embed, self.query_rel_flatten) for embed in self.attn_refType_embed]
        attn_refType = [tf.nn.softmax(x, axis=1) for x in attn_refType] # each element (flatten_batch_size, 2)
        self.attn_refType = attn_refType
    
        if self.flag_ruleLen_split:
            pass    
        else:
            # prob we arrive at different reference events
            refNode_attn = [tf.reduce_sum(self.random_walk_prob[:, i, :] * attn_rule, axis=1, keep_dims=True) for i in range(num_cases_probs)]  # shape: (flatten_batch_size, 1)
            refNode_attn_norm = [tf.reduce_sum(self.random_walk_ind[:, i, :] * attn_rule, axis=1, keep_dims=True) for i in range(2)]  # shape: (flatten_batch_size, 1)
            
            self.final_pred = []
            for i in range(2):
                if (not self.flag_int) and (i>0):
                    break

                if self.flag_int:
                    # attn_refType[3*i+2]: choose first or last event; 
                    # attn_refType[3*i][:, 0:1]: choose ts or te for the first event;
                    # attn_refType[3*i][:, 1:2]: choose ts or te for the last event.
                    probs = attn_refType[3*i+2][:, 0:1] * (attn_refType[3*i][:, 0:1]   * refNode_attn[4*i]   + attn_refType[3*i][:, 1:2]   * refNode_attn[4*i+1]) + \
                            attn_refType[3*i+2][:, 1:2] * (attn_refType[3*i+1][:, 0:1] * refNode_attn[4*i+2] + attn_refType[3*i+1][:, 1:2] * refNode_attn[4*i+3])  

                    norm = attn_refType[3*i+2][:, 0:1] * (attn_refType[3*i][:, 0:1]   * refNode_attn_norm[0] + attn_refType[3*i][:, 1:2]   * refNode_attn_norm[0]) + \
                           attn_refType[3*i+2][:, 1:2] * (attn_refType[3*i+1][:, 0:1] * refNode_attn_norm[1] + attn_refType[3*i+1][:, 1:2] * refNode_attn_norm[1])  
                else:
                    probs = attn_refType[0][:, 0:1] * refNode_attn[0] + attn_refType[0][:, 1:2] * refNode_attn[1]
                    norm =  attn_refType[0][:, 0:1] * refNode_attn_norm[0] + attn_refType[0][:, 1:2] * refNode_attn_norm[1]

                # refNode_source: shape: (batch_size, flatten_batch_size), prob: shape: (flatten_batch_size, num_timestamp)
                self.final_pred.append(tf.matmul(self.refNode_source, probs) / (tf.matmul(self.refNode_source, norm)) + 1e-20)


        # scaling the prob to make the learning more stable.
        if self.prob_scaling_factor is not None:
            for i in range(int(self.flag_int)+1):
                self.final_pred[i] = self._scale_final_prob(self.final_pred[i])


        # calculate the loss
        self.final_loss = self._calculate_loss(self.final_pred[0])
        self.final_loss += self._calculate_loss(self.final_pred[1]) if self.flag_int else 0
        self.final_loss *= 0.5 if self.flag_int else 1

        # param optimization
        self.optimizer_step = self._param_optimization(self.final_loss)


    def _run_comp_graph(self, sess, qq, hh, tt, connectivity_rel, connectivity_TR, probs, valid_sample_idx, inputs_for_enhancement, to_fetch):
        qq = [qq[idx] for idx in valid_sample_idx]
        hh = [hh[idx] for idx in valid_sample_idx]
        tt = [tt[idx] for idx in valid_sample_idx]

        # Create feed dict.
        feed = {}
        feed[self.queries] = [[q] * (self.num_step-1) + [self.num_query] for q in qq]
        feed[self.heads] = hh 
        feed[self.tails] = tt 
        feed[self.query_rels] = qq
        feed[self.probs] = [probs[idx] for idx in valid_sample_idx]

        for idx_rel in range(self.num_relation):
            feed[self.connectivity_rel[idx_rel]] = tf.SparseTensorValue(*connectivity_rel[idx_rel])
        for idx_TR in range(self.num_TR):
            feed[self.connectivity_TR[idx_TR]] = tf.SparseTensorValue(*connectivity_TR[idx_TR])


        if self.flag_state_vec_enhance:
            # Similar to fast version, we use shallow layers to enhance the state vector.
            query_rels_flatten, res_random_walk, refNode_index = inputs_for_enhancement
            
            # Create random_walk_ind from res_random_walk.
            # We only need to distinguish the cases for first or last event.
            # There is an inverse mapping since in the random walk use 0 for first event and 1 for last event.
            # In RNN model, we first calculate the last event and then the first event.
            random_walk_ind = np.zeros((len(query_rels_flatten), 2, self.num_rule))  
            for i, j in zip([0, -1], [1, 0]):
                if len(res_random_walk[i]) > 0:
                    x, y = res_random_walk[i][:, 0].astype(int), res_random_walk[i][:, 1].astype(int)
                    random_walk_ind[x, j, y] = 1
            
            feed[self.query_rels_flatten] = query_rels_flatten
            feed[self.refNode_index] = refNode_index
            feed[self.random_walk_ind] = random_walk_ind


        fetches = to_fetch
        graph_output = sess.run(fetches, feed)
        return graph_output


    def _run_comp_graph_fast_ver(self, sess, query_rel_flatten, refNode_source, probs, to_fetch):
        feed = {}
        feed[self.query_rel_flatten] = query_rel_flatten
        feed[self.refNode_source] = refNode_source

        num_cases = 8 if self.flag_int else 2
        random_walk_prob = np.zeros((len(query_rel_flatten), num_cases, self.num_rule)) 
        random_walk_ind = np.zeros((len(query_rel_flatten), 2, self.num_rule))  # We only need to distinguish the cases for first or last event.
        for i in range(num_cases):
            if len(probs[i]) > 0:
                x, y, p = probs[i][:, 0].astype(int), probs[i][:, 1].astype(int), probs[i][:, 2]
                random_walk_prob[x, i, y] = p
                if i in [0, num_cases-1]:
                    random_walk_ind[x, min(i, 1), y] = 1
        feed[self.random_walk_prob] = random_walk_prob
        feed[self.random_walk_ind] = random_walk_ind

        fetches = to_fetch
        graph_output = sess.run(fetches, feed)
        return graph_output


    def _process_probs_for_prediction(self, probs, num_entity):
        probs = np.array(probs)
        probs = np.transpose(probs, (0, 2, 1))
        probs = probs.reshape(-1, num_entity)
        return probs.tolist()


    def _create_flatten_probs_for_prediction(self, probs):
        return [prob[0] for prob in probs]


    def update(self, sess, inputs):
        to_fetch = [self.final_loss, self.optimizer_step]
        if self.flag_acceleration:
            qq, refNode_source, res_random_walk = inputs
            fetched = self._run_comp_graph_fast_ver(sess, qq, refNode_source, res_random_walk, to_fetch)
        else:
            qq, hh, tt, connectivity_rel, connectivity_TR, probs, valid_sample_idx, inputs_for_enhancement = inputs
            fetched = self._run_comp_graph(sess, qq, hh, tt, connectivity_rel, connectivity_TR, probs, valid_sample_idx, inputs_for_enhancement, to_fetch)
        return fetched[0]


    def get_state_vec(self, sess, inputs):
        '''
        Get the state vector for each sample. We split the inference process into two stages (1.obtain state vec; 2.time prediction) for acceleration.
        '''
        # For fasr version, we don't need to get the state vector.
        assert not self.flag_acceleration

        # Run the computation graph and get the results.
        qq, hh, tt, connectivity_rel, connectivity_TR, probs, valid_sample_idx, inputs_for_enhancement, valid_ref_idx, batch_idx_ls = inputs
        to_fetch = [self.state_vec_recorded, self.attn_refType]
        fetched = self._run_comp_graph(sess, qq, hh, tt, connectivity_rel, connectivity_TR, probs, valid_sample_idx, inputs_for_enhancement, to_fetch)
        state_vec_recorded, attn_refType = fetched[0], fetched[1]

        # Prepare the output.
        selected_state_vec = {}
        for cur_ref_idx in valid_ref_idx:
            # cur_ref_idx format: (idx_sample, idx_event_pos, idx_event)
            # state_vec_recorded: [vec for last event, vec for first event]
            cur_state = state_vec_recorded[1-cur_ref_idx[1]]

            # relative pos in the batch
            fn = lambda ls1, ele1, ls2: ls2.index(ls1.index(ele1))
            pos = [fn(batch_idx_ls, cur_ref_idx[0], valid_sample_idx), cur_ref_idx[2]]

            selected_state_vec[str(cur_ref_idx)] = cur_state[tuple(pos)].tolist()

        selected_attn_refType = {}
        for (i, idx) in enumerate(valid_sample_idx):
            cur_global_idx = batch_idx_ls[idx]
            selected_attn_refType[cur_global_idx] = [attn[i, :].tolist() for attn in attn_refType]

        output = {}
        output['final_state_vec'] = selected_state_vec
        output['attn_refType'] = selected_attn_refType

        return output


    def predict(self, sess, inputs):
        '''
        The original predict function. We do not use it due to the inefficiency.
        '''
        to_fetch = [self.final_pred]
        if self.flag_acceleration:
            qq, refNode_source, res_random_walk = inputs
            fetched = self._run_comp_graph_fast_ver(sess, qq, refNode_source, res_random_walk, to_fetch)
            return fetched[0]
        else:
            qq, hh, tt, connectivity_rel, connectivity_TR, probs, valid_sample_idx, inputs_for_enhancement = inputs
            fetched = self._run_comp_graph(sess, qq, hh, tt, connectivity_rel, connectivity_TR, probs, valid_sample_idx, inputs_for_enhancement, to_fetch)   
            return fetched[0]


    def get_rule_scores_fast_ver(self, sess):
        to_fetch = [self.attn_rule, self.attn_refType]

        qq = list(range(self.num_query))

        # They do not affect the scores of rules.
        refNode_source = [[0 for _ in range(self.num_query)]]
        num_cases = 8 if self.flag_int else 2
        res_random_walk = [np.array([[0 for _ in range(3)] for _ in range(self.num_query)]) for _ in range(num_cases)]
        
        fetched = self._run_comp_graph_fast_ver(sess, qq, refNode_source, res_random_walk, to_fetch)

        return fetched[0], fetched[1]
    

    def get_rule_scores(self, sess):
        to_fetch = [self.attn_rel, self.attn_TR, self.attn_refType, self.attn_memories_ls]
        if self.flag_state_vec_enhance:
            to_fetch.append(self.attn_rule)

        batch_size = self.num_query
        num_entity = 100
        num_cases = 8 if self.flag_int else 2
        
        qq = list(range(self.num_query))
        valid_sample_idx = list(range(batch_size))

        # They do not affect the scores of rules. 
        hh = [0 for _ in range(batch_size)]
        tt = [0 for _ in range(batch_size)]
        connectivity_rel = {idx_rel: ([[0,0]], [0.], [num_entity, num_entity]) for idx_rel in range(self.num_rel)}
        connectivity_TR = {idx_TR: ([[0,0]], [0.], [num_entity, num_entity]) for idx_TR in range(self.num_TR)}        
        probs = [[[0 for _ in range(num_entity)] for _ in range(num_cases)] for _ in range(batch_size)]

        inputs_for_enhancement = []
        if self.flag_state_vec_enhance:
            query_rels_flatten = list(range(self.num_query))

            # They do not affect the scores of rules. 
            res_random_walk = [np.array([[0 for _ in range(3)] for _ in range(batch_size)]) for _ in range(num_cases)]
            refNode_index = [[0, 0] for _ in range(batch_size)]
            
            # collect the inputs for enhancement
            inputs_for_enhancement = [query_rels_flatten, res_random_walk, refNode_index]

        fetched = self._run_comp_graph(sess, qq, hh, tt, connectivity_rel, connectivity_TR, probs, valid_sample_idx, inputs_for_enhancement, to_fetch)

        return fetched[0], fetched[1], fetched[2], fetched[3], fetched[4] if self.flag_state_vec_enhance else None