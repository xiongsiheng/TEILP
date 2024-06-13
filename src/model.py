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

        # self.num_entity = data['num_entity']  # dynamic
        self.num_relation = data['num_rel']
        self.num_TR = data['num_TR']
        self.num_rule = option.num_rule
        self.num_timestamp = len(data['timestamp_range'])
        self.num_query = data['num_rel']
        self.query_embed_size = option.query_embed_size
        
        # weight for the state vector enhancement (w/ shallow layers)
        # final state = gamma * final_state + (1-gamma) * enhance_state
        self.gamma = 0.7


        self.flag_int = option.flag_interval
        self.flag_rel_TR_split = option.different_states_for_rel_and_TR
        self.flag_ruleLen_split = option.flag_ruleLen_split_ver
        self.flag_state_vec_enhance = option.flag_state_vec_enhancement
        self.flag_acceleration = option.flag_acceleration
        self.flag_loss_scaling_factor = 0.1  # Fast ver: YAGO, wiki  0.04

        np.random.seed(self.seed)

        if option.flag_ruleLen_split_ver:
            print('Todo: ruleLen_split_ver')
            pass
        else:
            if self.flag_acceleration:
                self._build_comp_graph_fast_ver()
            else:
                self._build_comp_graph()


    def _init_ruleLen_embedding(self, pattern_ls):
        self.ruleLen_embedding = [np.zeros((self.num_query, self.num_rule))] * (self.num_step-1)

        for rel in pattern_ls:
            ruleLen = np.array([len(rule.split(' '))//2 for rule in pattern_ls[rel]])
            ruleLen = np.hstack((ruleLen, np.zeros((self.num_rule - len(pattern_ls[rel]),))))
            for l in range(1, self.num_step):
                ruleLen_cp = ruleLen.copy()
                ruleLen_cp[~(ruleLen == l)] = 0
                ruleLen_cp[ruleLen == l] = 1
                self.ruleLen_embedding[l-1][int(rel)] = ruleLen_cp


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


    def _attn_normalization(self, attn, sample_indices):
        '''
        Normalize the attention weights (Important!)

        Parameters:
            attn: attention weights, shape: (num_rules_total, 1)
            sample_indices: sample indices, shape: (num_samples, num_rules_total)
                            [[1, 0, 0, 0, 0],   # rule 1 from sample 1
                             [0, 1, 1, 0, 0],   # rule 2, 3 from sample 2
                             [0, 0, 0, 1, 1]]   # rule 4, 5 from sample 3

        Returns:
            normalized_attn: normalized attention weights, shape: (num_rules_total, 1)
        '''
        attn = tf.reshape(attn, (1, -1))
        # Compute sum of attn for each sample
        sample_sums = tf.reduce_sum(sample_indices*attn, axis=1) + 1e-20 # shape: (num_samples, )

        # Normalize attn
        normalized_attn = (sample_indices*attn) / tf.expand_dims(sample_sums, axis=1)
        normalized_attn = tf.reduce_sum(normalized_attn, axis=0)

        normalized_attn = tf.reshape(normalized_attn, (-1, 1))
        return normalized_attn


    def _build_rnn_inputs(self, num_entity):
        '''
        Build RNN inputs for rule score learning.
        '''
        self.tails = tf.placeholder(tf.int32, [None])
        self.heads = tf.placeholder(tf.int32, [None])
        self.targets = [tf.one_hot(indices=self.tails, depth=num_entity), tf.one_hot(indices=self.heads, depth=num_entity)]

        self.query_rels = tf.placeholder(tf.int32, [None])
        self.queries = tf.placeholder(tf.int32, [None, self.num_step])
        self.query_embedding_params = tf.Variable(self._random_uniform_unit(
                                                      self.num_query + 1, # <END> token 
                                                      self.query_embed_size), 
                                                      dtype=tf.float32, name="query_embedding_params")

        rnn_inputs = tf.nn.embedding_lookup(self.query_embedding_params, 
                                            self.queries)

        return rnn_inputs


    def _build_variables(self):
        # Different channels: [tqs, tqe] X [last_event, first_event] X [ts, te] 
        # Order: 000, 001, 010, 011, 100, 101, 110, 111
        num_cases = 8 if self.flag_int else 2
        self.probs = tf.placeholder(tf.float32, [None, num_cases, None])   
        num_entity = tf.shape(self.probs)[-1]
        num_sample = tf.shape(self.probs)[0]

        rnn_inputs = self._build_rnn_inputs(num_entity)
        self.rnn_inputs = [tf.reshape(q, [-1, self.query_embed_size]) for q in tf.split(rnn_inputs, self.num_step, axis=1)]

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

        # Instead of using multiple RNNs for different relations and TRs, we use a single RNN with a longer hidden state vector.
        cellLen = self.rnn_state_size*4 if self.flag_rel_TR_split else self.rnn_state_size*2
        cell = tf.nn.rnn_cell.LSTMCell(cellLen, state_is_tuple=True)
        self.cell = tf.nn.rnn_cell.MultiRNNCell([cell] * self.num_layer, 
                                                    state_is_tuple=True)
        init_state = self.cell.zero_state(tf.shape(self.heads)[0], tf.float32)

        self.rnn_outputs, self.final_state = tf.contrib.rnn.static_rnn(
                                                self.cell, 
                                                self.rnn_inputs,
                                                initial_state=init_state)

        # Different channels: [last_event, first_event]
        self.W_rel = tf.Variable(np.random.randn(self.rnn_state_size, 2, self.num_relation), dtype=tf.float32, name="W_rel")
        self.b_rel = tf.Variable(np.zeros((1, 2, self.num_relation)), dtype=tf.float32, name="b_rel")
        
        self.W_TR = tf.Variable(np.random.randn(self.rnn_state_size, 2, self.num_TR), dtype=tf.float32, name="W_TR")
        self.b_TR = tf.Variable(np.zeros((1, 2, self.num_TR)), dtype=tf.float32, name="b_TR")


        # Different cases: [last_event, first_event]
        # each uses self.rnn_state_size entry in the rnn_output
        self.attn_rel, self.attn_TR = [], []
        for idx_event_pos in range(2):
            state_start_pos = self.rnn_state_size*idx_event_pos if not self.flag_rel_TR_split else self.rnn_state_size*(2*idx_event_pos)
            self.attn_rel.append([tf.split(tf.nn.softmax(
                                  tf.matmul(rnn_output[:, state_start_pos:state_start_pos + self.rnn_state_size], 
                                            self.W_rel[:, idx_event_pos, :]) + self.b_rel[:, idx_event_pos, :], axis=1), self.num_relation, axis=1) 
                                            for rnn_output in self.rnn_outputs])

            # if self.flag_rel_TR_split, the RNN output will be split into four parts: [last event rel, last event TR, first event rel, first event TR]
            # otherwise, the RNN output will be split into two parts: [last event, first event]
            state_start_pos += 0 if not self.flag_rel_TR_split else self.rnn_state_size
            self.attn_TR.append([tf.split(tf.nn.softmax(
                                  tf.matmul(rnn_output[:, state_start_pos: state_start_pos+ self.rnn_state_size], 
                                            self.W_TR[:, idx_event_pos, :]) + self.b_TR[:, idx_event_pos, :], axis=1), self.num_TR, axis=1) 
                                            for rnn_output in self.rnn_outputs])      
        
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
        
        # The probability we choose different cases: [tqs, tqe] X [(last_ts, last_te), (first_ts, first_te), (last_event, first_event)]
        self.attn_refType_embed = [tf.Variable(self._random_uniform_unit(self.num_query, 2), dtype=tf.float32, name= 'attn_refType_embed_{}'.format(i)) for i in range(3*(int(self.flag_int)+1))] 

        if self.flag_state_vec_enhance:
            self.query_rels = tf.placeholder(tf.int32, [None])   # (dummy_batch_size) num_relevant_events in a batch
            self.random_walk_ind = tf.placeholder(tf.float32, [None, 2, self.num_rule])  # (dummy_batch_size, 2, num_rule)
            self.refNode_index = tf.placeholder(tf.int32, [None, 2])    # show the refNode index from which query

            # Use shallow layers to enhance the state vector.
            self.attn_rule_embed = tf.Variable(np.random.randn(self.num_query, self.num_rule), dtype=tf.float32, name="attn_rule_embed")
        
        return num_entity, num_sample


    def _build_comp_graph(self):
        '''
        Define forward process. Use both RNN and shallow layers.
        Please look at the Neural-LP paper (Yang, 2017) for more details and explanations.

        '''
        num_entity, num_sample = self._build_variables()
        
        # [tqs, tqe] * [(last_ts, last_te), (first_ts, first_te), (last_event, first_event)]
        attn_refType = [tf.nn.embedding_lookup(embed, self.query_rels) for embed in self.attn_refType_embed]
        attn_refType = [tf.nn.softmax(x, axis=1) for x in attn_refType]

        if self.flag_state_vec_enhance:
            # Use shallow layers to enhance the state vector.
            # For first and last events, we use different state vectors.
            attn_rule = tf.nn.embedding_lookup(self.attn_rule_embed, self.query_rels)
            attn_rule = tf.nn.softmax(attn_rule, axis=1)
            refNode_probs = [tf.reduce_sum(self.random_walk_ind[:, i, :] * attn_rule, axis=1, keep_dims=False) for i in range(2)]  # shape: (dummy_batch_size, )
            state_vec_enhance = [tf.scatter_nd(self.refNode_index, refNode_probs[i], [num_sample, num_entity]) for i in range(2)] # shape: (batch_size, 1)
        

        self.state_vec_recorded = [] # Record the final state vector for both tqs and tqe since they share it.
        self.pred = []
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
                memory_read = tf.squeeze(
                                tf.matmul(
                                    tf.expand_dims(self.attn_memories_ls[idx_event_pos][t], 1), 
                                    self.memories_ls[idx_event_pos]), squeeze_dims=[1])


                if  t < self.num_step - 1:   
                    # We first consider the selection of different temporal relations (TR).
                    memory_read = tf.transpose(memory_read)
                    if t == 0:
                        # for t = 0, TR is fixed as ukn since we don't know the query time.
                        # self.connectivity_TR[0] if for the TR of ukn.
                        memory_read = tf.sparse_tensor_dense_matmul(self.connectivity_TR[0], memory_read)
                        memory_read = tf.transpose(memory_read)
                    else:
                        database_results = []
                        for idx_TR in range(self.num_TR):   
                            # We have different choices: TR: [0: ukn, 1: bf, 2: t, 3: af]
                            product = tf.sparse_tensor_dense_matmul(self.connectivity_TR[idx_TR], memory_read)
                            database_results.append(tf.transpose(product) * self.attn_TR[idx_event_pos][t][idx_TR])
                        
                        # shape: (batch_size, num_nodes)
                        added_database_results = tf.add_n(database_results)
                        if self.norm:
                            added_database_results /= tf.maximum(self.thr, tf.reduce_sum(added_database_results, axis=1, keep_dims=True))
                        memory_read = tf.identity(added_database_results)

                    # We then consider the selection of different relations (predicates).    
                    memory_read = tf.transpose(memory_read)
                    database_results = []
                    for idx_rel in range(self.num_relation):  # now connectivity_rel is a diagonal matrix which shows the predicate of nodes
                        product = tf.sparse_tensor_dense_matmul(self.connectivity_rel[idx_rel], memory_read)
                        database_results.append(tf.transpose(product) * self.attn_rel[idx_event_pos][t][idx_rel])
                    
                    added_database_results = tf.add_n(database_results)
                    if self.norm:
                        added_database_results /= tf.maximum(self.thr, tf.reduce_sum(added_database_results, axis=1, keep_dims=True))
                    
                    # As the end of each step, we apply dropout (if choosen) and add the current memory to the memory list.
                    if self.dropout > 0.:
                        added_database_results = tf.nn.dropout(added_database_results, keep_prob=1.-self.dropout)
                    
                    self.memories_ls[idx_event_pos] = tf.concat( 
                                                            [self.memories_ls[idx_event_pos], 
                                                            tf.expand_dims(added_database_results, 1)],
                                                            axis=1)
                else:
                    # For the last step, we use the state vector (memory_read) to predict the query time.
                    
                    # We use a mixed state vector here.
                    if self.flag_state_vec_enhance:
                        memory_read = self.gamma * memory_read + (1-self.gamma) * state_vec_enhance[idx_event_pos]
            
                    self.state_vec_recorded.append(tf.identity(memory_read))


                    # The state vector is about the probability we arrive at different reference nodes. 
                    # Given each reference node, we can calculate a probability for the query time.
                    # shape change: (batch_size, num_nodes, 1) * (batch_size, num_nodes, num_timestamp) -> (batch_size, num_nodes, num_timestamp)
                    # For training, since we know the target timestamp, we can simply the calculatation as:
                    # shape change: (batch_size, num_nodes) * (batch_size, num_nodes) -> (batch_size, num_nodes)
                    norm = tf.identity(memory_read)
                    memory_read *= attn_refType[idx_event_pos][:, 0:1] * self.probs[:, 2*idx_event_pos, :] + \
                                   attn_refType[idx_event_pos][:, 1:2] * self.probs[:, 2*idx_event_pos+1, :]


                    # We require the path to be cyclic. And the last TR is ukn since we don't know the query time.
                    memory_read = tf.transpose(memory_read)
                    memory_read = tf.sparse_tensor_dense_matmul(self.connectivity_TR[0], memory_read)
                    memory_read = tf.transpose(memory_read)
                    
                    norm = tf.transpose(norm)
                    norm = tf.sparse_tensor_dense_matmul(self.connectivity_TR[0], norm)
                    norm = tf.transpose(norm)

                    # For last event, we start from tail entity and should come back to the tail entity.
                    # For first event, we start from head entity and should come back to the head entity.
                    norm = 1e-20 + tf.reduce_sum(self.targets[idx_event_pos] * norm, axis=1, keep_dims=True)
                    self.pred.append(tf.reduce_sum(self.targets[idx_event_pos] * memory_read, axis=1, keep_dims=True)/norm)
           

        self.attn_refType = attn_refType
        
        # We merge the results from last and first events to get the final prediction.
        self.final_pred = [attn_refType[2][:, 0:1] * self.pred[0] + attn_refType[2][:, 1:2] * self.pred[1]]

        # scaling the loss to make the learning more stable.
        if self.flag_loss_scaling_factor is not None:
            self.final_pred[0] /= self.flag_loss_scaling_factor

        self.final_loss = - tf.reduce_sum(tf.log(tf.maximum(self.final_pred[0], self.thr)), 1)


        if self.flag_int:
            for idx_event_pos in range(2):
                # Use the same final state vector for both tqs and tqe.
                memory_read = tf.identity(self.state_vec_recorded[idx_event_pos]) 
                
                # The state vector is about the probability we arrive at different reference nodes. 
                # Given each reference node, we can calculate a probability for the query time.
                norm = tf.identity(memory_read)
                memory_read *= attn_refType[3+idx_event_pos][:, 0:1] * self.probs[:, 2*idx_event_pos+4, :] + \
                               attn_refType[3+idx_event_pos][:, 1:2] * self.probs[:, 2*idx_event_pos+5, :]
                
                # We require the path to be cyclic. And the last TR is ukn since we don't know the query time.
                memory_read = tf.transpose(memory_read)
                memory_read = tf.sparse_tensor_dense_matmul(self.connectivity_TR[0], memory_read)
                memory_read = tf.transpose(memory_read)

                norm = tf.transpose(norm)
                norm = tf.sparse_tensor_dense_matmul(self.connectivity_TR[0], norm)
                norm = tf.transpose(norm)

                # For last event, we start from tail entity and should come back to the tail entity.
                # For first event, we start from head entity and should come back to the head entity.
                norm = 1e-20 + tf.reduce_sum(self.targets[idx_event_pos] * norm, axis=1, keep_dims=True)
                self.pred.append(tf.reduce_sum(self.targets[idx_event_pos] * memory_read, axis=1, keep_dims=True)/norm)
                
            
            # We merge the results from last and first events to get the final prediction.
            self.final_pred.append(attn_refType[5][:, 0:1] * self.pred[2] + attn_refType[5][:, 1:2] * self.pred[3])
           
            # scaling the loss to make the learning more stable.
            if self.flag_loss_scaling_factor is not None:
                self.final_pred[1] /= self.flag_loss_scaling_factor
            
            self.final_loss += - tf.reduce_sum(tf.log(tf.maximum(self.final_pred[1], self.thr)), 1)

        self.optimizer = tf.train.AdamOptimizer()
        gvs = self.optimizer.compute_gradients(tf.reduce_mean(self.final_loss))
        self.optimizer_step = self.optimizer.apply_gradients(gvs)

        
    def _build_comp_graph_fast_ver(self):
        '''
        Use shallow layers only to accelerate the training process.
        '''
        self.query_rels = tf.placeholder(tf.int32, [None])   # (dummy_batch_size) num_relevant_events in a batch
        self.refNode_source = tf.placeholder(tf.float32, [None, None])  # show where the refNode comes from (batch_size, dummy_batch_size)
        
        if self.flag_ruleLen_split:
            # Todo: ruleLen_split_ver
            pass
        else:
            num_cases = 8 if self.flag_int else 2   # [tqs, tqe] X [last_event, first_event] X [ts, te]
            self.random_walk_prob = tf.placeholder(tf.float32, [None, num_cases, self.num_rule])  # (dummy_batch_size, num_cases, num_rule)
            self.random_walk_ind = tf.placeholder(tf.float32, [None, 2, self.num_rule])  # (dummy_batch_size, 2, num_rule)


        # attention vec for different rules given the query relation
        self.attn_rule_embed = tf.Variable(np.random.randn(self.num_query, self.num_rule), dtype=tf.float32, name="attn_rule_embed")
        attn_rule = tf.nn.embedding_lookup(self.attn_rule_embed, self.query_rels)
        attn_rule = tf.nn.softmax(attn_rule, axis=1)
        self.attn_rule = attn_rule  # shape: (dummy_batch_size, num_rule)
        
        # attention vec for different positions: first or last event, using ts or te
        # [tqs, tqe] X [(first_event_ts, first_event_te), (last_event_ts, last_event_te), (first_event, last_event)]
        self.attn_refType_embed = [tf.Variable(self._random_uniform_unit(self.num_query, 2), dtype=tf.float32, name='attn_refType_embed_{}'.format(i)) 
                                   for i in range(3 * (int(self.flag_int) + 1))]
        attn_refType = [tf.nn.embedding_lookup(embed, self.query_rels) for embed in self.attn_refType_embed]
        attn_refType = [tf.nn.softmax(x, axis=1) for x in attn_refType] # each element (dummy_batch_size, 2)
        self.attn_refType = attn_refType
    
        if self.flag_ruleLen_split:
            pass    
        else:
            # prob we arrive at different reference events
            refNode_attn = [tf.reduce_sum(self.random_walk_prob[:, i, :] * attn_rule, axis=1, keep_dims=True) for i in range(num_cases)]  # shape: (dummy_batch_size, 1)
            refNode_attn_norm = [tf.reduce_sum(self.random_walk_ind[:, i, :] * attn_rule, axis=1, keep_dims=True) for i in range(2)]  # shape: (dummy_batch_size, 1)
            
            self.pred = []
            for i in range(int(self.flag_int)+1):
                # attn_refType[3*i+2]: choose first or last event; 
                # attn_refType[3*i][:, 0:1]: choose ts or te for the first event;
                # attn_refType[3*i][:, 1:2]: choose ts or te for the last event.
                probs = attn_refType[3*i+2][:, 0:1] * (attn_refType[3*i][:, 0:1]   * refNode_attn[4*i]   + attn_refType[3*i][:, 1:2]   * refNode_attn[4*i+1]) + \
                        attn_refType[3*i+2][:, 1:2] * (attn_refType[3*i+1][:, 0:1] * refNode_attn[4*i+2] + attn_refType[3*i+1][:, 1:2] * refNode_attn[4*i+3])  

                norm = attn_refType[3*i+2][:, 0:1] * (attn_refType[3*i][:, 0:1]   * refNode_attn_norm[0] + attn_refType[3*i][:, 1:2]   * refNode_attn_norm[0]) + \
                       attn_refType[3*i+2][:, 1:2] * (attn_refType[3*i+1][:, 0:1] * refNode_attn_norm[1] + attn_refType[3*i+1][:, 1:2] * refNode_attn_norm[1])  

                # refNode_source: shape: (batch_size, dummy_batch_size), prob: shape: (dummy_batch_size, num_timestamp)
                self.pred.append(tf.matmul(self.refNode_source, probs) / (tf.matmul(self.refNode_source, norm)) + 1e-20)


        # scaling the loss to make the learning more stable.
        if self.flag_loss_scaling_factor is not None:
            self.pred[0] /= self.flag_loss_scaling_factor
            if self.flag_int:
                self.pred[1] /= self.flag_loss_scaling_factor


        self.final_loss = -tf.reduce_sum(tf.log(tf.maximum(self.pred[0], self.thr)), 1)
        self.final_loss += -tf.reduce_sum(tf.log(tf.maximum(self.pred[1], self.thr)), 1) if self.flag_int else 0

        self.optimizer = tf.train.AdamOptimizer()
        gvs = self.optimizer.compute_gradients(tf.reduce_mean(self.final_loss))
        self.optimizer_step = self.optimizer.apply_gradients(gvs)


    def _run_comp_graph(self, sess, qq, hh, tt, connectivity_rel, connectivity_TR, probs, valid_sample_idx, inputs_for_enhancement, mode, to_fetch):
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

        if mode == 'Test':
            feed[self.probs] = [self._create_dummy_probs_for_prediction(probs[idx]) for idx in valid_sample_idx]

        for idx_rel in range(self.num_relation):
            feed[self.connectivity_rel[idx_rel]] = tf.SparseTensorValue(*connectivity_rel[idx_rel])
        for idx_TR in range(self.num_TR):
            feed[self.connectivity_TR[idx_TR]] = tf.SparseTensorValue(*connectivity_TR[idx_TR])


        if self.flag_state_vec_enhance:
            # Similar to fast version, we use shallow layers to enhance the state vector.
            query_rels, res_random_walk, refNode_index  = inputs_for_enhancement
            
            # Create random_walk_ind from res_random_walk.
            # We only need to distinguish the cases for first or last event.
            random_walk_ind = np.zeros((len(query_rels), 2, self.num_rule))  
            for i in [0, -1]:
                if len(res_random_walk[i]) > 0:
                    x, y = res_random_walk[i][:, 0].astype(int), res_random_walk[i][:, 1].astype(int)
                    random_walk_ind[x, i, y] = 1
            
            feed[self.query_rels] = query_rels
            feed[self.refNode_index] = refNode_index
            feed[self.random_walk_ind] = random_walk_ind


        fetches = to_fetch
        graph_output = sess.run(fetches, feed)
        return graph_output


    def _run_comp_graph_fast_ver(self, sess, query_rels, refNode_source, res_random_walk, to_fetch):
        feed = {}
        feed[self.query_rels] = query_rels
        feed[self.refNode_source] = refNode_source

        num_cases = 8 if self.flag_int else 2
        random_walk_prob = np.zeros((len(query_rels), num_cases, self.num_rule)) 
        random_walk_ind = np.zeros((len(query_rels), 2, self.num_rule))  # We only need to distinguish the cases for first or last event.
        for i in range(num_cases):
            if len(res_random_walk[i]) > 0:
                x, y, p = res_random_walk[i][:, 0].astype(int), res_random_walk[i][:, 1].astype(int), res_random_walk[i][:, 2]
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


    def _create_dummy_probs_for_prediction(self, probs):
        return [prob[0] for prob in probs]


    def update(self, sess, inputs):
        if self.flag_acceleration:
            qq, refNode_source, res_random_walk = inputs 
            to_fetch = [self.final_loss, self.optimizer_step]
            fetched = self._run_comp_graph_fast_ver(sess, qq, refNode_source, res_random_walk, to_fetch)
        else:
            qq, hh, tt, connectivity_rel, connectivity_TR, probs, valid_sample_idx, inputs_for_enhancement = inputs
            to_fetch = [self.final_loss, self.optimizer_step]
            fetched = self._run_comp_graph(sess, qq, hh, tt, connectivity_rel, connectivity_TR, probs, valid_sample_idx, inputs_for_enhancement, 'Train', to_fetch)
        return fetched[0]


    def predict(self, sess, inputs):
        if self.flag_acceleration:
            qq, refNode_source, res_random_walk, probs = inputs
            to_fetch = [self.pred]
            fetched = self._run_comp_graph_fast_ver(sess, qq, refNode_source, res_random_walk, probs, to_fetch)
            return fetched[0]
        else:
            qq, hh, tt, mdb, connectivity, probs, valid_sample_idx, valid_ref_event_idx = inputs
            to_fetch = [self.state_vector, self.attn_refType]
            fetched = self._run_comp_graph(sess, qq, hh, tt, mdb, connectivity, probs, valid_sample_idx, 'Test', to_fetch)

            # [tqs, tqe] X [last_event, first_event]
            state_vector = []
            for i in range(2*(int(self.flag_int)+1)):
                state_vector.append(np.expand_dims(fetched[0][i%2] * valid_ref_event_idx[i], axis=2))

            # Todo: check the order
            pred = []
            for i in range(int(self.flag_int)+1):
                input_prob = []
                for j in range(4):
                    input_prob.append(np.array([probs[j][idx] for idx in valid_sample_idx]))
                    
                pred.append(fetched[1][3*i+2][:, 0:1] *np.sum((np.expand_dims(fetched[1][3*i][:, 0:1], axis=2) * input_prob[4*i] + \
                                                            np.expand_dims(fetched[1][3*i+1][:, 1:2], axis=2) * input_prob[4*i+1]) \
                                                            * state_vector[0], axis = 1) + \
                            fetched[1][3*i+2][:, 1:2] *np.sum((np.expand_dims(fetched[1][3*i+2][:, 0:1], axis=2) * input_prob[4*i+2] + \
                                                            np.expand_dims(fetched[1][3*i+3][:, 1:2], axis=2) * input_prob[4*i+3]) \
                                                            * state_vector[1], axis = 1))

            return pred


    def get_rule_scores_fast_ver(self, sess):
        to_fetch = [self.attn_rule, self.attn_refType]

        qq = list(range(self.num_query))
        refNode_source = [[0 for _ in range(self.num_query)]]

        num_cases = 8 if self.flag_int else 2
        res_random_walk = [np.array([[0 for _ in range(3)] for _ in range(self.num_query)]) for _ in range(num_cases)]
        
        fetched = self._run_comp_graph_fast_ver(sess, qq, refNode_source, res_random_walk, to_fetch)

        return fetched[0], fetched[1]