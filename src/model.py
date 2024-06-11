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
        self.learning_rate = option.learning_rate
        self.accuracy = option.accuracy
        self.top_k = option.top_k

        self.num_entity = data['num_entity']
        self.num_operator = data['num_rel']
        self.num_TR = data['num_TR']
        self.num_rule = option.num_rule
        self.num_timestamp = len(data['timestamp_range'])
        self.num_query = data['num_rel']
        self.query_embed_size = option.query_embed_size


        self.flag_int = option.flag_interval
        self.flag_rel_TR_split = option.different_states_for_rel_and_TR
        self.flag_ruleLen_split = option.flag_ruleLen_split_ver
        self.flag_state_vec_enhan = option.flag_state_vec_enhancement
        self.flag_acceleration = option.flag_acceleration
        self.flag_loss_scaling_factor = 0.04  # YAGO, wiki

        np.random.seed(self.seed)

        if option.flag_ruleLen_split_ver:
            if self.flag_acceleration:
                self._init_ruleLen_embedding(data['pattern_ls'])
                self._build_graph_acc_ver()
            else:
                self._build_graph_ruleLen_split_ver()
        else:
            if self.flag_acceleration:
                self._build_graph_acc_ver()
            else:
                self._build_graph()


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


    def _build_input(self):
        self.tails = tf.placeholder(tf.int32, [None])
        self.heads = tf.placeholder(tf.int32, [None])
        self.targets_h = tf.one_hot(indices=self.heads, depth=self.num_entity)
        self.targets_t = tf.one_hot(indices=self.tails, depth=self.num_entity)


        self.query_rels = tf.placeholder(tf.int32, [None])
        self.queries = tf.placeholder(tf.int32, [None, self.num_step])
        self.query_embedding_params = tf.Variable(self._random_uniform_unit(
                                                      self.num_query + 1, # <END> token 
                                                      self.query_embed_size), 
                                                  dtype=tf.float32)

        rnn_inputs = tf.nn.embedding_lookup(self.query_embedding_params, 
                                            self.queries)

        return rnn_inputs


    def _build_variables(self):
        rnn_inputs = self._build_input()
        self.rnn_inputs = [tf.reshape(q, [-1, self.query_embed_size]) for q in tf.split(rnn_inputs, self.num_step, axis=1)]

        # Different channels: [tqs, tqe] X [last_event, first_event] X [ts, te] 
        # Order: 000, 001, 010, 011, 100, 101, 110, 111
        self.probs = tf.placeholder(tf.float32, [None, 4*(int(self.flag_int)+1), self.num_entity])      
        
        self.database = {r: tf.sparse_placeholder(
                            dtype=tf.float32, 
                            name="database_%d" % r)
                            for r in range(self.num_operator)}

        self.connectivity = {TR: tf.sparse_placeholder(
                            dtype=tf.float32,
                            name="connectivity_%d" % TR)
                            for TR in range(self.num_TR)}

        cellLen = self.rnn_state_size*4 if self.flag_rel_TR_split else self.rnn_state_size*2
        cellLen *= 2 if self.flag_int else 1

        cell = tf.nn.rnn_cell.LSTMCell(cellLen, state_is_tuple=True)
        self.cell = tf.nn.rnn_cell.MultiRNNCell([cell] * self.num_layer, 
                                                    state_is_tuple=True)
        init_state = self.cell.zero_state(tf.shape(self.heads)[0], tf.float32)

        self.rnn_outputs, self.final_state = tf.contrib.rnn.static_rnn(
                                                self.cell, 
                                                self.rnn_inputs,
                                                initial_state=init_state)

        # Different channels: [tqs, tqe] X [last_event, first_event]
        # Order: 00, 01, 10, 11
        self.W_rel = tf.Variable(np.random.randn(self.rnn_state_size, 2*(int(self.flag_int)+1), self.num_operator), dtype=tf.float32)
        self.b_rel = tf.Variable(np.zeros((1, 2*(int(self.flag_int)+1), self.num_operator)), dtype=tf.float32)
        
        self.W_TR = tf.Variable(np.random.randn(self.rnn_state_size, 2*(int(self.flag_int)+1), self.num_TR), dtype=tf.float32)
        self.b_TR = tf.Variable(np.zeros((1, 2*(int(self.flag_int)+1), self.num_TR)), dtype=tf.float32)


        # Different elements: [tqs, tqe] X [last_event, first_event]
        # each combination uses self.rnn_state_size entry in the rnn_output
        self.attn_rel = []
        for i in range(2*(int(self.flag_int)+1)):
            self.attn_rel.append([tf.split(tf.nn.softmax(
                                  tf.matmul(rnn_output[:, self.rnn_state_size*i:self.rnn_state_size*(i+1)], self.W_rel[:, i, :]) + self.b_rel[:, i, :],
                                   axis=1), self.num_operator, axis=1) 
                                  for rnn_output in self.rnn_outputs])     # todo: check rnn_output[:, self.rnn_state_size*i:self.rnn_state_size*(i+1)]

        # if self.flag_rel_TR_split, the RNN output will be split into two parts, the first half for relation and the second for TR;
        # otherwise, the RNN output will be used for both relation and TR
        idx_start_TR = self.rnn_state_size*2*(int(self.flag_int)+1) if self.flag_rel_TR_split else 0

        self.attn_TR = []
        for i in range(2*(int(self.flag_int)+1)):
            self.attn_TR.append([tf.split(tf.nn.softmax(
                                  tf.matmul(rnn_output[:, idx_start_TR + self.rnn_state_size*i: idx_start_TR + self.rnn_state_size*(i+1)], 
                                            self.W_TR[:, i, :]) + self.b_TR[:, i, :], axis=1), self.num_TR, axis=1) 
                                            for rnn_output in self.rnn_outputs])     # todo: check rnn_output[:, self.rnn_state_size*i:self.rnn_state_size*(i+1)]        


        # Different elements: [tqs, tqe] x [last_event, first_event]
        self.attn_memories = [[] for _ in range(2*(int(self.flag_int)+1))]

        self.memories = [tf.expand_dims(tf.one_hot(indices=self.heads, depth=self.num_entity), 1),
                         tf.expand_dims(tf.one_hot(indices=self.tails, depth=self.num_entity), 1)] * (int(self.flag_int)+1)
        self.memories = sum(self.memories, [])
        
        # [tqs, tqe] * [(first_event, last_event) , (first_ts, first_te), (last_ts, last_te)]
        self.attn_refType_embed = [tf.Variable(self._random_uniform_unit(self.num_query, 2), dtype=tf.float32)] * 3 * (int(self.flag_int)+1)

        if self.flag_state_vec_enhan:
            self.query_rels = tf.placeholder(tf.int32, [None])    # dummy batch
            self.refNode_source = tf.placeholder(tf.float32, [None, None])  # show where the refNode comes from
            self.res_random_walk = tf.placeholder(tf.float32, [None, self.num_rule])   # dummy batch
            self.refNode_index = tf.placeholder(tf.int32, [None])    # show the refNode index
            self.attn_rule_embed = tf.Variable(np.random.randn(self.num_query, self.num_rule), dtype=tf.float32)

        return 


    def _build_graph(self):
        '''
        Define forward process. Use both RNN and shallow layers.

        '''
        self._build_variables()
        
        # [tqs, tqe] * [(last_ts, last_te), (first_ts, first_te), (first_event, last_event)]
        attn_refType = [tf.nn.embedding_lookup(embed, self.query_rels) for embed in self.attn_refType_embed]
        attn_refType = [tf.nn.softmax(x, axis=1) for x in attn_refType]

        if self.flag_state_vec_enhan:
            attn_rule = tf.nn.embedding_lookup(self.attn_rule_embed, self.query_rels)
            attn_rule = tf.nn.softmax(attn_rule, axis=1)
            refNode_probs = tf.reduce_sum(self.res_random_walk * attn_rule, axis=1, keep_dims=True)
            state_vec_est = tf.matmul(self.refNode_source, refNode_probs)

        
        self.state_vector = []
        self.pred = []
        for i in range(2):
            # [last/first event]
            for t in range(self.num_step):
                self.attn_memories[i].append(tf.nn.softmax(tf.squeeze(tf.matmul(
                                            tf.expand_dims(self.rnn_outputs[t][:, self.rnn_state_size*i:self.rnn_state_size*(i+1)], 1), 
                                            tf.stack([rnn_output[:, self.rnn_state_size*i:self.rnn_state_size*(i+1)] 
                                                    for rnn_output in self.rnn_outputs[0:t+1]], axis=2)), 
                                                    squeeze_dims=[1])))
                
                memory_read = tf.squeeze(
                                tf.matmul(
                                    tf.expand_dims(self.attn_memories[i][t], 1), 
                                    self.memories[i]),
                                squeeze_dims=[1])
                
                if  t < self.num_step - 1:   
                    memory_read = tf.transpose(memory_read)
                    if t == 0:
                        # for t == 0, TR is fixed as ukn
                        memory_read = tf.sparse_tensor_dense_matmul(self.connectivity[0], memory_read)   # todo: define connectivity matrixs {0: ukn, 1: bf, 2: t, 3: af} for edges
                    else:
                        database_results = []
                        for TR in range(self.num_TR):   
                            # TR: [0: ukn, 1: bf, 2: t, 3: af]
                            product = tf.sparse_tensor_dense_matmul(self.connectivity[TR], memory_read)
                            database_results.append(tf.transpose(product) * self.attn_TR[i][t][TR])

                        added_database_results = tf.add_n(database_results)
                        if self.norm:
                            added_database_results /= tf.maximum(self.thr, tf.reduce_sum(added_database_results, axis=1, keep_dims=True))
                        memory_read = tf.identity(added_database_results)
                        memory_read = tf.transpose(memory_read)

                    database_results = []
                    for r in range(self.num_operator):  # now database is a diagonal matrix which shows the relation of matrix
                        product = tf.sparse_tensor_dense_matmul(self.database[r], memory_read)  # todo: check the shape of memory_read, whether it is the same
                        database_results.append(tf.transpose(product) * self.attn_rel[i][t][r])

                    added_database_results = tf.add_n(database_results)
                    if self.norm:
                        added_database_results /= tf.maximum(self.thr, tf.reduce_sum(added_database_results, axis=1, keep_dims=True))
                    if self.dropout > 0.:
                        added_database_results = tf.nn.dropout(added_database_results, keep_prob=1.-self.dropout)

                    self.memories[i] = tf.concat( 
                                        [self.memories[i], 
                                        tf.expand_dims(added_database_results, 1)],
                                        axis=1)
                else:
                    if self.flag_state_vec_enhan:
                        memory_read[self.refNode_index] = 0.5 * memory_read[self.refNode_index] + 0.5 * state_vec_est   # todo: distinguish first and last events

                    self.state_vector.append(tf.identity(memory_read))

                    memory_read = memory_read * (attn_refType[i][:, 0:1] * self.probs[:, 2*i, :] + \
                                                 attn_refType[i][:, 1:2] * self.probs[:, 2*i+1, :])
                    memory_read = tf.transpose(memory_read)
                    memory_read = tf.sparse_tensor_dense_matmul(self.connectivity[0], memory_read)
                    memory_read = tf.transpose(memory_read)
                    if i % 2 == 0:
                        self.pred.append(tf.reduce_sum(self.targets_h * memory_read, axis=1, keep_dims=True))
                    else:
                        self.pred.append(tf.reduce_sum(self.targets_t * memory_read, axis=1, keep_dims=True))

        self.attn_refType = attn_refType
        self.final_pred = [attn_refType[2][:, 0:1] * self.pred[0] + attn_refType[2][:, 1:2] * self.pred[1]]
        self.final_loss = - tf.reduce_sum(tf.log(tf.maximum(self.final_pred[0], self.thr)), 1)

        if self.flag_interval:
            for i in range(2):
                # [last/first event]
                memory_read = tf.identity(self.state_vector[0]) # use the same state vector for both tqs and tqe
                memory_read = memory_read * (attn_refType[3+i][:, 0:1] * self.probs[:, 2*i+4, :] + \
                                             attn_refType[3+i][:, 1:2] * self.probs[:, 2*i+5, :])
                memory_read = tf.transpose(memory_read)
                memory_read = tf.sparse_tensor_dense_matmul(self.connectivity[0], memory_read)
                memory_read = tf.transpose(memory_read)
                if i % 2 == 0:
                    self.pred.append(tf.reduce_sum(self.targets_h * memory_read, axis=1, keep_dims=True))
                else:
                    self.pred.append(tf.reduce_sum(self.targets_t * memory_read, axis=1, keep_dims=True))
            
            self.final_pred.append(attn_refType[5][:, 0:1] * self.pred[2] + attn_refType[5][:, 1:2] * self.pred[3])
            self.final_loss += - tf.reduce_sum(tf.log(tf.maximum(self.final_pred[1], self.thr)), 1)

        self.optimizer = tf.train.AdamOptimizer()
        gvs = self.optimizer.compute_gradients(tf.reduce_mean(self.final_loss))
        # capped_gvs = map(lambda (grad, var): self._clip_if_not_None(grad, var, -10., 10.), gvs) 
        # self.optimizer_step = self.optimizer.apply_gradients(capped_gvs)
        self.optimizer_step = self.optimizer.apply_gradients(gvs)


    def _build_graph_acc_ver(self):
        '''
        Use shallow layers only to accelerate the training process.
        '''
        self.query_rels = tf.placeholder(tf.int32, [None])   # (dummy_batch_size) num_relevant_events in a batch
        self.refNode_source = tf.placeholder(tf.float32, [None, None])  # show where the refNode comes from (batch_size, dummy_batch_size)
        
        num_cases = 8 if self.flag_int else 2
        self.random_walk_prob = tf.placeholder(tf.float32, [None, num_cases, self.num_rule])  # (dummy_batch_size, num_cases, num_rule)
        self.random_walk_ind = tf.placeholder(tf.float32, [None, 2, self.num_rule])  # (dummy_batch_size, 2, num_rule)


        if self.flag_ruleLen_split:
            # [tqs, tqe] X [last_event, first_event] X [ts, te]
            self.probs = tf.placeholder(tf.float32, [None, 4*(int(self.flag_int)+1), self.num_step-1, None])
        else:
            # Shape: train: (dummy_batch_size, 4*(int(self.flag_int)+1), 1), test: (dummy_batch_size, 4*(int(self.flag_int)+1), num_timestamp)
            self.probs = tf.placeholder(tf.float32, [None, 4*(int(self.flag_int)+1), None])  


        # attention vec for different rules given the query relation
        self.attn_rule_embed = tf.Variable(np.random.randn(self.num_query, self.num_rule),  dtype=tf.float32)
        attn_rule = tf.nn.embedding_lookup(self.attn_rule_embed, self.query_rels)
        attn_rule = tf.nn.softmax(attn_rule, axis=1)
        self.attn_rule = attn_rule  # shape: (dummy_batch_size, num_rule)
        
        # attention vec for different positions: first or last event, using ts or te
        # [tqs, tqe] X [(first_event_ts, first_event_te), (last_event_ts, last_event_te), (first_event, last_event)]
        self.attn_refType_embed = [tf.Variable(self._random_uniform_unit(self.num_query, 2), dtype=tf.float32) for _ in range(3 * (int(self.flag_int) + 1))]
        attn_refType = [tf.nn.embedding_lookup(embed, self.query_rels) for embed in self.attn_refType_embed]
        attn_refType = [tf.nn.softmax(x, axis=1) for x in attn_refType] # each element (dummy_batch_size, 2)
        self.attn_refType = attn_refType
    
        if self.flag_ruleLen_split:
            # Todo: update the code since we now use random_walk_prob instead of probs
            refNode_probs_mat = self.random_walk_prob * attn_rule
            ruleLen_table = [tf.nn.embedding_lookup(self.ruleLen_embedding[l], self.query_rels) for l in range(self.num_step-1)]
            refNode_probs = [tf.reduce_sum(refNode_probs_mat * ruleLen_table[l], axis=1, keep_dims=True) for l in range(self.num_step-1)]

            self.pred = []
            for i in range(int(self.flag_int)+1):
                probs_from_refNode = [attn_refType[3*i+2][:, 0:1] * (attn_refType[3*i][:, 0:1] * self.probs[:,4*i,l,:] + attn_refType[3*i][:, 1:2] * self.probs[:,4*i+1,l,:]) + \
                                      attn_refType[3*i+2][:, 1:2] * (attn_refType[3*i+1][:, 0:1]*self.probs[:,4*i+2,l,:] + attn_refType[3*i+1][:, 1:2] * self.probs[:,4*i+3,l,:]) \
                                      for l in range(self.num_step-1)]
                pred_refNode = 0
                for l in range(self.num_step-1):
                    pred_refNode += refNode_probs[l] * probs_from_refNode[l]

                self.pred.append(tf.matmul(self.refNode_source, pred_refNode))
                
        else:
            # prob we arrive at different reference events
            refNode_attn = []
            for i in range(num_cases):
                refNode_attn.append(tf.reduce_sum(self.random_walk_prob[:, i, :] * attn_rule, axis=1, keep_dims=True))  # shape: (dummy_batch_size, 1)
            
            refNode_attn_norm = []
            for i in range(2):
                refNode_attn_norm.append(tf.reduce_sum(self.random_walk_ind[:, i, :] * attn_rule, axis=1, keep_dims=True))  # shape: (dummy_batch_size, 1)
            
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
        # capped_gvs = map(lambda (grad, var): self._clip_if_not_None(grad, var, -10., 10.), gvs)
        # self.optimizer_step = self.optimizer.apply_gradients(capped_gvs)
        self.optimizer_step = self.optimizer.apply_gradients(gvs)



    def _run_graph(self, sess, qq, hh, tt, mdb, connectivity, probs, valid_sample_idx, mode, to_fetch):
        qq = [qq[idx] for idx in valid_sample_idx]
        hh = [hh[idx] for idx in valid_sample_idx]
        tt = [tt[idx] for idx in valid_sample_idx]

        feed = {}

        feed[self.queries] = [[q] * (self.num_step-1) + [self.num_query] for q in qq]
        feed[self.heads] = hh 
        feed[self.tails] = tt 
        feed[self.query_rels] = qq

        feed[self.probs] = [probs[idx] for idx in valid_sample_idx]

        if mode == 'Test':
            feed[self.probs] = [self.create_dummy_probs_for_prediction(probs[idx]) for idx in valid_sample_idx]

        for r in range(self.num_operator):
            feed[self.database[r]] = tf.SparseTensorValue(*mdb[r])
        for TR in range(self.num_TR):
            feed[self.connectivity[TR]] = tf.SparseTensorValue(*connectivity[TR])

        fetches = to_fetch
        graph_output = sess.run(fetches, feed)
        return graph_output


    def _run_graph_acc(self, sess, query_rels, refNode_source, res_random_walk, probs, to_fetch):
        feed = {}
        feed[self.query_rels] = query_rels
        feed[self.refNode_source] = refNode_source

        num_cases = 8 if self.flag_int else 2
        random_walk_prob = np.zeros((len(query_rels), num_cases, self.num_rule)) 
        random_walk_ind = np.zeros((len(query_rels), 2, self.num_rule))  # We only need to consider the cases for first or last event.
        for i in range(num_cases):
            if len(res_random_walk[i]) > 0:
                x, y, p = res_random_walk[i][:, 0].astype(int), res_random_walk[i][:, 1].astype(int), res_random_walk[i][:, 2]
                random_walk_prob[x, i, y] = p
                if i in [0, num_cases-1]:
                    random_walk_ind[x, min(i, 1), y] = 1
        feed[self.random_walk_prob] = random_walk_prob
        feed[self.random_walk_ind] = random_walk_ind

        feed[self.probs] = probs

        fetches = to_fetch
        graph_output = sess.run(fetches, feed)
        return graph_output


    def process_probs_for_prediction(self, probs):
        probs = np.array(probs)
        probs = np.transpose(probs, (0, 2, 1))
        probs = probs.reshape(-1, self.num_entity)
        return probs.tolist()


    def create_dummy_probs_for_prediction(self, probs):
        return [prob[0] for prob in probs]



    def update(self, sess, inputs):
        if self.flag_acceleration:
            qq, refNode_source, res_random_walk, probs = inputs 
            to_fetch = [self.final_loss, self.optimizer_step]
            fetched = self._run_graph_acc(sess, qq, refNode_source, res_random_walk, probs, to_fetch)
        else:
            qq, hh, tt, mdb, connectivity, probs, valid_sample_idx = inputs
            to_fetch = [self.final_loss, self.optimizer_step]
            fetched = self._run_graph(sess, qq, hh, tt, mdb, connectivity, probs, valid_sample_idx, 'Train', to_fetch)
        return fetched[0]



    def predict(self, sess, inputs):
        if self.flag_acceleration:
            qq, refNode_source, res_random_walk, probs = inputs
            to_fetch = [self.pred]
            fetched = self._run_graph_acc(sess, qq, refNode_source, res_random_walk, probs, to_fetch)
            return fetched[0]
        else:
            qq, hh, tt, mdb, connectivity, probs, valid_sample_idx, valid_ref_event_idx = inputs
            to_fetch = [self.state_vector, self.attn_refType]
            fetched = self._run_graph(sess, qq, hh, tt, mdb, connectivity, probs, valid_sample_idx, 'Test', to_fetch)

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



    def get_rule_scores_acc(self, sess):
        to_fetch = [self.attn_rule, self.attn_refType]

        qq = list(range(self.num_query))
        refNode_source = [[0] * (self.num_query)]

        num_cases = 8 if self.flag_int else 2
        res_random_walk = [np.array([[0] * 3] * (self.num_query))] * num_cases
        
        probs = [[[0]]*self.num_timestamp] * 4
        if self.flag_int:
            probs += [[[0]]*self.num_timestamp] * 4
        probs = np.array(probs).transpose(1, 0, 2)

        fetched = self._run_graph_acc(sess, qq, refNode_source, res_random_walk, probs, to_fetch)

        return fetched[0], fetched[1]