import numpy as np 
import tensorflow as tf
from utils import *


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


        self.flag_interval = option.flag_interval
        self.flag_different_states_for_rel_and_TR = option.different_states_for_rel_and_TR
        self.flag_ruleLen_split_ver = option.flag_ruleLen_split_ver
        self.flag_state_vec_enhancement = option.flag_state_vec_enhancement


        np.random.seed(self.seed)

        if option.flag_ruleLen_split_ver:
            if option.flag_acceleration:
                self._init_ruleLen_embedding(data['pattern_ls'])
                self._build_graph_acc_ver()
            else:
                self._build_graph_ruleLen_split_ver()
        else:
            if option.flag_acceleration:
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


    def _build_graph(self):
        rnn_inputs = self._build_input()
        self.rnn_inputs = [tf.reshape(q, [-1, self.query_embed_size]) 
                           for q in tf.split(rnn_inputs, 
                                             self.num_step, 
                                             axis=1)]

        self.ts_probs_last_event_ts = tf.placeholder(tf.float32, [None, self.num_entity])
        self.ts_probs_last_event_te = tf.placeholder(tf.float32, [None, self.num_entity])
        self.ts_probs_first_event_ts = tf.placeholder(tf.float32, [None, self.num_entity])
        self.ts_probs_first_event_te = tf.placeholder(tf.float32, [None, self.num_entity])

        if self.flag_interval:
            self.te_probs_last_event_ts = tf.placeholder(tf.float32, [None, self.num_entity])
            self.te_probs_last_event_te = tf.placeholder(tf.float32, [None, self.num_entity])
            self.te_probs_first_event_ts = tf.placeholder(tf.float32, [None, self.num_entity])
            self.te_probs_first_event_te = tf.placeholder(tf.float32, [None, self.num_entity])


        self.database = {r: tf.sparse_placeholder(
                            dtype=tf.float32, 
                            name="database_%d" % r)
                            for r in xrange(self.num_operator)}


        self.connectivity = {TR: tf.sparse_placeholder(
                            dtype=tf.float32,
                            name="connectivity_%d" % TR)
                            for TR in xrange(self.num_TR)}

        if flag_state_vec_enhancement:
            self.query_rels = tf.placeholder(tf.int32, [None])    # dummy batch
            self.refNode_source = tf.placeholder(tf.float32, [None, None])  # show where the refNode comes from
            self.res_random_walk = tf.placeholder(tf.float32, [None, self.num_rule])   # dummy batch
            self.refNode_index = tf.placeholder(tf.int32, [None])    # show the refNode index
            self.attn_rule_embedding = tf.Variable(np.random.randn(self.num_query, self.num_rule),  dtype=tf.float32)

            attn_rule = tf.nn.embedding_lookup(self.attn_rule_embedding, self.query_rels)
            attn_rule = tf.nn.softmax(attn_rule, axis=1)
            refNode_probs = tf.reduce_sum(self.res_random_walk * attn_rule, axis=1, keep_dims=True)
            state_vec_est = tf.matmul(self.refNode_source, refNode_probs)



        if self.flag_interval:
            if self.flag_different_states_for_rel_and_TR:
                cellLen = self.rnn_state_size * 8
            else:
                cellLen = self.rnn_state_size * 4
        else:
            if self.flag_different_states_for_rel_and_TR:
                cellLen = self.rnn_state_size * 4
            else:
                cellLen = self.rnn_state_size * 2


        cell = tf.nn.rnn_cell.LSTMCell(cellLen, state_is_tuple=True)


        self.cell = tf.nn.rnn_cell.MultiRNNCell([cell] * self.num_layer, 
                                                    state_is_tuple=True)


        init_state = self.cell.zero_state(tf.shape(self.heads)[0], tf.float32)


        self.rnn_outputs, self.final_state = tf.contrib.rnn.static_rnn(
                                                self.cell, 
                                                self.rnn_inputs,
                                                initial_state=init_state)


        self.ts_W_first_event = tf.Variable(np.random.randn(
                                                self.rnn_state_size, 
                                                self.num_operator), 
                                            dtype=tf.float32)
        self.ts_b_first_event = tf.Variable(np.zeros(
                                                (1, self.num_operator)), 
                                            dtype=tf.float32)


        self.ts_W_TR_first_event = tf.Variable(np.random.randn(
                                                    self.rnn_state_size, 
                                                    self.num_TR), 
                                                dtype=tf.float32)
        self.ts_b_TR_first_event = tf.Variable(np.zeros(
                                                    (1, self.num_TR)), 
                                                dtype=tf.float32)



        self.ts_W_last_event = tf.Variable(np.random.randn(
                                                self.rnn_state_size, 
                                                self.num_operator), 
                                            dtype=tf.float32)
        self.ts_b_last_event = tf.Variable(np.zeros(
                                                (1, self.num_operator)), 
                                            dtype=tf.float32)


        self.ts_W_TR_last_event = tf.Variable(np.random.randn(
                                self.rnn_state_size, 
                                self.num_TR), 
                            dtype=tf.float32)
        self.ts_b_TR_last_event = tf.Variable(np.zeros(
                                (1, self.num_TR)), 
                            dtype=tf.float32)



        if self.flag_interval:
            self.te_W_first_event = tf.Variable(np.random.randn(
                                    self.rnn_state_size, 
                                    self.num_operator), 
                                dtype=tf.float32)
            self.te_b_first_event = tf.Variable(np.zeros(
                                    (1, self.num_operator)), 
                                dtype=tf.float32)


            self.te_W_TR_first_event = tf.Variable(np.random.randn(
                                    self.rnn_state_size, 
                                    self.num_TR), 
                                dtype=tf.float32)
            self.te_b_TR_first_event = tf.Variable(np.zeros(
                                    (1, self.num_TR)), 
                                dtype=tf.float32)


            self.te_W_last_event = tf.Variable(np.random.randn(
                                    self.rnn_state_size, 
                                    self.num_operator), 
                                dtype=tf.float32)
            self.te_b_last_event = tf.Variable(np.zeros(
                                    (1, self.num_operator)), 
                                dtype=tf.float32)


            self.te_W_TR_last_event = tf.Variable(np.random.randn(
                                    self.rnn_state_size, 
                                    self.num_TR), 
                                dtype=tf.float32)
            self.te_b_TR_last_event = tf.Variable(np.zeros(
                                    (1, self.num_TR)), 
                                dtype=tf.float32)


        self.ts_attention_operators_first_event = [tf.split(
                                                    tf.nn.softmax(
                                                      tf.matmul(rnn_output[:, :self.rnn_state_size], self.ts_W_first_event)\
                                                                     + self.ts_b_first_event,
                                                                     axis=1),
                                                        self.num_operator, 
                                                        axis=1) 
                                                        for rnn_output in self.rnn_outputs]     # todo: check rnn_output[:, :self.rnn_state_size]


        self.ts_attention_operators_last_event = [tf.split(
                                                    tf.nn.softmax(
                                                      tf.matmul(rnn_output[:, self.rnn_state_size:self.rnn_state_size*2], self.ts_W_last_event) \
                                                                    + self.ts_b_last_event,
                                                                    axis=1), 
                                                        self.num_operator, 
                                                        axis=1) 
                                                        for rnn_output in self.rnn_outputs]     # todo: check rnn_output[:, self.rnn_state_size:self.rnn_state_size*2]

        state_start_TR = 0
        if self.flag_different_states_for_rel_and_TR:
            state_start_TR = self.rnn_state_size*2


        self.ts_attention_operators_TR_first_event = [tf.split(
                                                      tf.nn.softmax(
                                                      tf.matmul(rnn_output[:, state_start_TR: state_start_TR + self.rnn_state_size],
                                                                 self.ts_W_TR_first_event) + self.ts_b_TR_first_event, axis=1), 
                                                        self.num_TR, 
                                                        axis=1) 
                                                        for rnn_output in self.rnn_outputs]     # todo: check rnn_output[:, :self.rnn_state_size]


        self.ts_attention_operators_TR_last_event = [tf.split(
                                                      tf.nn.softmax(
                                                      tf.matmul(rnn_output[:, state_start_TR + self.rnn_state_size: state_start_TR + self.rnn_state_size*2],
                                                                    self.ts_W_TR_last_event) + self.ts_b_TR_last_event, axis=1), 
                                                        self.num_TR, 
                                                        axis=1) 
                                                        for rnn_output in self.rnn_outputs]     # todo: check rnn_output[:, self.rnn_state_size:self.rnn_state_size*2]



        if self.flag_interval:
            self.te_attention_operators_first_event = [tf.split(
                                                        tf.nn.softmax(
                                                          tf.matmul(rnn_output[:, state_start_TR + self.rnn_state_size*2: state_start_TR + self.rnn_state_size*3], 
                                                                      self.te_W_first_event) + self.te_b_first_event, axis=1), 
                                                            self.num_operator, 
                                                            axis=1) 
                                                            for rnn_output in self.rnn_outputs]


            self.te_attention_operators_last_event = [tf.split(
                                                        tf.nn.softmax(
                                                          tf.matmul(rnn_output[:, state_start_TR + self.rnn_state_size*3: state_start_TR + self.rnn_state_size*4], 
                                                                        self.te_W_last_event) + self.te_b_last_event, axis=1), 
                                                            self.num_operator, 
                                                            axis=1) 
                                                            for rnn_output in self.rnn_outputs]


            self.te_attention_operators_TR_first_event = [tf.split(
                                                          tf.nn.softmax(
                                                          tf.matmul(rnn_output[:, state_start_TR*2 + self.rnn_state_size*2: state_start_TR*2 + self.rnn_state_size*3], 
                                                                     self.te_W_TR_first_event) + self.te_b_TR_first_event, axis=1), 
                                                            self.num_TR, 
                                                            axis=1) 
                                                            for rnn_output in self.rnn_outputs]     # todo: check rnn_output[:, :self.rnn_state_size]


            self.te_attention_operators_TR_last_event = [tf.split(
                                                          tf.nn.softmax(
                                                          tf.matmul(rnn_output[:, state_start_TR*2 + self.rnn_state_size*3: state_start_TR*2 + self.rnn_state_size*4],
                                                                        self.te_W_TR_last_event) + self.te_b_TR_last_event, axis=1), 
                                                            self.num_TR, 
                                                            axis=1) 
                                                            for rnn_output in self.rnn_outputs]     # todo: check rnn_output[:, self.rnn_state_size:self.rnn_state_size*2]


        self.ts_attention_memories_first_event = []
        self.ts_attention_memories_last_event = []


        self.ts_memories_last_event = tf.expand_dims(
                                         tf.one_hot(
                                                indices=self.heads, 
                                                depth=self.num_entity), 1) 

        self.ts_memories_first_event = tf.expand_dims(
                                         tf.one_hot(
                                                indices=self.tails, 
                                                depth=self.num_entity), 1) 


        if self.flag_interval:
            self.te_attention_memories_first_event = []
            self.te_attention_memories_last_event = []

            self.te_memories_last_event = tf.expand_dims(
                                             tf.one_hot(
                                                    indices=self.heads, 
                                                    depth=self.num_entity), 1) 

            self.te_memories_first_event = tf.expand_dims(
                                             tf.one_hot(
                                                    indices=self.tails, 
                                                    depth=self.num_entity), 1) 


        self.ts_attn_refType_first_event_embedding = tf.Variable(self._random_uniform_unit(self.num_query, 2), dtype=tf.float32)
        self.ts_attn_refType_last_event_embedding = tf.Variable(self._random_uniform_unit(self.num_query, 2), dtype=tf.float32)
        self.ts_attn_refType_embedding = tf.Variable(self._random_uniform_unit(self.num_query, 2), dtype=tf.float32)


        ts_attention_refType_first_event = tf.nn.embedding_lookup(self.ts_attn_refType_first_event_embedding, self.query_rels)
        ts_attention_refType_first_event = tf.nn.softmax(ts_attention_refType_first_event, axis=1)

        ts_attention_refType_last_event = tf.nn.embedding_lookup(self.ts_attn_refType_last_event_embedding, self.query_rels)
        ts_attention_refType_last_event = tf.nn.softmax(ts_attention_refType_last_event, axis=1)

        ts_attention_refType = tf.nn.embedding_lookup(self.ts_attn_refType_embedding, self.query_rels)
        ts_attention_refType = tf.nn.softmax(ts_attention_refType, axis=1)


        if self.flag_interval:
            self.te_attn_refType_first_event_embedding = tf.Variable(self._random_uniform_unit(self.num_query, 2), dtype=tf.float32)
            self.te_attn_refType_last_event_embedding = tf.Variable(self._random_uniform_unit(self.num_query, 2), dtype=tf.float32)
            self.te_attn_refType_embedding = tf.Variable(self._random_uniform_unit(self.num_query, 2), dtype=tf.float32)


            te_attention_refType_first_event = tf.nn.embedding_lookup(self.te_attn_refType_first_event_embedding, self.query_rels)
            te_attention_refType_first_event = tf.nn.softmax(te_attention_refType_first_event, axis=1)

            te_attention_refType_last_event = tf.nn.embedding_lookup(self.te_attn_refType_last_event_embedding, self.query_rels)
            te_attention_refType_last_event = tf.nn.softmax(te_attention_refType_last_event, axis=1)

            te_attention_refType = tf.nn.embedding_lookup(self.te_attn_refType_embedding, self.query_rels)
            te_attention_refType = tf.nn.softmax(te_attention_refType, axis=1)


        # define forward process

        # use last event to predict
        for t in xrange(self.num_step):
            self.ts_attention_memories_last_event.append(
                            tf.nn.softmax(
                            tf.squeeze(
                                tf.matmul(
                                    tf.expand_dims(self.rnn_outputs[t][:, self.rnn_state_size:self.rnn_state_size*2], 1), 
                                    tf.stack([rnn_output[:, self.rnn_state_size:self.rnn_state_size*2] for rnn_output in self.rnn_outputs[0:t+1]], axis=2)), 
                            squeeze_dims=[1])))
            

            memory_read = tf.squeeze(
                            tf.matmul(
                                tf.expand_dims(self.ts_attention_memories_last_event[t], 1), 
                                self.ts_memories_last_event),
                            squeeze_dims=[1])

            if t == 0:
                memory_read = tf.transpose(memory_read)
                memory_read = tf.sparse_tensor_dense_matmul(self.connectivity[0], memory_read)   # todo: define connectivity matrixs {0: ukn, 1: bf, 2: t, 3: af} for edges

                database_results = []
                for r in xrange(self.num_operator):  # now database is a diagonal matrix which shows the relation of matrix
                    product = tf.sparse_tensor_dense_matmul(self.database[r], memory_read)  # todo: check the shape of memory_read, whether it is the same
                    database_results.append(tf.transpose(product) * self.ts_attention_operators_last_event[t][r])

                added_database_results = tf.add_n(database_results)
                if self.norm:
                    added_database_results /= tf.maximum(self.thr, tf.reduce_sum(added_database_results, axis=1, keep_dims=True))
                
                if self.dropout > 0.:
                  added_database_results = tf.nn.dropout(added_database_results, keep_prob=1.-self.dropout)

                self.ts_memories_last_event = tf.concat( 
                    [self.ts_memories_last_event, 
                    tf.expand_dims(added_database_results, 1)],
                    axis=1)


            elif t < self.num_step - 1:
                memory_read = tf.transpose(memory_read)

                database_results = []
                for TR in xrange(self.num_TR):   # self.num_TR = 4
                    product = tf.sparse_tensor_dense_matmul(self.connectivity[TR], memory_read)
                    database_results.append(tf.transpose(product) * self.ts_attention_operators_TR_last_event[t][TR])

                added_database_results = tf.add_n(database_results)
                if self.norm:
                    added_database_results /= tf.maximum(self.thr, tf.reduce_sum(added_database_results, axis=1, keep_dims=True))

                memory_read = tf.identity(added_database_results)
                memory_read = tf.transpose(memory_read)

                database_results = []
                for r in xrange(self.num_operator):  # now database is a diagonal matrix which shows the relation of matrix
                    product = tf.sparse_tensor_dense_matmul(self.database[r], memory_read)  # todo: check the shape of memory_read, whether it is the same
                    database_results.append(tf.transpose(product) * self.ts_attention_operators_last_event[t][r])

                added_database_results = tf.add_n(database_results)
                if self.norm:
                    added_database_results /= tf.maximum(self.thr, tf.reduce_sum(added_database_results, axis=1, keep_dims=True))

                if self.dropout > 0.:
                  added_database_results = tf.nn.dropout(added_database_results, keep_prob=1.-self.dropout)

                self.ts_memories_last_event = tf.concat( 
                    [self.ts_memories_last_event, 
                    tf.expand_dims(added_database_results, 1)],
                    axis=1)
            else:

                if self.flag_state_vec_enhancement:
                    memory_read[self.refNode_index] = 0.5 * memory_read[self.refNode_index] + 0.5 * state_vec_est   # todo: distinguish first and last events

                self.state_vector_last_event = tf.identity(memory_read)

                memory_read = memory_read * (ts_attention_refType_last_event[:, 0:1] * self.ts_probs_last_event_ts + \
                                             ts_attention_refType_last_event[:, 1:2] * self.ts_probs_last_event_te)
                memory_read = tf.transpose(memory_read)
                memory_read = tf.sparse_tensor_dense_matmul(self.connectivity[0], memory_read)
                memory_read = tf.transpose(memory_read)
                self.ts_predictions_last_event = tf.reduce_sum(self.targets_h * memory_read, axis=1, keep_dims=True)

        ###############################################



        for t in xrange(self.num_step):
            self.ts_attention_memories_first_event.append(
                            tf.nn.softmax(
                            tf.squeeze(
                                tf.matmul(
                                    tf.expand_dims(self.rnn_outputs[t][:, :self.rnn_state_size], 1), 
                                    tf.stack([rnn_output[:, :self.rnn_state_size] for rnn_output in self.rnn_outputs[0:t+1]], axis=2)), 
                            squeeze_dims=[1])))

            memory_read = tf.squeeze(
                            tf.matmul(
                                tf.expand_dims(self.ts_attention_memories_first_event[t], 1), 
                                self.ts_memories_first_event),
                            squeeze_dims=[1])

            if t == 0:
                memory_read = tf.transpose(memory_read)
                memory_read = tf.sparse_tensor_dense_matmul(self.connectivity[0], memory_read)   # todo: define connectivity matrixs {0: ukn, 1: bf, 2: t, 3: af} for edges

                database_results = []
                for r in xrange(self.num_operator):  # now database is a diagonal matrix which shows the relation of matrix
                    product = tf.sparse_tensor_dense_matmul(self.database[r], memory_read)  # todo: check the shape of memory_read, whether it is the same
                    database_results.append(tf.transpose(product) * self.ts_attention_operators_first_event[t][r])

                added_database_results = tf.add_n(database_results)
                if self.norm:
                    added_database_results /= tf.maximum(self.thr, tf.reduce_sum(added_database_results, axis=1, keep_dims=True))
                
                if self.dropout > 0.:
                  added_database_results = tf.nn.dropout(added_database_results, keep_prob=1.-self.dropout)


                self.ts_memories_first_event = tf.concat( 
                    [self.ts_memories_first_event, 
                    tf.expand_dims(added_database_results, 1)],
                    axis=1)


            elif t < self.num_step - 1:
                memory_read = tf.transpose(memory_read)

                database_results = []
                for TR in xrange(self.num_TR):   # self.num_TR = 4
                    product = tf.sparse_tensor_dense_matmul(self.connectivity[TR], memory_read)
                    database_results.append(tf.transpose(product) * self.ts_attention_operators_TR_first_event[t][TR])

                added_database_results = tf.add_n(database_results)
                if self.norm:
                    added_database_results /= tf.maximum(self.thr, tf.reduce_sum(added_database_results, axis=1, keep_dims=True))

                memory_read = tf.identity(added_database_results)
                memory_read = tf.transpose(memory_read)

                database_results = []
                for r in xrange(self.num_operator):  # now database is a diagonal matrix which shows the relation of matrix
                    product = tf.sparse_tensor_dense_matmul(self.database[r], memory_read)  # todo: check the shape of memory_read, whether it is the same
                    database_results.append(tf.transpose(product) * self.ts_attention_operators_first_event[t][r])

                added_database_results = tf.add_n(database_results)
                if self.norm:
                    added_database_results /= tf.maximum(self.thr, tf.reduce_sum(added_database_results, axis=1, keep_dims=True))

                if self.dropout > 0.:
                  added_database_results = tf.nn.dropout(added_database_results, keep_prob=1.-self.dropout)


                self.ts_memories_first_event = tf.concat( 
                    [self.ts_memories_first_event, 
                    tf.expand_dims(added_database_results, 1)],
                    axis=1)
            else:
                if self.flag_state_vec_enhancement:
                    memory_read[self.refNode_index] = 0.5 * memory_read[self.refNode_index] + 0.5 * state_vec_est   # todo: distinguish first and last events

                self.state_vector_first_event = tf.identity(memory_read)

                memory_read = memory_read * (ts_attention_refType_first_event[:, 0:1] * self.ts_probs_first_event_ts + \
                                             ts_attention_refType_first_event[:, 1:2] * self.ts_probs_first_event_te)
                memory_read = tf.transpose(memory_read)
                memory_read = tf.sparse_tensor_dense_matmul(self.connectivity[0], memory_read)
                memory_read = tf.transpose(memory_read)
                self.ts_predictions_first_event = tf.reduce_sum(self.targets_t * memory_read, axis=1, keep_dims=True)

            ###############################################

        self.ts_predictions = ts_attention_refType[:, 0:1] * self.ts_predictions_first_event + ts_attention_refType[:, 1:2] * self.ts_predictions_last_event




        if not self.flag_interval:
            self.final_loss = - tf.reduce_sum(tf.log(tf.maximum(self.ts_predictions, self.thr)), 1)
        else:
            memory_read = tf.identity(self.state_vector_last_event)
            self.state_vector_te_last_event = tf.identity(memory_read)
            memory_read = memory_read * (te_attention_refType_last_event[:, 0:1] * self.te_probs_last_event_ts + \
                                         te_attention_refType_last_event[:, 1:2] * self.te_probs_last_event_te)
            memory_read = tf.transpose(memory_read)
            memory_read = tf.sparse_tensor_dense_matmul(self.connectivity[0], memory_read)
            memory_read = tf.transpose(memory_read)
            self.te_predictions_last_event = tf.reduce_sum(self.targets_h * memory_read, axis=1, keep_dims=True)

            # ###############################################

            memory_read = tf.identity(self.state_vector_first_event)
            memory_read = memory_read * (te_attention_refType_first_event[0, 0:1] * self.te_probs_first_event_ts + \
                                         te_attention_refType_first_event[0, 1:2] * self.te_probs_first_event_te)
            memory_read = tf.transpose(memory_read)
            memory_read = tf.sparse_tensor_dense_matmul(self.connectivity[0], memory_read)
            memory_read = tf.transpose(memory_read)
            self.te_predictions_first_event = tf.reduce_sum(self.targets_t * memory_read, axis=1, keep_dims=True)

            ###############################################

            self.te_predictions = te_attention_refType[:, 0:1] * self.te_predictions_first_event + te_attention_refType[:, 1:2] * self.te_predictions_last_event

            self.final_loss = - tf.reduce_sum(tf.log(tf.maximum(self.ts_predictions, self.thr)), 1) - tf.reduce_sum(tf.log(tf.maximum(self.te_predictions, self.thr)), 1)



            self.ts_attention_refType_first_event, self.ts_attention_refType_last_event, self.ts_attention_refType = ts_attention_refType_first_event, ts_attention_refType_last_event, ts_attention_refType

            if self.flag_interval:
                self.te_attention_refType_first_event, self.te_attention_refType_last_event, self.te_attention_refType = te_attention_refType_first_event, te_attention_refType_last_event, te_attention_refType


        self.optimizer = tf.train.AdamOptimizer()
        gvs = self.optimizer.compute_gradients(tf.reduce_mean(self.final_loss))
        capped_gvs = map(
            lambda (grad, var): self._clip_if_not_None(grad, var, -10., 10.), gvs) 
        self.optimizer_step = self.optimizer.apply_gradients(capped_gvs)




    def _build_graph_ruleLen_split_ver(self):
        rnn_inputs = self._build_input()
        self.rnn_inputs = [tf.reshape(q, [-1, self.query_embed_size]) 
                           for q in tf.split(rnn_inputs, 
                                             self.num_step, 
                                             axis=1)]

        self.ts_probs_last_event_ts = tf.placeholder(tf.float32, [None, self.num_entity, self.num_step-1])
        self.ts_probs_last_event_te = tf.placeholder(tf.float32, [None, self.num_entity, self.num_step-1])
        self.ts_probs_first_event_ts = tf.placeholder(tf.float32, [None, self.num_entity, self.num_step-1])
        self.ts_probs_first_event_te = tf.placeholder(tf.float32, [None, self.num_entity, self.num_step-1])

        if self.flag_interval:
            self.te_probs_last_event_ts = tf.placeholder(tf.float32, [None, self.num_entity, self.num_step-1])
            self.te_probs_last_event_te = tf.placeholder(tf.float32, [None, self.num_entity, self.num_step-1])
            self.te_probs_first_event_ts = tf.placeholder(tf.float32, [None, self.num_entity, self.num_step-1])
            self.te_probs_first_event_te = tf.placeholder(tf.float32, [None, self.num_entity, self.num_step-1])


        if self.flag_interval:
            if self.flag_different_states_for_rel_and_TR:
                cellLen_single_step = self.rnn_state_size * 8
            else:
                cellLen_single_step = self.rnn_state_size * 4
        else:
            if self.flag_different_states_for_rel_and_TR:
                cellLen_single_step = self.rnn_state_size * 4
            else:
                cellLen_single_step = self.rnn_state_size * 2


        cell = tf.nn.rnn_cell.LSTMCell(cellLen_single_step * (self.num_step-1), 
                                                                 state_is_tuple=True)


        self.cell = tf.nn.rnn_cell.MultiRNNCell([cell] * self.num_layer, 
                                                    state_is_tuple=True)


        init_state = self.cell.zero_state(tf.shape(self.heads)[0], tf.float32)


        self.rnn_outputs, self.final_state = tf.contrib.rnn.static_rnn(
                                                self.cell, 
                                                self.rnn_inputs,
                                                initial_state=init_state)


        self.ts_W_first_event = tf.Variable(np.random.randn(
                                                self.rnn_state_size * (self.num_step-1), 
                                                self.num_operator), 
                                            dtype=tf.float32)
        self.ts_b_first_event = tf.Variable(np.zeros(
                                                (self.num_step-1, self.num_operator)), 
                                            dtype=tf.float32)


        self.ts_W_TR_first_event = tf.Variable(np.random.randn(
                                                    self.rnn_state_size * (self.num_step-1), 
                                                    self.num_TR), 
                                                dtype=tf.float32)
        self.ts_b_TR_first_event = tf.Variable(np.zeros(
                                                    (self.num_step-1, self.num_TR)), 
                                                dtype=tf.float32)



        self.ts_W_last_event = tf.Variable(np.random.randn(
                                                self.rnn_state_size * (self.num_step-1), 
                                                self.num_operator), 
                                            dtype=tf.float32)
        self.ts_b_last_event = tf.Variable(np.zeros(
                                                (self.num_step-1, self.num_operator)), 
                                            dtype=tf.float32)


        self.ts_W_TR_last_event = tf.Variable(np.random.randn(
                                self.rnn_state_size * (self.num_step-1), 
                                self.num_TR), 
                            dtype=tf.float32)
        self.ts_b_TR_last_event = tf.Variable(np.zeros(
                                (self.num_step-1, self.num_TR)), 
                            dtype=tf.float32)



        if self.flag_interval:
            self.te_W_first_event = tf.Variable(np.random.randn(
                                    self.rnn_state_size * (self.num_step-1), 
                                    self.num_operator), 
                                dtype=tf.float32)
            self.te_b_first_event = tf.Variable(np.zeros(
                                    (self.num_step-1, self.num_operator)), 
                                dtype=tf.float32)


            self.te_W_TR_first_event = tf.Variable(np.random.randn(
                                    self.rnn_state_size * (self.num_step-1), 
                                    self.num_TR), 
                                dtype=tf.float32)
            self.te_b_TR_first_event = tf.Variable(np.zeros(
                                    (self.num_step-1, self.num_TR)), 
                                dtype=tf.float32)


            self.te_W_last_event = tf.Variable(np.random.randn(
                                    self.rnn_state_size * (self.num_step-1), 
                                    self.num_operator), 
                                dtype=tf.float32)
            self.te_b_last_event = tf.Variable(np.zeros(
                                    (self.num_step-1, self.num_operator)), 
                                dtype=tf.float32)


            self.te_W_TR_last_event = tf.Variable(np.random.randn(
                                    self.rnn_state_size * (self.num_step-1), 
                                    self.num_TR), 
                                dtype=tf.float32)
            self.te_b_TR_last_event = tf.Variable(np.zeros(
                                    (self.num_step-1, self.num_TR)), 
                                dtype=tf.float32)



        self.ts_attention_operators_first_event = [[tf.split(
                                                    tf.nn.softmax(
                                                      tf.matmul(rnn_output[:, cellLen_single_step*l: cellLen_single_step*l + self.rnn_state_size],
                                                                self.ts_W_first_event[self.rnn_state_size*l :self.rnn_state_size*l + self.rnn_state_size])\
                                                                     + self.ts_b_first_event[l],
                                                                     axis=1),
                                                        self.num_operator, 
                                                        axis=1) 
                                                        for rnn_output in self.rnn_outputs] for l in range(self.num_step-1)]     # todo: check rnn_output[:, :self.rnn_state_size]


        self.ts_attention_operators_last_event = [[tf.split(
                                                   tf.nn.softmax(
                                                   tf.matmul(rnn_output[:, cellLen_single_step*l + self.rnn_state_size: cellLen_single_step*l + self.rnn_state_size*2],
                                                    self.ts_W_last_event[self.rnn_state_size*l :self.rnn_state_size*l + self.rnn_state_size]) \
                                                        + self.ts_b_last_event[l],
                                                        axis=1), 
                                                        self.num_operator, 
                                                        axis=1) 
                                                        for rnn_output in self.rnn_outputs] for l in range(self.num_step-1)]    # todo: check rnn_output[:, self.rnn_state_size:self.rnn_state_size*2]

        state_start_TR = 0
        if self.flag_different_states_for_rel_and_TR:
            state_start_TR = self.rnn_state_size*2


        self.ts_attention_operators_TR_first_event = [[tf.split(
                                                      tf.nn.softmax(
                                                      tf.matmul(rnn_output[:, cellLen_single_step*l + state_start_TR: cellLen_single_step*l + state_start_TR + self.rnn_state_size],
                                                                 self.ts_W_TR_first_event[self.rnn_state_size*l :self.rnn_state_size*l + self.rnn_state_size]) \
                                                                 + self.ts_b_TR_first_event[l], axis=1), 
                                                        self.num_TR, 
                                                        axis=1) 
                                                        for rnn_output in self.rnn_outputs] for l in range(self.num_step-1)]     # todo: check rnn_output[:, :self.rnn_state_size]


        self.ts_attention_operators_TR_last_event = [[tf.split(
                                                      tf.nn.softmax(
                                                      tf.matmul(rnn_output[:, cellLen_single_step*l + state_start_TR + self.rnn_state_size: cellLen_single_step*l + state_start_TR + self.rnn_state_size*2],
                                                        self.ts_W_TR_last_event[self.rnn_state_size*l :self.rnn_state_size*l + self.rnn_state_size]) \
                                                        + self.ts_b_TR_last_event[l], axis=1), 
                                                        self.num_TR, 
                                                        axis=1) 
                                                        for rnn_output in self.rnn_outputs] for l in range(self.num_step-1)]     # todo: check rnn_output[:, self.rnn_state_size:self.rnn_state_size*2]



        if self.flag_interval:
            self.te_attention_operators_first_event = [[tf.split(
                                                        tf.nn.softmax(
                                                          tf.matmul(rnn_output[:, cellLen_single_step*l + state_start_TR + self.rnn_state_size*2: cellLen_single_step*l + state_start_TR + self.rnn_state_size*3], 
                                                                      self.te_W_first_event[self.rnn_state_size*l :self.rnn_state_size*l + self.rnn_state_size]) + self.te_b_first_event[l], axis=1), 
                                                            self.num_operator, 
                                                            axis=1) 
                                                            for rnn_output in self.rnn_outputs] for l in range(self.num_step-1)]


            self.te_attention_operators_last_event = [[tf.split(
                                                        tf.nn.softmax(
                                                          tf.matmul(rnn_output[:, cellLen_single_step*l + state_start_TR + self.rnn_state_size*3: cellLen_single_step*l + state_start_TR + self.rnn_state_size*4], 
                                                                        self.te_W_last_event[self.rnn_state_size*l :self.rnn_state_size*l + self.rnn_state_size]) + self.te_b_last_event[l], axis=1), 
                                                            self.num_operator, 
                                                            axis=1) 
                                                            for rnn_output in self.rnn_outputs] for l in range(self.num_step-1)]


            self.te_attention_operators_TR_first_event = [[tf.split(
                                                          tf.nn.softmax(
                                                          tf.matmul(rnn_output[:, cellLen_single_step*l + state_start_TR*2 + self.rnn_state_size*2: cellLen_single_step*l + state_start_TR*2 + self.rnn_state_size*3], 
                                                                     self.te_W_TR_first_event[self.rnn_state_size*l :self.rnn_state_size*l + self.rnn_state_size]) + self.te_b_TR_first_event[l], axis=1), 
                                                            self.num_TR, 
                                                            axis=1) 
                                                            for rnn_output in self.rnn_outputs] for l in range(self.num_step-1)]     # todo: check rnn_output[:, :self.rnn_state_size]


            self.te_attention_operators_TR_last_event = [[tf.split(
                                                          tf.nn.softmax(
                                                          tf.matmul(rnn_output[:, cellLen_single_step*l + state_start_TR*2 + self.rnn_state_size*3: cellLen_single_step*l + state_start_TR*2 + self.rnn_state_size*4],
                                                                        self.te_W_TR_last_event[self.rnn_state_size*l :self.rnn_state_size*l + self.rnn_state_size]) + self.te_b_TR_last_event[l], axis=1), 
                                                            self.num_TR, 
                                                            axis=1) 
                                                            for rnn_output in self.rnn_outputs] for l in range(self.num_step-1)]     # todo: check rnn_output[:, self.rnn_state_size:self.rnn_state_size*2]



        self.ts_memories_last_event = [tf.one_hot(indices=self.heads, depth=self.num_entity) for l in range(self.num_step-1)]
        self.ts_memories_first_event = [tf.one_hot(indices=self.tails, depth=self.num_entity) for l in range(self.num_step-1)]


        if self.flag_interval:
            self.te_memories_last_event = [tf.one_hot(indices=self.heads, depth=self.num_entity) for l in range(self.num_step-1)]
            self.te_memories_first_event = [tf.one_hot(indices=self.tails, depth=self.num_entity) for l in range(self.num_step-1)]


        self.ts_attn_refType_first_event_embedding = [tf.Variable(self._random_uniform_unit(self.num_query, 2), dtype=tf.float32) for l in range(self.num_step-1)]
        self.ts_attn_refType_last_event_embedding = [tf.Variable(self._random_uniform_unit(self.num_query, 2), dtype=tf.float32) for l in range(self.num_step-1)]
        self.ts_attn_refType_embedding = [tf.Variable(self._random_uniform_unit(self.num_query, 2), dtype=tf.float32) for l in range(self.num_step-1)]
        self.ts_attn_step_embedding = tf.Variable(self._random_uniform_unit(self.num_query, self.num_step-1), dtype=tf.float32)


        ts_attention_refType_first_event = [tf.nn.embedding_lookup(self.ts_attn_refType_first_event_embedding[l], self.query_rels) for l in range(self.num_step-1)]
        ts_attention_refType_first_event = [tf.nn.softmax(ts_attention_refType_first_event[l], axis=1) for l in range(self.num_step-1)]

        ts_attention_refType_last_event = [tf.nn.embedding_lookup(self.ts_attn_refType_last_event_embedding[l], self.query_rels) for l in range(self.num_step-1)]
        ts_attention_refType_last_event = [tf.nn.softmax(ts_attention_refType_last_event[l], axis=1) for l in range(self.num_step-1)]

        ts_attention_refType = [tf.nn.embedding_lookup(self.ts_attn_refType_embedding[l], self.query_rels) for l in range(self.num_step-1)]
        ts_attention_refType = [tf.nn.softmax(ts_attention_refType[l], axis=1) for l in range(self.num_step-1)]

        ts_attention_step = tf.nn.embedding_lookup(self.ts_attn_step_embedding, self.query_rels)
        ts_attention_step = tf.nn.softmax(ts_attention_step, axis=1)


        if self.flag_interval:
            self.te_attn_refType_first_event_embedding = [tf.Variable(self._random_uniform_unit(self.num_query, 2), dtype=tf.float32) for l in range(self.num_step-1)]
            self.te_attn_refType_last_event_embedding = [tf.Variable(self._random_uniform_unit(self.num_query, 2), dtype=tf.float32) for l in range(self.num_step-1)]
            self.te_attn_refType_embedding = [tf.Variable(self._random_uniform_unit(self.num_query, 2), dtype=tf.float32) for l in range(self.num_step-1)]
            self.te_attn_step_embedding = tf.Variable(self._random_uniform_unit(self.num_query, self.num_step-1), dtype=tf.float32)


            te_attention_refType_first_event = [tf.nn.embedding_lookup(self.te_attn_refType_first_event_embedding[l], self.query_rels) for l in range(self.num_step-1)]
            te_attention_refType_first_event = [tf.nn.softmax(te_attention_refType_first_event[l], axis=1) for l in range(self.num_step-1)]

            te_attention_refType_last_event = [tf.nn.embedding_lookup(self.te_attn_refType_last_event_embedding[l], self.query_rels) for l in range(self.num_step-1)]
            te_attention_refType_last_event = [tf.nn.softmax(te_attention_refType_last_event[l], axis=1) for l in range(self.num_step-1)]

            te_attention_refType = [tf.nn.embedding_lookup(self.te_attn_refType_embedding[l], self.query_rels) for l in range(self.num_step-1)]
            te_attention_refType = [tf.nn.softmax(te_attention_refType[l], axis=1) for l in range(self.num_step-1)]

            te_attention_step = tf.nn.embedding_lookup(self.te_attn_step_embedding, self.query_rels)
            te_attention_step = tf.nn.softmax(te_attention_step, axis=1)


        self.database = {r: tf.sparse_placeholder(
                            dtype=tf.float32, 
                            name="database_%d" % r)
                            for r in xrange(self.num_operator)}


        self.connectivity = {TR: tf.sparse_placeholder(
                            dtype=tf.float32,
                            name="connectivity_%d" % TR)
                            for TR in xrange(self.num_TR)}



        self.ts_predictions_last_event = []
        self.ts_predictions_first_event = []
        self.state_vector_ts_last_event = []
        self.state_vector_ts_first_event = []
        self.ts_predictions = 0

        # define forward process
        for l in range(self.num_step-1):
            # use last event to predict
            for t in xrange(l+2):
                # memory_read: tensor of size (batch_size, num_entity)
                memory_read = tf.identity(self.ts_memories_last_event[l])

                if t == 0:
                    memory_read = tf.transpose(memory_read)
                    memory_read = tf.sparse_tensor_dense_matmul(self.connectivity[0], memory_read)   # todo: define connectivity matrixs {0: ukn, 1: bf, 2: t, 3: af} for edges

                    database_results = []
                    for r in xrange(self.num_operator):  # now database is a diagonal matrix which shows the relation of matrix
                        product = tf.sparse_tensor_dense_matmul(self.database[r], memory_read)  # todo: check the shape of memory_read, whether it is the same
                        database_results.append(tf.transpose(product) * self.ts_attention_operators_last_event[l][t][r])

                    # self.x_tmp = database_results
                    added_database_results = tf.add_n(database_results)
                    if self.norm:
                        added_database_results /= tf.maximum(self.thr, tf.reduce_sum(added_database_results, axis=1, keep_dims=True))
                    
                    if self.dropout > 0.:
                      added_database_results = tf.nn.dropout(added_database_results, keep_prob=1.-self.dropout)

                    # Populate a new cell in memory by concatenating.
                    self.ts_memories_last_event[l] = tf.identity(added_database_results)


                elif t < l + 1:
                    # database_results: (will be) a list of num_operator tensors,
                    # each of size (batch_size, num_entity).
                    memory_read = tf.transpose(memory_read)

                    database_results = []
                    for TR in xrange(self.num_TR):   # self.num_TR = 4
                        product = tf.sparse_tensor_dense_matmul(self.connectivity[TR], memory_read)
                        database_results.append(tf.transpose(product) * self.ts_attention_operators_TR_last_event[l][t][TR])

                    added_database_results = tf.add_n(database_results)
                    if self.norm:
                        added_database_results /= tf.maximum(self.thr, tf.reduce_sum(added_database_results, axis=1, keep_dims=True))

                    memory_read = tf.identity(added_database_results)
                    memory_read = tf.transpose(memory_read)

                    database_results = []
                    for r in xrange(self.num_operator):  # now database is a diagonal matrix which shows the relation of matrix
                        product = tf.sparse_tensor_dense_matmul(self.database[r], memory_read)  # todo: check the shape of memory_read, whether it is the same
                        database_results.append(tf.transpose(product) * self.ts_attention_operators_last_event[l][t][r])

                    added_database_results = tf.add_n(database_results)
                    if self.norm:
                        added_database_results /= tf.maximum(self.thr, tf.reduce_sum(added_database_results, axis=1, keep_dims=True))

                    if self.dropout > 0.:
                      added_database_results = tf.nn.dropout(added_database_results, keep_prob=1.-self.dropout)

                    # Populate a new cell in memory by concatenating.  
                    self.ts_memories_last_event[l] = tf.identity(added_database_results)
                else:
                    self.state_vector_ts_last_event.append(tf.identity(memory_read))
                    memory_read = memory_read * (ts_attention_refType_last_event[l][:, 0:1] * self.ts_probs_last_event_ts[:,:,l] + \
                                                 ts_attention_refType_last_event[l][:, 1:2] * self.ts_probs_last_event_te[:,:,l])
                    memory_read = tf.transpose(memory_read)
                    memory_read = tf.sparse_tensor_dense_matmul(self.connectivity[0], memory_read)
                    memory_read = tf.transpose(memory_read)
                    self.ts_predictions_last_event.append(tf.reduce_sum(self.targets_h * memory_read, axis=1, keep_dims=True))

            ###############################################



            # use first event to predict
            for t in xrange(l+2):
                # memory_read: tensor of size (batch_size, num_entity)
                memory_read = tf.identity(self.ts_memories_first_event[l])

                if t == 0:
                    memory_read = tf.transpose(memory_read)
                    memory_read = tf.sparse_tensor_dense_matmul(self.connectivity[0], memory_read)   # todo: define connectivity matrixs {0: ukn, 1: bf, 2: t, 3: af} for edges

                    database_results = []
                    for r in xrange(self.num_operator):  # now database is a diagonal matrix which shows the relation of matrix
                        product = tf.sparse_tensor_dense_matmul(self.database[r], memory_read)  # todo: check the shape of memory_read, whether it is the same
                        database_results.append(tf.transpose(product) * self.ts_attention_operators_first_event[l][t][r])

                    added_database_results = tf.add_n(database_results)
                    if self.norm:
                        added_database_results /= tf.maximum(self.thr, tf.reduce_sum(added_database_results, axis=1, keep_dims=True))
                    
                    if self.dropout > 0.:
                      added_database_results = tf.nn.dropout(added_database_results, keep_prob=1.-self.dropout)

                    # Populate a new cell in memory by concatenating.  
                    self.ts_memories_first_event[l] = tf.identity(added_database_results)


                elif t < l + 1:
                    # database_results: (will be) a list of num_operator tensors,
                    # each of size (batch_size, num_entity).
                    memory_read = tf.transpose(memory_read)

                    database_results = []
                    for TR in xrange(self.num_TR):   # self.num_TR = 4
                        product = tf.sparse_tensor_dense_matmul(self.connectivity[TR], memory_read)
                        database_results.append(tf.transpose(product) * self.ts_attention_operators_TR_first_event[l][t][TR])

                    added_database_results = tf.add_n(database_results)
                    if self.norm:
                        added_database_results /= tf.maximum(self.thr, tf.reduce_sum(added_database_results, axis=1, keep_dims=True))

                    memory_read = tf.identity(added_database_results)
                    memory_read = tf.transpose(memory_read)

                    database_results = []
                    for r in xrange(self.num_operator):  # now database is a diagonal matrix which shows the relation of matrix
                        product = tf.sparse_tensor_dense_matmul(self.database[r], memory_read)  # todo: check the shape of memory_read, whether it is the same
                        database_results.append(tf.transpose(product) * self.ts_attention_operators_first_event[l][t][r])

                    added_database_results = tf.add_n(database_results)
                    if self.norm:
                        added_database_results /= tf.maximum(self.thr, tf.reduce_sum(added_database_results, axis=1, keep_dims=True))

                    if self.dropout > 0.:
                      added_database_results = tf.nn.dropout(added_database_results, keep_prob=1.-self.dropout)

                    # Populate a new cell in memory by concatenating.  
                    self.ts_memories_first_event[l] = tf.identity(added_database_results)

                else:
                    self.state_vector_ts_first_event.append(tf.identity(memory_read))
                    memory_read = memory_read * (ts_attention_refType_first_event[l][0, 0:1] * self.ts_probs_first_event_ts[:,:,l] + \
                                                 ts_attention_refType_first_event[l][0, 1:2] * self.ts_probs_first_event_te[:,:,l])
                    memory_read = tf.transpose(memory_read)
                    memory_read = tf.sparse_tensor_dense_matmul(self.connectivity[0], memory_read)
                    memory_read = tf.transpose(memory_read)
                    self.ts_predictions_first_event.append(tf.reduce_sum(self.targets_t * memory_read, axis=1, keep_dims=True))

                ###############################################

            self.ts_predictions += ts_attention_step[:, l:l+1] * (ts_attention_refType[l][:, 0:1] * self.ts_predictions_first_event[l] + ts_attention_refType[l][:, 1:2] * self.ts_predictions_last_event[l])



        if not self.flag_interval:
            self.final_loss = - tf.reduce_sum(tf.log(tf.maximum(self.ts_predictions, self.thr)), 1)
        else:
            self.te_predictions_last_event = []
            self.te_predictions_first_event = []
            self.state_vector_te_last_event = []
            self.state_vector_te_first_event = []
            self.te_predictions = 0

            for l in range(self.num_step-1):
                # use last event to predict
                for t in xrange(l+2):
                    # memory_read: tensor of size (batch_size, num_entity)
                    memory_read = tf.identity(self.te_memories_last_event[l])

                    if t == 0:
                        memory_read = tf.transpose(memory_read)
                        memory_read = tf.sparse_tensor_dense_matmul(self.connectivity[0], memory_read)   # todo: define connectivity matrixs {0: ukn, 1: bf, 2: t, 3: af} for edges

                        database_results = []
                        for r in xrange(self.num_operator):  # now database is a diagonal matrix which shows the relation of matrix
                            product = tf.sparse_tensor_dense_matmul(self.database[r], memory_read)  # todo: check the shape of memory_read, whether it is the same
                            database_results.append(tf.transpose(product) * self.te_attention_operators_last_event[l][t][r])

                        added_database_results = tf.add_n(database_results)
                        if self.norm:
                            added_database_results /= tf.maximum(self.thr, tf.reduce_sum(added_database_results, axis=1, keep_dims=True))
                        
                        if self.dropout > 0.:
                          added_database_results = tf.nn.dropout(added_database_results, keep_prob=1.-self.dropout)

                        # Populate a new cell in memory by concatenating.  
                        self.te_memories_last_event[l] = tf.identity(added_database_results)


                    elif t < l + 1:
                        # database_results: (will be) a list of num_operator tensors,
                        # each of size (batch_size, num_entity).
                        memory_read = tf.transpose(memory_read)

                        database_results = []
                        for TR in xrange(self.num_TR):   # self.num_TR = 4
                            product = tf.sparse_tensor_dense_matmul(self.connectivity[TR], memory_read)
                            database_results.append(tf.transpose(product) * self.te_attention_operators_TR_last_event[l][t][TR])

                        added_database_results = tf.add_n(database_results)
                        if self.norm:
                            added_database_results /= tf.maximum(self.thr, tf.reduce_sum(added_database_results, axis=1, keep_dims=True))

                        memory_read = tf.identity(added_database_results)
                        memory_read = tf.transpose(memory_read)

                        database_results = []
                        for r in xrange(self.num_operator):  # now database is a diagonal matrix which shows the relation of matrix
                            product = tf.sparse_tensor_dense_matmul(self.database[r], memory_read)  # todo: check the shape of memory_read, whether it is the same
                            database_results.append(tf.transpose(product) * self.te_attention_operators_last_event[l][t][r])

                        added_database_results = tf.add_n(database_results)
                        if self.norm:
                            added_database_results /= tf.maximum(self.thr, tf.reduce_sum(added_database_results, axis=1, keep_dims=True))

                        if self.dropout > 0.:
                          added_database_results = tf.nn.dropout(added_database_results, keep_prob=1.-self.dropout)

                        # Populate a new cell in memory by concatenating.  
                        self.te_memories_last_event[l] = tf.identity(added_database_results)

                    else:
                        self.state_vector_te_last_event.append(tf.identity(memory_read))
                        memory_read = memory_read * (te_attention_refType_last_event[l][:, 0:1] * self.te_probs_last_event_ts[:,:,l] + \
                                                     te_attention_refType_last_event[l][:, 1:2] * self.te_probs_last_event_te[:,:,l])
                        memory_read = tf.transpose(memory_read)
                        memory_read = tf.sparse_tensor_dense_matmul(self.connectivity[0], memory_read)
                        memory_read = tf.transpose(memory_read)
                        self.te_predictions_last_event.append(tf.reduce_sum(self.targets_h * memory_read, axis=1, keep_dims=True))

                ###############################################


                # use first event to predict
                for t in xrange(l+2):
                    # memory_read: tensor of size (batch_size, num_entity)
                    memory_read = tf.identity(self.te_memories_first_event[l])

                    if t == 0:
                        memory_read = tf.transpose(memory_read)
                        memory_read = tf.sparse_tensor_dense_matmul(self.connectivity[0], memory_read)   # todo: define connectivity matrixs {0: ukn, 1: bf, 2: t, 3: af} for edges

                        database_results = []
                        for r in xrange(self.num_operator):  # now database is a diagonal matrix which shows the relation of matrix
                            product = tf.sparse_tensor_dense_matmul(self.database[r], memory_read)  # todo: check the shape of memory_read, whether it is the same
                            database_results.append(tf.transpose(product) * self.te_attention_operators_first_event[l][t][r])

                        added_database_results = tf.add_n(database_results)
                        if self.norm:
                            added_database_results /= tf.maximum(self.thr, tf.reduce_sum(added_database_results, axis=1, keep_dims=True))
                        
                        if self.dropout > 0.:
                          added_database_results = tf.nn.dropout(added_database_results, keep_prob=1.-self.dropout)

                        # Populate a new cell in memory by concatenating.  
                        self.te_memories_first_event[l] = tf.identity(added_database_results)

                    elif t < l + 1:
                        # database_results: (will be) a list of num_operator tensors,
                        # each of size (batch_size, num_entity).
                        memory_read = tf.transpose(memory_read)

                        database_results = []
                        for TR in xrange(self.num_TR):   # self.num_TR = 4
                            product = tf.sparse_tensor_dense_matmul(self.connectivity[TR], memory_read)
                            database_results.append(tf.transpose(product) * self.te_attention_operators_TR_first_event[l][t][TR])

                        added_database_results = tf.add_n(database_results)
                        if self.norm:
                            added_database_results /= tf.maximum(self.thr, tf.reduce_sum(added_database_results, axis=1, keep_dims=True))

                        memory_read = tf.identity(added_database_results)
                        memory_read = tf.transpose(memory_read)

                        database_results = []
                        for r in xrange(self.num_operator):  # now database is a diagonal matrix which shows the relation of matrix
                            product = tf.sparse_tensor_dense_matmul(self.database[r], memory_read)  # todo: check the shape of memory_read, whether it is the same
                            database_results.append(tf.transpose(product) * self.te_attention_operators_first_event[l][t][r])

                        added_database_results = tf.add_n(database_results)
                        if self.norm:
                            added_database_results /= tf.maximum(self.thr, tf.reduce_sum(added_database_results, axis=1, keep_dims=True))

                        if self.dropout > 0.:
                          added_database_results = tf.nn.dropout(added_database_results, keep_prob=1.-self.dropout)

                        # Populate a new cell in memory by concatenating.  
                        self.te_memories_first_event[l] = tf.identity(added_database_results)

                    else:
                        self.state_vector_te_first_event.append(tf.identity(memory_read))
                        memory_read = memory_read * (te_attention_refType_first_event[l][:, 0:1] * self.te_probs_first_event_ts[:,:,l] + \
                                                     te_attention_refType_first_event[l][:, 1:2] * self.te_probs_first_event_te[:,:,l])
                        memory_read = tf.transpose(memory_read)
                        memory_read = tf.sparse_tensor_dense_matmul(self.connectivity[0], memory_read)
                        memory_read = tf.transpose(memory_read)
                        self.te_predictions_first_event.append(tf.reduce_sum(self.targets_t * memory_read, axis=1, keep_dims=True))

                ###############################################

                self.te_predictions += te_attention_step[:, l:l+1] * (te_attention_refType[l][:, 0:1] * self.te_predictions_first_event[l] + te_attention_refType[l][0, 1:2] * self.te_predictions_last_event[l])

            self.final_loss = - tf.reduce_sum(tf.log(tf.maximum(self.ts_predictions, self.thr)), 1) - tf.reduce_sum(tf.log(tf.maximum(self.te_predictions, self.thr)), 1)


        self.optimizer = tf.train.AdamOptimizer()
        gvs = self.optimizer.compute_gradients(tf.reduce_mean(self.final_loss))
        capped_gvs = map(lambda (grad, var): self._clip_if_not_None(grad, var, -10., 10.), gvs) 
        self.optimizer_step = self.optimizer.apply_gradients(capped_gvs)



    # def _build_graph_rule_split_ver(self):
    #     # input rule output res state vector


    #     # define forward process
    #     rule 
    #     # use last event to predict
    #     for t in xrange(l+2):
    #         # memory_read: tensor of size (batch_size, num_entity)
    #         memory_read = tf.identity(self.ts_memories_last_event[l])

    #         if t == 0:
    #             memory_read = tf.transpose(memory_read)
    #             memory_read = tf.sparse_tensor_dense_matmul(self.connectivity[0], memory_read)

    #             r = rule_rel_ls[t]  # todo
    #             product = tf.sparse_tensor_dense_matmul(self.database[r], memory_read)
    #             added_database_results = tf.transpose(product) * self.ts_attention_operators_last_event[l][t][r]

    #             if self.norm:
    #                 added_database_results /= tf.maximum(self.thr, tf.reduce_sum(added_database_results, axis=1, keep_dims=True))

    #             if self.dropout > 0.:
    #               added_database_results = tf.nn.dropout(added_database_results, keep_prob=1.-self.dropout)

    #             self.ts_memories_last_event[l] = tf.identity(added_database_results)

    #         elif t < l + 1:
    #             memory_read = tf.transpose(memory_read)

    #             TR = rule_TR_ls[t]   # todo

    #             product = tf.sparse_tensor_dense_matmul(self.connectivity[TR], memory_read)
    #             added_database_results = tf.transpose(product) * self.ts_attention_operators_TR_last_event[l][t][TR]

    #             if self.norm:
    #                 added_database_results /= tf.maximum(self.thr, tf.reduce_sum(added_database_results, axis=1, keep_dims=True))

    #             memory_read = tf.identity(added_database_results)
    #             memory_read = tf.transpose(memory_read)

    #             r = rule_rel_ls[t]  # todo
    #             product = tf.sparse_tensor_dense_matmul(self.database[r], memory_read)
    #             added_database_results = tf.transpose(product) * self.ts_attention_operators_last_event[l][t][r]

    #             if self.norm:
    #                 added_database_results /= tf.maximum(self.thr, tf.reduce_sum(added_database_results, axis=1, keep_dims=True))

    #             if self.dropout > 0.:
    #               added_database_results = tf.nn.dropout(added_database_results, keep_prob=1.-self.dropout)

    #             # Populate a new cell in memory by concatenating.  
    #             self.ts_memories_last_event[l] = tf.identity(added_database_results)
    #         else:
    #             self.state_vector_ts_last_event.append(tf.identity(memory_read))

    #             memory_read = memory_read * (ts_attention_refType_last_event[l][:, 0:1] * self.ts_probs_last_event_ts[:,:,l] + \
    #                                          ts_attention_refType_last_event[l][:, 1:2] * self.ts_probs_last_event_te[:,:,l])
    #             memory_read = tf.transpose(memory_read)
    #             memory_read = tf.sparse_tensor_dense_matmul(self.connectivity[0], memory_read)
    #             memory_read = tf.transpose(memory_read)

    #             self.ts_predictions_last_event.append(tf.reduce_sum(self.targets_h * memory_read, axis=1, keep_dims=True))



    def _build_graph_acc_ver_old(self):
        self.query_rels = tf.placeholder(tf.int32, [None])   # dummy
        self.refNode_source = tf.placeholder(tf.float32, [None, None])  # show where the refNode comes from
        self.res_random_walk = tf.placeholder(tf.float32, [None, self.num_rule])  # dummy

        if self.flag_ruleLen_split_ver:
            self.ts_probs_last_event_ts = tf.placeholder(tf.float32, [None, self.num_step-1, None])  # dummy
            self.ts_probs_last_event_te = tf.placeholder(tf.float32, [None, self.num_step-1, None])
            self.ts_probs_first_event_ts = tf.placeholder(tf.float32, [None, self.num_step-1, None])
            self.ts_probs_first_event_te = tf.placeholder(tf.float32, [None, self.num_step-1, None])
        else:
            self.ts_probs_last_event_ts = tf.placeholder(tf.float32, [None, None])  # dummy
            self.ts_probs_last_event_te = tf.placeholder(tf.float32, [None, None])
            self.ts_probs_first_event_ts = tf.placeholder(tf.float32, [None, None])
            self.ts_probs_first_event_te = tf.placeholder(tf.float32, [None, None])

        if self.flag_interval:
            if self.flag_ruleLen_split_ver:
                self.te_probs_last_event_ts = tf.placeholder(tf.float32, [None, self.num_step-1, None])
                self.te_probs_last_event_te = tf.placeholder(tf.float32, [None, self.num_step-1, None])
                self.te_probs_first_event_ts = tf.placeholder(tf.float32, [None, self.num_step-1, None])
                self.te_probs_first_event_te = tf.placeholder(tf.float32, [None, self.num_step-1, None])
            else:
                self.te_probs_last_event_ts = tf.placeholder(tf.float32, [None, None])
                self.te_probs_last_event_te = tf.placeholder(tf.float32, [None, None])
                self.te_probs_first_event_ts = tf.placeholder(tf.float32, [None, None])
                self.te_probs_first_event_te = tf.placeholder(tf.float32, [None, None])


        self.attn_rule_embedding = tf.Variable(np.random.randn(self.num_query, self.num_rule),  dtype=tf.float32)

        attn_rule = tf.nn.embedding_lookup(self.attn_rule_embedding, self.query_rels)
        attn_rule = tf.nn.softmax(attn_rule, axis=1)


        self.ts_attn_refType_first_event_embedding = tf.Variable(self._random_uniform_unit(self.num_query, 2), dtype=tf.float32)
        self.ts_attn_refType_last_event_embedding = tf.Variable(self._random_uniform_unit(self.num_query, 2), dtype=tf.float32)
        self.ts_attn_refType_embedding = tf.Variable(self._random_uniform_unit(self.num_query, 2), dtype=tf.float32)

        ts_attention_refType_first_event = tf.nn.embedding_lookup(self.ts_attn_refType_first_event_embedding, self.query_rels)
        ts_attention_refType_first_event = tf.nn.softmax(ts_attention_refType_first_event, axis=1)

        ts_attention_refType_last_event = tf.nn.embedding_lookup(self.ts_attn_refType_last_event_embedding, self.query_rels)
        ts_attention_refType_last_event = tf.nn.softmax(ts_attention_refType_last_event, axis=1)

        ts_attention_refType = tf.nn.embedding_lookup(self.ts_attn_refType_embedding, self.query_rels)
        ts_attention_refType = tf.nn.softmax(ts_attention_refType, axis=1)


        if self.flag_interval:
            self.te_attn_refType_first_event_embedding = tf.Variable(self._random_uniform_unit(self.num_query, 2), dtype=tf.float32)
            self.te_attn_refType_last_event_embedding = tf.Variable(self._random_uniform_unit(self.num_query, 2), dtype=tf.float32)
            self.te_attn_refType_embedding = tf.Variable(self._random_uniform_unit(self.num_query, 2), dtype=tf.float32)

            te_attention_refType_first_event = tf.nn.embedding_lookup(self.te_attn_refType_first_event_embedding, self.query_rels)
            te_attention_refType_first_event = tf.nn.softmax(te_attention_refType_first_event, axis=1)

            te_attention_refType_last_event = tf.nn.embedding_lookup(self.te_attn_refType_last_event_embedding, self.query_rels)
            te_attention_refType_last_event = tf.nn.softmax(te_attention_refType_last_event, axis=1)

            te_attention_refType = tf.nn.embedding_lookup(self.te_attn_refType_embedding, self.query_rels)
            te_attention_refType = tf.nn.softmax(te_attention_refType, axis=1)


        if self.flag_ruleLen_split_ver:
            refNode_probs_mat = self.res_random_walk * attn_rule
            ruleLen_table = [tf.nn.embedding_lookup(self.ruleLen_embedding[l], self.query_rels) for l in range(self.num_step-1)]
            refNode_probs = [tf.reduce_sum(refNode_probs_mat * ruleLen_table[l], axis=1, keep_dims=True) for l in range(self.num_step-1)]

            ts_probs_from_refNode = [ts_attention_refType[:, 0:1] * (ts_attention_refType_first_event[:, 0:1] * self.ts_probs_first_event_ts[:,l,:] + ts_attention_refType_first_event[:, 1:2] * self.ts_probs_first_event_te[:,l,:]) + \
                                     ts_attention_refType[:, 1:2] * (ts_attention_refType_last_event[:, 0:1] * self.ts_probs_last_event_ts[:,l,:] + ts_attention_refType_last_event[:, 1:2] * self.ts_probs_last_event_te[:,l,:]) \
                                     for l in range(self.num_step-1)]

            ts_pred_refNode = 0
            for l in range(self.num_step-1):
                ts_pred_refNode += refNode_probs[l] * ts_probs_from_refNode[l]

            self.ts_predictions = tf.matmul(self.refNode_source, ts_pred_refNode)

        else:
            refNode_probs = tf.reduce_sum(self.res_random_walk * attn_rule, axis=1, keep_dims=True)

            ts_probs_from_refNode = ts_attention_refType[:, 0:1] * (ts_attention_refType_first_event[:, 0:1] * self.ts_probs_first_event_ts + ts_attention_refType_first_event[:, 1:2] * self.ts_probs_first_event_te) + \
                                    ts_attention_refType[:, 1:2] * (ts_attention_refType_last_event[:, 0:1] * self.ts_probs_last_event_ts + ts_attention_refType_last_event[:, 1:2] * self.ts_probs_last_event_te)

            self.ts_predictions = tf.matmul(self.refNode_source, refNode_probs * ts_probs_from_refNode)


        if self.flag_interval:
            if self.flag_ruleLen_split_ver:
                te_probs_from_refNode = [te_attention_refType[:, 0:1] * (te_attention_refType_first_event[:, 0:1] * self.te_probs_first_event_ts[:,l,:] + te_attention_refType_first_event[:, 1:2] * self.te_probs_first_event_te[:,l,:]) + \
                                         te_attention_refType[:, 1:2] * (te_attention_refType_last_event[:, 0:1] * self.te_probs_last_event_ts[:,l,:] + te_attention_refType_last_event[:, 1:2] * self.te_probs_last_event_te[:,l,:]) \
                                         for l in range(self.num_step-1)]

                te_pred_refNode = 0
                for l in range(self.num_step-1):
                    te_pred_refNode += refNode_probs[l] * te_probs_from_refNode[l]

                self.te_predictions = tf.matmul(self.refNode_source, te_pred_refNode)

            else:
                te_probs_from_refNode = te_attention_refType[:, 0:1] * (te_attention_refType_first_event[:, 0:1] * self.te_probs_first_event_ts + te_attention_refType_first_event[:, 1:2] * self.te_probs_first_event_te) + \
                                        te_attention_refType[:, 1:2] * (te_attention_refType_last_event[:, 0:1] * self.te_probs_last_event_ts + te_attention_refType_last_event[:, 1:2] * self.te_probs_last_event_te)

                self.te_predictions = tf.matmul(self.refNode_source, refNode_probs * te_probs_from_refNode)


        self.final_loss = -tf.reduce_sum(tf.log(tf.maximum(self.ts_predictions, self.thr)), 1)
        if self.flag_interval:
            self.final_loss += -tf.reduce_sum(tf.log(tf.maximum(self.te_predictions, self.thr)), 1)


        self.optimizer = tf.train.AdamOptimizer()
        gvs = self.optimizer.compute_gradients(tf.reduce_mean(self.final_loss))
        capped_gvs = map(
            lambda (grad, var): self._clip_if_not_None(grad, var, -10., 10.), gvs)
        self.optimizer_step = self.optimizer.apply_gradients(capped_gvs)




    def _build_graph_acc_ver(self):
        self.query_rels = tf.placeholder(tf.int32, [None])   # dummy
        self.refNode_source = tf.placeholder(tf.float32, [None, None])  # show where the refNode comes from
        self.res_random_walk = tf.placeholder(tf.float32, [None, self.num_rule])  # dummy

        if self.flag_ruleLen_split_ver:
            self.ts_probs_last_event_ts = tf.placeholder(tf.float32, [None, self.num_step-1, None])  # dummy
            self.ts_probs_last_event_te = tf.placeholder(tf.float32, [None, self.num_step-1, None])
            self.ts_probs_first_event_ts = tf.placeholder(tf.float32, [None, self.num_step-1, None])
            self.ts_probs_first_event_te = tf.placeholder(tf.float32, [None, self.num_step-1, None])
        else:
            self.ts_probs_last_event_ts = tf.placeholder(tf.float32, [None, None])  # dummy
            self.ts_probs_last_event_te = tf.placeholder(tf.float32, [None, None])
            self.ts_probs_first_event_ts = tf.placeholder(tf.float32, [None, None])
            self.ts_probs_first_event_te = tf.placeholder(tf.float32, [None, None])

        if self.flag_interval:
            if self.flag_ruleLen_split_ver:
                self.te_probs_last_event_ts = tf.placeholder(tf.float32, [None, self.num_step-1, None])
                self.te_probs_last_event_te = tf.placeholder(tf.float32, [None, self.num_step-1, None])
                self.te_probs_first_event_ts = tf.placeholder(tf.float32, [None, self.num_step-1, None])
                self.te_probs_first_event_te = tf.placeholder(tf.float32, [None, self.num_step-1, None])
            else:
                self.te_probs_last_event_ts = tf.placeholder(tf.float32, [None, None])
                self.te_probs_last_event_te = tf.placeholder(tf.float32, [None, None])
                self.te_probs_first_event_ts = tf.placeholder(tf.float32, [None, None])
                self.te_probs_first_event_te = tf.placeholder(tf.float32, [None, None])


        self.attn_rule_embedding = tf.Variable(np.random.randn(self.num_query, self.num_rule),  dtype=tf.float32)

        attn_rule = tf.nn.embedding_lookup(self.attn_rule_embedding, self.query_rels)
        attn_rule = tf.nn.softmax(attn_rule, axis=1)

        self.attn_rule = attn_rule

        self.ts_attn_refType_first_event_embedding = tf.Variable(self._random_uniform_unit(self.num_query, 2), dtype=tf.float32)
        self.ts_attn_refType_last_event_embedding = tf.Variable(self._random_uniform_unit(self.num_query, 2), dtype=tf.float32)
        self.ts_attn_refType_embedding = tf.Variable(self._random_uniform_unit(self.num_query, 2), dtype=tf.float32)

        ts_attention_refType_first_event = tf.nn.embedding_lookup(self.ts_attn_refType_first_event_embedding, self.query_rels)
        ts_attention_refType_first_event = tf.nn.softmax(ts_attention_refType_first_event, axis=1)

        ts_attention_refType_last_event = tf.nn.embedding_lookup(self.ts_attn_refType_last_event_embedding, self.query_rels)
        ts_attention_refType_last_event = tf.nn.softmax(ts_attention_refType_last_event, axis=1)

        ts_attention_refType = tf.nn.embedding_lookup(self.ts_attn_refType_embedding, self.query_rels)
        ts_attention_refType = tf.nn.softmax(ts_attention_refType, axis=1)


        if self.flag_interval:
            self.te_attn_refType_first_event_embedding = tf.Variable(self._random_uniform_unit(self.num_query, 2), dtype=tf.float32)
            self.te_attn_refType_last_event_embedding = tf.Variable(self._random_uniform_unit(self.num_query, 2), dtype=tf.float32)
            self.te_attn_refType_embedding = tf.Variable(self._random_uniform_unit(self.num_query, 2), dtype=tf.float32)

            te_attention_refType_first_event = tf.nn.embedding_lookup(self.te_attn_refType_first_event_embedding, self.query_rels)
            te_attention_refType_first_event = tf.nn.softmax(te_attention_refType_first_event, axis=1)

            te_attention_refType_last_event = tf.nn.embedding_lookup(self.te_attn_refType_last_event_embedding, self.query_rels)
            te_attention_refType_last_event = tf.nn.softmax(te_attention_refType_last_event, axis=1)

            te_attention_refType = tf.nn.embedding_lookup(self.te_attn_refType_embedding, self.query_rels)
            te_attention_refType = tf.nn.softmax(te_attention_refType, axis=1)


        if self.flag_ruleLen_split_ver:
            refNode_probs_mat = self.res_random_walk * attn_rule
            ruleLen_table = [tf.nn.embedding_lookup(self.ruleLen_embedding[l], self.query_rels) for l in range(self.num_step-1)]
            refNode_probs = [tf.reduce_sum(refNode_probs_mat * ruleLen_table[l], axis=1, keep_dims=True) for l in range(self.num_step-1)]

            ts_probs_from_refNode = [ts_attention_refType[:, 0:1] * (ts_attention_refType_first_event[:, 0:1] * self.ts_probs_first_event_ts[:,l,:] + ts_attention_refType_first_event[:, 1:2] * self.ts_probs_first_event_te[:,l,:]) + \
                                     ts_attention_refType[:, 1:2] * (ts_attention_refType_last_event[:, 0:1] * self.ts_probs_last_event_ts[:,l,:] + ts_attention_refType_last_event[:, 1:2] * self.ts_probs_last_event_te[:,l,:]) \
                                     for l in range(self.num_step-1)]

            ts_pred_refNode = 0
            for l in range(self.num_step-1):
                ts_pred_refNode += refNode_probs[l] * ts_probs_from_refNode[l]

            self.ts_predictions = tf.matmul(self.refNode_source, ts_pred_refNode)

        else:
            refNode_probs = tf.reduce_sum(self.res_random_walk * attn_rule, axis=1, keep_dims=True)

            ts_probs_from_refNode = ts_attention_refType[:, 0:1] * (ts_attention_refType_first_event[:, 0:1] * self.ts_probs_first_event_ts + ts_attention_refType_first_event[:, 1:2] * self.ts_probs_first_event_te) + \
                                    ts_attention_refType[:, 1:2] * (ts_attention_refType_last_event[:, 0:1] * self.ts_probs_last_event_ts + ts_attention_refType_last_event[:, 1:2] * self.ts_probs_last_event_te)

            self.ts_predictions = tf.matmul(self.refNode_source, refNode_probs * ts_probs_from_refNode)


        if self.flag_interval:
            if self.flag_ruleLen_split_ver:
                te_probs_from_refNode = [te_attention_refType[:, 0:1] * (te_attention_refType_first_event[:, 0:1] * self.te_probs_first_event_ts[:,l,:] + te_attention_refType_first_event[:, 1:2] * self.te_probs_first_event_te[:,l,:]) + \
                                         te_attention_refType[:, 1:2] * (te_attention_refType_last_event[:, 0:1] * self.te_probs_last_event_ts[:,l,:] + te_attention_refType_last_event[:, 1:2] * self.te_probs_last_event_te[:,l,:]) \
                                         for l in range(self.num_step-1)]

                te_pred_refNode = 0
                for l in range(self.num_step-1):
                    te_pred_refNode += refNode_probs[l] * te_probs_from_refNode[l]

                self.te_predictions = tf.matmul(self.refNode_source, te_pred_refNode)

            else:
                te_probs_from_refNode = te_attention_refType[:, 0:1] * (te_attention_refType_first_event[:, 0:1] * self.te_probs_first_event_ts + te_attention_refType_first_event[:, 1:2] * self.te_probs_first_event_te) + \
                                        te_attention_refType[:, 1:2] * (te_attention_refType_last_event[:, 0:1] * self.te_probs_last_event_ts + te_attention_refType_last_event[:, 1:2] * self.te_probs_last_event_te)

                self.te_predictions = tf.matmul(self.refNode_source, refNode_probs * te_probs_from_refNode)


        self.final_loss = -tf.reduce_sum(tf.log(tf.maximum(self.ts_predictions, self.thr)), 1)
        if self.flag_interval:
            self.final_loss += -tf.reduce_sum(tf.log(tf.maximum(self.te_predictions, self.thr)), 1)


        self.optimizer = tf.train.AdamOptimizer()
        gvs = self.optimizer.compute_gradients(tf.reduce_mean(self.final_loss))
        capped_gvs = map(
            lambda (grad, var): self._clip_if_not_None(grad, var, -10., 10.), gvs)
        self.optimizer_step = self.optimizer.apply_gradients(capped_gvs)






    def _run_graph(self, sess, qq, hh, tt, mdb, connectivity, probs, valid_sample_idx, mode, to_fetch):
        qq = [qq[idx] for idx in valid_sample_idx]
        hh = [hh[idx] for idx in valid_sample_idx]
        tt = [tt[idx] for idx in valid_sample_idx]


        if self.flag_interval:
            ts_probs_first_event, ts_probs_last_event, te_probs_first_event, te_probs_last_event = probs
        else:
            ts_probs_first_event, ts_probs_last_event = probs


        feed = {}

        feed[self.queries] = [[q] * (self.num_step-1) + [self.num_query]
                              for q in qq]
        feed[self.heads] = hh 
        feed[self.tails] = tt 
        feed[self.query_rels] = qq

        feed[self.ts_probs_last_event_ts] = [ts_probs_last_event[0][idx] for idx in valid_sample_idx]
        feed[self.ts_probs_last_event_te] = [ts_probs_last_event[1][idx] for idx in valid_sample_idx]
        feed[self.ts_probs_first_event_ts] = [ts_probs_first_event[0][idx] for idx in valid_sample_idx]
        feed[self.ts_probs_first_event_te] = [ts_probs_first_event[1][idx] for idx in valid_sample_idx]

        if mode == 'Test':
            feed[self.ts_probs_last_event_ts] = [self.create_dummy_probs_for_prediction(ts_probs_last_event[0][idx]) for idx in valid_sample_idx]
            feed[self.ts_probs_last_event_te] = [self.create_dummy_probs_for_prediction(ts_probs_last_event[1][idx]) for idx in valid_sample_idx]
            feed[self.ts_probs_first_event_ts] = [self.create_dummy_probs_for_prediction(ts_probs_first_event[0][idx]) for idx in valid_sample_idx]
            feed[self.ts_probs_first_event_te] = [self.create_dummy_probs_for_prediction(ts_probs_first_event[1][idx]) for idx in valid_sample_idx]



        if self.flag_interval:
            feed[self.te_probs_last_event_ts] = [te_probs_last_event[0][idx] for idx in valid_sample_idx]
            feed[self.te_probs_last_event_te] = [te_probs_last_event[1][idx] for idx in valid_sample_idx]
            feed[self.te_probs_first_event_ts] = [te_probs_first_event[0][idx] for idx in valid_sample_idx]
            feed[self.te_probs_first_event_te] = [te_probs_first_event[1][idx] for idx in valid_sample_idx]

            if mode == 'Test':
                feed[self.te_probs_last_event_ts] = [self.create_dummy_probs_for_prediction(te_probs_last_event[0][idx]) for idx in valid_sample_idx]
                feed[self.te_probs_last_event_te] = [self.create_dummy_probs_for_prediction(te_probs_last_event[1][idx]) for idx in valid_sample_idx]
                feed[self.te_probs_first_event_ts] = [self.create_dummy_probs_for_prediction(te_probs_first_event[0][idx]) for idx in valid_sample_idx]
                feed[self.te_probs_first_event_te] = [self.create_dummy_probs_for_prediction(te_probs_first_event[1][idx]) for idx in valid_sample_idx]


        for r in xrange(self.num_operator):
            feed[self.database[r]] = tf.SparseTensorValue(*mdb[r])
        for TR in xrange(self.num_TR):
            feed[self.connectivity[TR]] = tf.SparseTensorValue(*connectivity[TR])

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




    def _run_graph_acc(self, sess, query_rels, refNode_source, res_random_walk, probs, to_fetch):
        if self.flag_interval:
            ts_probs_first_event_ts, ts_probs_first_event_te, ts_probs_last_event_ts, ts_probs_last_event_te, \
            te_probs_first_event_ts, te_probs_first_event_te, te_probs_last_event_ts, te_probs_last_event_te  = probs
        else:
            ts_probs_first_event_ts, ts_probs_first_event_te, ts_probs_last_event_ts, ts_probs_last_event_te = probs

        feed = {}
        feed[self.query_rels] = query_rels

        feed[self.refNode_source] = refNode_source

        res_random_walk_mat = np.zeros((len(query_rels), self.num_rule))

        res_random_walk_mat[res_random_walk[:, 0], res_random_walk[:, 1]] = 1

        feed[self.res_random_walk] = res_random_walk_mat

        feed[self.ts_probs_last_event_ts] = ts_probs_last_event_ts
        feed[self.ts_probs_last_event_te] = ts_probs_last_event_te
        feed[self.ts_probs_first_event_ts] = ts_probs_first_event_ts
        feed[self.ts_probs_first_event_te] = ts_probs_first_event_te

        if self.flag_interval:
            feed[self.te_probs_last_event_ts] = te_probs_last_event_ts
            feed[self.te_probs_last_event_te] = te_probs_last_event_te
            feed[self.te_probs_first_event_ts] = te_probs_first_event_ts
            feed[self.te_probs_first_event_te] = te_probs_first_event_te


        fetches = to_fetch
        graph_output = sess.run(fetches, feed)
        return graph_output





    def update(self, sess, qq, hh, tt, mdb, connectivity, probs, valid_sample_idx, valid_ref_event_idx):
        to_fetch = [self.final_loss, self.optimizer_step]
        fetched = self._run_graph(sess, qq, hh, tt, mdb, connectivity, probs, valid_sample_idx, 'Train', to_fetch)

        return fetched[0]



    def predict(self, sess, qq, hh, tt, mdb, connectivity, probs, valid_sample_idx, valid_ref_event_idx):
        to_fetch = [self.state_vector_ts_first_event, self.state_vector_ts_last_event, 
                    self.state_vector_te_first_event, self.state_vector_te_last_event,
                    self.ts_attention_refType_first_event, self.ts_attention_refType_last_event, self.ts_attention_refType,
                    self.te_attention_refType_first_event, self.te_attention_refType_last_event, self.te_attention_refType]

        fetched = self._run_graph(sess, qq, hh, tt, mdb, connectivity, probs, valid_sample_idx, 'Test', to_fetch)

        valid_ref_event_idx_ts_first, valid_ref_event_idx_ts_last, valid_ref_event_idx_te_first, \
                                                                valid_ref_event_idx_te_last = valid_ref_event_idx

        state_vector_ts_first = np.expand_dims(fetched[0] * valid_ref_event_idx_ts_first, axis=2)
        state_vector_ts_last = np.expand_dims(fetched[1] * valid_ref_event_idx_ts_last, axis=2)

        state_vector_te_first = np.expand_dims(fetched[2] * valid_ref_event_idx_te_first, axis=2)
        state_vector_te_last = np.expand_dims(fetched[3] * valid_ref_event_idx_te_last, axis=2)

        if self.flag_interval:
            ts_probs_first_event, ts_probs_last_event, te_probs_first_event, te_probs_last_event = probs
        else:
            ts_probs_first_event, ts_probs_last_event = probs

        ts_probs_ts_first_event = np.array([ts_probs_first_event[0][idx] for idx in valid_sample_idx])
        ts_probs_ts_last_event = np.array([ts_probs_last_event[0][idx] for idx in valid_sample_idx])
        ts_probs_te_first_event = np.array([ts_probs_first_event[1][idx] for idx in valid_sample_idx])
        ts_probs_te_last_event = np.array([ts_probs_last_event[1][idx] for idx in valid_sample_idx])

        ts_predictions = fetched[6][:, 0:1] *np.sum((np.expand_dims(fetched[4][:, 0:1], axis=2) * ts_probs_ts_first_event + \
                                                     np.expand_dims(fetched[4][:, 1:2], axis=2) * ts_probs_te_first_event) \
                                                    * state_vector_ts_first, axis = 1)\
                       + fetched[6][:, 1:2] *np.sum((np.expand_dims(fetched[5][:, 0:1], axis=2) * ts_probs_ts_last_event + \
                                                     np.expand_dims(fetched[5][:, 1:2], axis=2) * ts_probs_te_last_event) \
                                                    * state_vector_ts_last, axis = 1)

        te_probs_ts_first_event = np.array([ts_probs_first_event[0][idx] for idx in valid_sample_idx])
        te_probs_ts_last_event = np.array([ts_probs_last_event[0][idx] for idx in valid_sample_idx])
        te_probs_te_first_event = np.array([ts_probs_first_event[1][idx] for idx in valid_sample_idx])
        te_probs_te_last_event = np.array([ts_probs_last_event[1][idx] for idx in valid_sample_idx])

        te_predictions = fetched[9][:, 0:1] *np.sum((np.expand_dims(fetched[7][:, 0:1], axis=2) * te_probs_ts_first_event + \
                                                     np.expand_dims(fetched[7][:, 1:2], axis=2) * te_probs_te_first_event) \
                                                    * state_vector_te_first, axis = 1)\
                       + fetched[9][:, 1:2] *np.sum((np.expand_dims(fetched[8][:, 0:1], axis=2) * te_probs_ts_last_event + \
                                                     np.expand_dims(fetched[8][:, 1:2], axis=2) * te_probs_te_last_event) \
                                                    * state_vector_te_last, axis = 1)

        return ts_predictions, te_predictions


    def update_acc(self, sess, qq, refNode_source, res_random_walk, probs):
        to_fetch = [self.final_loss, self.optimizer_step]
        fetched = self._run_graph_acc(sess, qq, refNode_source, res_random_walk, probs, to_fetch)

        return fetched[0]


    def predict_acc(self, sess, qq, refNode_source, res_random_walk, probs):
        to_fetch = [self.ts_predictions]

        if self.flag_interval:
            to_fetch.append(self.te_predictions)

        fetched = self._run_graph_acc(sess, qq, refNode_source, res_random_walk, probs, to_fetch)

        if self.flag_interval:
            return fetched[0], fetched[1]
        else:
            return [fetched[0]]


    def get_rule_scores_acc(self, sess):
        to_fetch = [self.attn_rule]

        qq = list(range(self.num_query))
        refNode_source = [[0] * (self.num_query)]
        res_random_walk = np.array([[0] * 2] * (self.num_query))
        probs = [[[0]]*self.num_timestamp] * 4
        if self.flag_interval:
            probs += [[[0]]*self.num_timestamp] * 4

        fetched = self._run_graph_acc(sess, qq, refNode_source, res_random_walk, probs, to_fetch)

        return fetched[0]