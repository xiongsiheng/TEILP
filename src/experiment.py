import sys
import os
import time
import numpy as np
from tqdm import tqdm
from utlis import *
from Graph import *


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
        if self.option.create_log:
            self.log_file = open(os.path.join(self.option.this_expsdir, "log.txt"), "w")

        if self.option.flag_interval:
            self.metrics = ['aeIOU', 'TAC']
        else:
            self.metrics = ['MAE']

        self.idx_ls_dict = {"Train": self.data['train_idx_ls'], "Valid": self.data['valid_idx_ls'], "Test": self.data['test_idx_ls']}


    def _calculate_output(self, myTEKG, run_fn, batch_idx_ls, mode, stage, flag_rm_seen_ts, useful_data):
        '''
        Calculate the output of the model for a batch of data.
        There are three modes: 'Train', 'Valid', 'Test'.
        For validation and testing, we have two stages: 'obtain state vec' and 'time prediction'.
        '''
        train_nodes, timestamp_range, pred_dur_dict, final_state_vec, attn_refType = useful_data
        extra_data = [final_state_vec, attn_refType] if stage == 'time prediction' else None
        
        if self.option.flag_acceleration:
            # Variables used by the fast-ver model:
            #     query_rel_flatten: [] * flatten_batch_size (num_nodes);  
            #     refNode_source: [(flatten_batch_size,)] * batch_size;
            #     probs: (num_rules_in_total_for_different_nodes, 3); [event_idx, rule_idx, prob]
            qq, query_rel_flatten, refNode_source, probs, _, valid_sample_idx, \
                            query_time, query_samples, final_preds = myTEKG.graph.create_graph(batch_idx_ls, mode)
        else:
            qq, hh, tt, connectivity_rel, connectivity_TR, probs, valid_sample_idx, valid_refNode_idx, inputs_for_enhancement, query_time,\
                    query_samples, final_preds = myTEKG.graph.create_graph(batch_idx_ls, mode, stage, extra_data)

        gts = np.array(query_time)
        
        # We first random guess the time.
        preds = [[1900, 2000] for _ in range(len(batch_idx_ls))]  if self.option.dataset in ['wiki', 'YAGO'] else \
                [[self._find_middle_point(self.data['timestamp_range'])] for _ in range(len(batch_idx_ls))]
        preds = np.array(preds)
        
        if mode == "Train":
            if len(valid_sample_idx) == 0:
                # all samples are invalid (no walk can be found)
                return [], [], []

            inputs = [query_rel_flatten, refNode_source, probs] if self.option.flag_acceleration else \
                     [qq, hh, tt, connectivity_rel, connectivity_TR, probs, valid_sample_idx, inputs_for_enhancement]        
            output = run_fn(self.sess, inputs)

            # # For debugging only.
            # if np.isnan(output).any():
            #     print("Nan in the output")
            #     print(batch_idx_ls, output)
            #     sys.exit(0)
        else:
            # For inference, there might be different stages.
            if stage == 'obtain state vec':
                if len(valid_sample_idx) == 0:
                    # all samples are invalid (no walk can be found)
                    return {}, [], []
          
                inputs = [qq, hh, tt, connectivity_rel, connectivity_TR, probs, valid_sample_idx, inputs_for_enhancement, valid_refNode_idx, batch_idx_ls]
                output = run_fn(self.sess, inputs)
                output['batch_idx'] = batch_idx_ls
            else:
                useful_data = [valid_sample_idx, query_samples, train_nodes]
                output = final_preds

                # Obtain the predictions from the output.
                # final_preds: [(bacth_size, num_timestamp)] * (1 + int(flag_int))
                for (i, prob_t) in enumerate(output):
                    if len(valid_sample_idx) > 0:
                        prob_t = prob_t.reshape(-1)   # prob_t: (batch_size, num_timestamp)
                        prob_t = np.array(split_list_into_batches(prob_t, batch_size=len(timestamp_range)))

                        preds[valid_sample_idx, i] = self._get_prediction(prob_t, timestamp_range, useful_data, flag_rm_seen_ts)
                    
                preds = self._adjust_preds_based_on_dur(preds, qq, pred_dur_dict, batch_idx_ls)
        
        return output, preds, gts


    def _find_middle_point(self, array):
        # Find the length of the array
        length = len(array)

        # Calculate the middle index
        middle_index = length // 2

        # Get the middle point
        middle_point = array[middle_index]

        return middle_point


    def _get_prediction(self, prob_t, timestamp_range, useful_data, flag_rm_seen_ts):
        '''
        Given the triple (subject, relation, object), remove the seen timestamps in the training set during inference.
        Only used for timestamp-based datasets.
        '''
        valid_sample_idx, query_samples, train_nodes = useful_data
        
        if flag_rm_seen_ts:
            query_samples = np.array(query_samples)
            query_samples = query_samples[valid_sample_idx]
            pred_t = []
            for idx in range(len(prob_t)):
                # For each sample, we remove the seen timestamps in the training set.
                cur_prob_t = prob_t[idx]
                seen_ts = train_nodes[np.all(train_nodes[:, :3] == query_samples[idx, :3], axis=1), 3]
                seen_ts = [timestamp_range.tolist().index(ts) for ts in seen_ts if ts in timestamp_range.tolist()]
                cur_prob_t[seen_ts] = 0
                pred_t.append(timestamp_range[np.argmax(cur_prob_t)])
            pred_t = np.array(pred_t).reshape((-1,))
        else:
            pred_t = timestamp_range[np.argmax(prob_t, axis=1)].reshape((-1,))
        
        return pred_t


    def _adjust_preds_based_on_dur(self, preds, qq, pred_dur_dict, batch_idx_ls):
        '''
        Adjust the predictions based on the duration information.
        Only used for interval-based datasets.
        '''
        if not self.option.flag_interval:
            return preds
        
        if self.option.flag_use_dur:
            pred_ts, pred_te = preds[0], preds[1]
            pred_dur = []
            for data_idx in batch_idx_ls:
                pred_dur1 = pred_dur_dict[str(data_idx)]
                pred_dur.append(abs(pred_dur1[1] - pred_dur1[0]))
            pred_te = pred_ts + np.array(pred_dur).reshape((-1, 1))
            preds = np.hstack([pred_ts, pred_te])
        
        if self.data['rel_ls_no_dur'] is not None:
            # For the relations without duration information, we use the average time as the prediction.
            mask = np.isin(np.array(qq), self.data['rel_ls_no_dur'])

            avg_time = np.mean(preds[mask], axis=1).reshape((-1,1))
            preds[mask] = np.hstack((avg_time, avg_time))

        return preds


    def _load_saved_final_state(self, save_path):
        '''
        Load the saved final state vectors and attention scores.
        '''
        if not os.path.exists(save_path):
            return None, None, None
        
        with open(save_path) as json_file:
            res = json.load(json_file)

        final_state_vec = res['final_state_vec'] if 'final_state_vec' in res else None
        attn_refType = res['attn_refType'] if 'attn_refType' in res else None
        batch_idx_ls = res['batch_idx'] if 'batch_idx' in res else None
        return final_state_vec, attn_refType, batch_idx_ls


    def running_model(self, model, run_fn, idx_ls, mode, stage, flag_rm_seen_ts):
        '''
        Run the mode according to the function and settings.
        '''
        timestamp_range = self.data['timestamp_range']
        train_nodes = self.data['train_nodes']
        pred_dur_dict = self.data['pred_dur'] if self.option.flag_use_dur else None
        
        # Split the index list into batches.       
        idx_ls = split_list_into_batches(idx_ls, batch_size=self.option.batch_size)
        
        # Prepare all batches to find the global batch idx (if needed).
        all_batch_ls = split_list_into_batches(self.idx_ls_dict[mode], batch_size=self.option.batch_size)

        desc = stage if stage is not None else mode

        epoch_loss, epoch_eval_aeIOU, epoch_eval_TAC, epoch_eval_MAE = [], [], [], []
        for batch_idx_ls in tqdm(idx_ls, desc=desc):
            # Prepare useful data for inference
            final_state_vec, attn_refType = None, None
            
            if mode in ['Valid', 'Test'] and not self.option.flag_acceleration:
                # Find the corresponding batch index in the saved file.
                save_path = '../output/{}/final_state_vec/{}_batch_{}.json'.format(self.data['short_name'], mode, all_batch_ls.index(batch_idx_ls))
            
            if stage == 'time prediction':
                # Read the saved final state vectors and attention scores.
                final_state_vec, attn_refType, batch_idx_ls_read = self._load_saved_final_state(save_path)
                batch_idx_ls = batch_idx_ls_read if batch_idx_ls_read is not None else batch_idx_ls

            useful_data = [train_nodes, timestamp_range, pred_dur_dict, final_state_vec, attn_refType]

            output, preds, gts = self._calculate_output(model, run_fn, batch_idx_ls, mode, stage, flag_rm_seen_ts, useful_data)
                     
            if mode == "Train":
                if len(output) == 0:
                    continue
                epoch_loss += output.tolist()
            else: 
                if stage == 'obtain state vec':
                    with open(save_path, 'w') as json_file:
                        json.dump(output, json_file)
                else:
                    if 'aeIOU' in self.metrics:
                        epoch_eval_aeIOU += obtain_aeIoU(preds, gts)
                    if 'TAC' in self.metrics:
                        epoch_eval_TAC += obtain_TAC(preds, gts)
                    if 'MAE' in self.metrics:
                        epoch_eval_MAE += np.abs(np.array(preds).reshape(-1) - gts.reshape(-1)).tolist()

        if mode == "Train":
            if len(epoch_loss) == 0:
                epoch_loss = [100]
            if self.option.create_log:
                msg = self.msg_with_time("Epoch %d mode %s Loss %0.4f " % (self.epoch+1, mode, np.mean(epoch_loss)))
                self.log_file.write(msg + "\n")
            print("Epoch %d mode %s Loss %0.4f " % (self.epoch+1, mode, np.mean(epoch_loss)))
            return epoch_loss
        else:
            if stage == 'obtain state vec':
                return None
            else:
                return epoch_eval_aeIOU, epoch_eval_TAC, epoch_eval_MAE


    def one_epoch(self, total_idx, mode, stage):
        assert mode in ["Train", "Valid", "Test"]
        assert stage in ['obtain state vec', 'time prediction', None]  # We only use stage during inference.

        flag_rm_seen_ts = False # remove seen ts in training set
        if self.data['short_name'] in ['icews14', 'icews05-15', 'gdelt'] and not self.option.shift:
            flag_rm_seen_ts = True

        # Create the model based on the option. 
        myTEKG = TEKG_family(self.option, self.data)

        # To accelerate the inference, we use online algorithm to calculate the output instead of re-running the model.
        # For fast version, we don't need to run the model. We only need rule scores and refType scores.
        # For normal version, we only need to run the model to get and save the state vectors. And then we predict time without running the model.
        if mode == "Train":
            run_fn = self.learner.update
        else:
            run_fn = None
            if stage == 'obtain state vec':
                run_fn = self.learner.get_state_vec
        
        return self.running_model(myTEKG, run_fn, total_idx, mode, stage, flag_rm_seen_ts)


    def one_epoch_train(self, total_idx=None):       
        if total_idx is None:
            # Randomly sample the training data if the option is choosen.
            processor = Data_Processor()
            total_idx = processor._trainig_data_sampling(self.data['train_nodes'], self.data['train_idx_ls'], self.data['num_query'], 
                                                         num_sample_per_rel=self.option.num_samples_per_rel)
            random.shuffle(total_idx)

        loss = self.one_epoch(total_idx, "Train", None)
        self.train_stats.append([loss])


    def one_epoch_valid(self, total_idx=None):
        stage = None if self.option.flag_acceleration else 'time prediction'
        total_idx = self.idx_ls_dict["Valid"] if total_idx is None else total_idx
        
        eval1 = self.one_epoch(total_idx, "Valid", stage)
        self.valid_stats.append([eval1])
        self.best_valid_eval1 = max(self.best_valid_eval1, np.mean(eval1[0]))


    def one_epoch_test(self, total_idx=None):
        stage = None if self.option.flag_acceleration else 'time prediction'
        total_idx = self.idx_ls_dict["Test"] if total_idx is None else total_idx

        eval1 = self.one_epoch(total_idx, "Test", stage)
        self.test_stats.append([eval1])
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


    def train(self, total_idx=None):
        while (self.epoch < self.option.max_epoch and not self.early_stopped):
            self.one_epoch_train(total_idx=total_idx)

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


    def save_rule_scores(self):
        rule_scores, refType_scores = self.learner.get_rule_scores_fast_ver(self.sess)
        path_suffix = '' if not self.option.shift else '_time_shifting'

        if not os.path.exists('../output/' + self.option.dataset + '/rule_scores'):
            os.mkdir('../output/' + self.option.dataset + '/rule_scores')

        for rel in range(self.data['num_query']):
            output = {}
            output['rule_scores'] = rule_scores[rel, :].tolist()
            output['refType_scores'] = [scores[rel, :].tolist() for scores in refType_scores]
        
            cur_path = '../output/' + self.option.dataset + '/rule_scores/' + self.option.dataset + path_suffix
            # if self.option.shift and rel >= self.data['num_query']:
            #     cur_path += 'fix_ref_time_'
            cur_path += '_rel_' + str(rel) + '.json'
                    
            with open(cur_path, 'w') as file:
                json.dump(output, file)


    def save_state_vectors(self, mode, total_idx):
        save_path = '../output/{}/final_state_vec'.format(self.data['short_name'])
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        self.one_epoch(total_idx, mode, 'obtain state vec')


    def close_log_file(self):
        if self.option.create_log:
            self.log_file.close()