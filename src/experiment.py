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

        if self.data['dataset_name'] in ['wiki', 'YAGO']:
            self.metrics = ['aeIOU', 'TAC']
        else:
            self.metrics = ['MAE']


    def _calculate_output(self, myTEKG, run_fn, batch_idx_ls, mode, flag_rm_seen_ts, train_edges, timestamp_range, pred_dur_dict):
        '''
        Calculate the output of the model for a batch of data.
        '''
        if self.option.flag_acceleration:
            # Shape: qq: [] * dummy_batch_size (num_events);  refNode_source: [(dummy_batch_size,)] * batch_size;
            #        res_random_walk: (num_rules_in_total_for_different_events, 2); [event_idx, rule_idx]
            #        probs: [[(num_timestamp, )] * dummy_batch_size ] * 8
            qq, query_rels, refNode_source, res_random_walk, valid_sample_idx, input_intervals, input_samples, final_preds = myTEKG.graph.create_graph(batch_idx_ls, mode)
        else:
            qq, hh, tt, connectivity_rel, connectivity_TR, probs, valid_sample_idx, valid_ref_event_idx, input_intervals, input_samples, final_preds = myTEKG.graph.create_graph(batch_idx_ls, mode)


        if len(valid_sample_idx) == 0:
            # all samples are invalid (no walk can be found)
            return [], [], []


        if mode == "Train":            
            inputs = [query_rels, refNode_source, res_random_walk] if self.option.flag_acceleration else \
                     [qq, hh, tt, connectivity_rel, connectivity_TR, probs, valid_sample_idx]
            output = run_fn(self.sess, inputs)
            preds, gts = [], []
        else:
            output = final_preds  # [(bacth_size, num_timestamp)] * (1 + int(flag_int))

            # Obtain the predictions from the output.
            if len(output[0]) == 0:
                # Since there is no walk for the sample, we just random guess the time.
                preds = [[1900, 2000] for _ in range(len(valid_sample_idx))]
            else:
                preds = []
                for prob_t in output:
                    # prob_t: shape: (dummy_batch_size, num_timestamp)
                    prob_t = prob_t.reshape(-1)
                    prob_t = np.array(split_list_into_batches(prob_t, batch_size=len(timestamp_range)))
                    prob_t = self._rm_seen_time(flag_rm_seen_ts, valid_sample_idx, input_samples, prob_t, train_edges, timestamp_range)
                    preds.append(prob_t)
                preds = self._adjust_preds_based_on_dur(preds, batch_idx_ls, valid_sample_idx, pred_dur_dict, qq)
            
            gts = np.array(input_intervals)[valid_sample_idx]

        return output, preds, gts


    def _rm_seen_time(self, flag_rm_seen_ts, valid_sample_idx, input_samples, prob_t, train_edges, timestamp_range):
        if flag_rm_seen_ts:
            input_samples = input_samples[valid_sample_idx]
            prob_t_new = []
            for idx in range(len(prob_t)):
                cur_prob_t = prob_t[idx]
                seen_ts = train_edges[np.all(train_edges[:, :3] == input_samples[idx, :3], axis=1), 3]
                seen_ts = [timestamp_range.tolist().index(ts) for ts in seen_ts if ts in timestamp_range.tolist()]
                cur_prob_t[seen_ts] = 0
                
                pred_t = timestamp_range[np.argmax(cur_prob_t)]
                prob_t_new.append(pred_t)

            prob_t = np.array(prob_t_new).reshape((-1, 1))
        else:
            prob_t = timestamp_range[np.argmax(prob_t, axis=1)].reshape((-1, 1))
        return prob_t


    def _adjust_preds_based_on_dur(self, preds, batch_idx_ls, valid_sample_idx, pred_dur_dict, qq):
        if self.option.flag_use_dur:
            pred_ts, pred_te = preds[0], preds[1]
            pred_dur = []
            for data_idx in np.array(batch_idx_ls)[valid_sample_idx]:
                pred_dur1 = pred_dur_dict[str(data_idx - self.data['num_samples_dist'][1])]
                pred_dur.append(abs(pred_dur1[1] - pred_dur1[0]))

            pred_te = pred_ts + np.array(pred_dur).reshape((-1, 1))
            preds = np.hstack([pred_ts, pred_te])
        
        if self.data['rel_ls_no_dur'] is not None:
            preds = np.hstack(preds)
            qq = np.array(qq)[valid_sample_idx]
            x_tmp = preds[np.isin(qq, self.data['rel_ls_no_dur'])]
            x_tmp = np.mean(x_tmp, axis=1).reshape((-1,1))
            preds[np.isin(qq, self.data['rel_ls_no_dur'])] = np.hstack((x_tmp, x_tmp))

        return preds


    def running_model(self, model, run_fn, batch_size, idx_ls, mode, flag_rm_seen_ts):
        timestamp_range = self.data['timestamp_range']
        train_edges = self.data['train_edges']
        pred_dur_dict = self.data['pred_dur'] if self.option.flag_use_dur else None

        idx_ls = split_list_into_batches(idx_ls, batch_size=batch_size)
        
        epoch_loss, epoch_eval_aeIOU, epoch_eval_TAC, epoch_eval_MAE = [], [], [], []
        for batch_idx_ls in tqdm(idx_ls, desc=mode):
            output, preds, gts = self._calculate_output(model, run_fn, batch_idx_ls, mode, flag_rm_seen_ts, train_edges, timestamp_range, pred_dur_dict)
            if mode == "Train":
                epoch_loss += list(output)
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
                msg = self.msg_with_time(
                    "Epoch %d mode %s Loss %0.4f " 
                    % (self.epoch+1, mode, np.mean(epoch_loss)))
                self.log_file.write(msg + "\n")
            
            print("Epoch %d mode %s Loss %0.4f " % (self.epoch+1, mode, np.mean(epoch_loss)))
            return epoch_loss
        else:
            return epoch_eval_aeIOU, epoch_eval_TAC, epoch_eval_MAE


    def one_epoch(self, mode, total_idx=None, batch_size=32):
        flag_rm_seen_ts = False # remove seen ts in training set
        if self.data['dataset_name'] in ['icews14', 'icews05-15', 'gdelt'] and not self.option.shift:
            flag_rm_seen_ts = True

        # Create the model based on the option. 
        myTEKG = TEKG_family(self.option, self.data)

        # To accelerate the inference, we use online algorithm to calculate the output instead of re-running the model.
        run_fn = self.learner.update if mode == "Train" else None

        # Prepare the index list for the current mode.
        idx_ls_dict = {"Train": self.data['train_idx_ls'], "Valid": self.data['valid_idx_ls'], "Test": self.data['test_idx_ls']}
        idx_ls = total_idx if total_idx is not None else idx_ls_dict[mode]
        if mode == "Train":
            random.shuffle(idx_ls)
     
        return self.running_model(myTEKG, run_fn, batch_size, idx_ls, mode, flag_rm_seen_ts)


    def one_epoch_train(self, total_idx=None):
        batch_size = 8 if self.option.flag_acceleration else 4
        loss = self.one_epoch("Train", total_idx=total_idx, batch_size=batch_size)
        self.train_stats.append([loss])


    def one_epoch_valid(self, total_idx=None):
        batch_size = 8 if self.option.flag_acceleration else 4
        eval1 = self.one_epoch("Valid", total_idx=total_idx, batch_size=batch_size)
        self.valid_stats.append([eval1])
        self.best_valid_eval1 = max(self.best_valid_eval1, np.mean(eval1[0]))


    def one_epoch_test(self, total_idx=None):
        batch_size = 8 if self.option.flag_acceleration else 4
        eval1 = self.one_epoch("Test", total_idx=total_idx, batch_size=batch_size)
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

            if self.early_stop():
                self.early_stopped = True
                print("Early stopped at epoch %d" % (self.epoch))


    def test(self, total_idx=None):
        eval1 = self.one_epoch_test(total_idx=total_idx)
        return eval1


    def get_rule_scores(self):
        # Toodo: normal version
        if self.option.flag_acceleration:
            rule_scores, refType_scores = self.learner.get_rule_scores_acc(self.sess)
            return rule_scores, refType_scores

    def close_log_file(self):
        self.log_file.close()