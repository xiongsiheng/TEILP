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

        # save_data(self.option.savetxt, len(idx_ls))

        timestamp_range = self.data['timestamp_range']

        flag_rm_seen_ts = False # rm seen ts in training set
        if self.data['dataset_name'] in ['icews14', 'icews05-15', 'gdelt'] and not self.option.shift:
            flag_rm_seen_ts = True
            train_edges = self.data['train_edges']

        if self.option.flag_use_dur:
            pred_dur_dict = self.data['pred_dur']


        if self.option.flag_acceleration:
            if self.data['dataset_name'] in ['icews14', 'icews05-15', 'gdelt']:
                model = TEKG_timestamp_acc_ver(self.option, self.data)
            else:
                model = TEKG_int_acc_ver(self.option, self.data)
        else:
            model = TEKG(self.option, self.data)


        for i, batch_idx_ls in enumerate(tqdm(idx_ls, desc=mode)):
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
                qq, query_rels, refNode_source, res_random_walk, probs, valid_sample_idx, input_intervals, input_samples, ref_time_ls = model.create_graph(batch_idx_ls, mode)
            else:
                qq, hh, tt, mdb, connectivity, probs, valid_sample_idx, valid_ref_event_idx, input_intervals, input_samples = model.create_graph(batch_idx_ls, mode)

            # save_data(self.option.savetxt, 'TEKG_prepared!')

            # print(len(valid_sample_idx))

            if len(valid_sample_idx) == 0:
                continue

            probs = np.array(probs).transpose(1, 0, 2)
            # print(probs.shape)

            input_intervals = np.array(input_intervals)

            # print(probs)
            # print(ref_time_ls)


            if self.option.flag_acceleration:
                output = run_fn(self.sess, query_rels, refNode_source, res_random_walk, probs)
            else:
                output = run_fn(self.sess, qq, hh, tt, mdb, connectivity, probs, valid_sample_idx, valid_ref_event_idx)

            # save_data(self.option.savetxt, 'model processed!')
            # print(output)

            if mode == "Train":
                epoch_loss += list(output)
                # save_data(self.option.savetxt, i)
                # save_data(self.option.savetxt, output)
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

                # print(timestamp_range)
                # print(np.argmax(prob_ts, axis=1))
                # print(timestamp_range[np.argmax(prob_ts, axis=1)])

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

                    # print(preds)
                    # print(valid_sample_idx)
                    # print(input_intervals[valid_sample_idx])

                    if self.data['rel_ls_no_dur'] is not None:
                        qq = np.array(qq)[valid_sample_idx]
                        x_tmp = preds[np.isin(qq, self.data['rel_ls_no_dur'])]
                        x_tmp = np.mean(x_tmp, axis=1).reshape((-1,1))
                        preds[np.isin(qq, self.data['rel_ls_no_dur'])] = np.hstack((x_tmp, x_tmp))


                # save_data(self.option.savetxt, i)
                # save_data(self.option.savetxt, preds)
                # save_data(self.option.savetxt, input_intervals[valid_sample_idx])


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
            print("Epoch %d mode %s Loss %0.4f " % (self.epoch+1, mode, np.mean(epoch_loss)))
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