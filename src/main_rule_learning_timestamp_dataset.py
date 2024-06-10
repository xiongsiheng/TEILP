import os
import argparse
import time
import sys
import json
import tensorflow as tf
import numpy as np
from model import Learner
from experiment import Experiment
from utlis import *
from gadgets import *



def save_rule_scores(data, option, experiment):
    rule_scores = experiment.get_rule_scores()

    path_suffix = '_' if not option.shift else '_time_shifting_'
    if not os.path.exists('../output/' + option.dataset + path_suffix[:-1]):
        os.mkdir('../output/' + option.dataset + path_suffix[:-1])

    for rel in range(data['num_rel']):
        cur_path = '../output/' + option.dataset + path_suffix[:-1] + '/' + option.dataset + path_suffix
        if option.shift and rel >= data['num_rel']//2:
            cur_path += 'fix_ref_time_'
        cur_path += 'rule_scores_rel_' + str(rel) + '.json'
        with open(cur_path, 'w') as file:
            json.dump(rule_scores[rel, :].tolist(), file)


def main():
    parser = argparse.ArgumentParser(description="Experiment setup")
    parser.add_argument('--gpu', default="0", type=str)
    parser.add_argument('--train', default=False, action="store_true")
    parser.add_argument('--test', default=False, action="store_true")
    parser.add_argument('--from_model_ckpt', default=None, type=str)
    parser.add_argument('--dataset', default=None, type=str)
    parser.add_argument('--max_epoch', default=30, type=int)
    parser.add_argument('--min_epoch', default=5, type=int)
    parser.add_argument('--shift', default=0, type=int)


    d = vars(parser.parse_args())
    option = Option(d)
    option.tag = time.strftime("%y-%m-%d-%H-%M")
    option.exps_dir = '../exps/'
    option.seed = 33
    option.diff_states_for_rel_and_TR = False  # different entries of states in RNN for relations and temporal relations
    option.flag_acc = True   # acceleration (only shallow layers are used)
    option.flag_ruleLen_split_ver = False  # learn the rules with different lengths separately (Todo: fix issues)
    option.flag_use_dur = False  # use duration information (Todo: fix issues)
    option.flag_state_vec_enhance = False
    option.prob_type_for_train = ['max', 'mean'][0]  # given a rule, the probability of the rule is the max or mean of the probabilities of the samples

    os.environ["CUDA_VISIBLE_DEVICES"] = option.gpu

    processor = Data_preprocessor()
    data = processor.prepare_data(option)


    learner = Learner(option, data)
    print("Learner built.")

    saver = tf.train.Saver(max_to_keep=option.max_epoch)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = False
    config.log_device_placement = False
    config.allow_soft_placement = True

    with tf.Session(config=config) as sess:
        tf.set_random_seed(option.seed)
        sess.run(tf.global_variables_initializer())
        print("Session initialized.")

        if option.from_model_ckpt is not None:
            saver.restore(sess, option.from_model_ckpt)
            print("Checkpoint restored from model %s" % option.from_model_ckpt)

        # data.reset(option.batch_size)
        experiment = Experiment(sess, saver, option, learner, data)
        print("Experiment created.")


        if option.train:
            print("Start training...")
            experiment.train()
            
            save_rule_scores(data, option, experiment)

        if option.test:
            _, _, eval_MAE = experiment.test()
            res = {'MAE': np.mean(eval_MAE)}
            print('MAE:', np.mean(eval_MAE))


    experiment.close_log_file()
    print("="*36 + "Finish" + "="*36)


if __name__ == "__main__":
    main()