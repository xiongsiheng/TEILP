import os
import argparse
import time
import sys
import json
import tensorflow as tf
import numpy as np
from model import Learner
from experiment import Experiment
from gadgets import *



def main():
    parser = argparse.ArgumentParser(description="Experiment setup")
    parser.add_argument('--gpu', default="0", type=str)
    parser.add_argument('--train', default=False, action="store_true")
    parser.add_argument('--test', default=False, action="store_true")
    parser.add_argument('--from_model_ckpt', default=None, type=str)
    parser.add_argument('--dataset', default=None, type=str)
    parser.add_argument('--num_step', default=3, type=int)
    parser.add_argument('--num_layer', default=1, type=int)
    parser.add_argument('--rnn_state_size', default=128, type=int)
    parser.add_argument('--query_embed_size', default=128, type=int)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--print_per_batch', default=3, type=int)
    parser.add_argument('--max_epoch', default=30, type=int)
    parser.add_argument('--min_epoch', default=5, type=int)
    parser.add_argument('--learning_rate', default=0.001, type=float)
    parser.add_argument('--no_norm', default=False, action="store_true")
    parser.add_argument('--thr', default=1e-20, type=float)
    parser.add_argument('--dropout', default=0., type=float)
    parser.add_argument('--accuracy', default=False, action="store_true")
    parser.add_argument('--top_k', default=10, type=int)
    parser.add_argument('--shift', default=0, type=int)


    d = vars(parser.parse_args())
    option = Option(d)

    option.tag = time.strftime("%y-%m-%d-%H-%M")
    option.exps_dir = '../exps/'
    option.seed = 33
    option.different_states_for_rel_and_TR = False
    option.flag_acceleration = True
    option.flag_ruleLen_split_ver = False
    option.flag_use_dur = False
    option.flag_state_vec_enhancement = False
    option.prob_type_for_training = ['max', 'mean'][0]
 

    os.environ["CUDA_VISIBLE_DEVICES"] = option.gpu
    tf.logging.set_verbosity(tf.logging.ERROR)

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

        experiment = Experiment(sess, saver, option, learner, data)
        print("Experiment created.")


        if option.train:
            print("Start training...")
            experiment.train()


        if option.test:
            eval_aeIOU, eval_TAC, _ = experiment.test()
            res = {'aeIOU': np.mean(eval_aeIOU), 'TAC': np.mean(eval_TAC)}
            print('aeIOU: ', np.mean(eval_aeIOU))
            print('TAC: ', np.mean(eval_TAC))


    experiment.close_log_file()
    print("="*36 + "Finish" + "="*36)


if __name__ == "__main__":
    main()