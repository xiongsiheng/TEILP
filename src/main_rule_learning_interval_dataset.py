import os
import sys
import argparse
import time
import tensorflow as tf
import numpy as np
from model import Learner
from experiment import Experiment
from gadgets import *
from utlis import *
from joblib import Parallel, delayed




def main():
    parser = argparse.ArgumentParser(description="Experiment setup")
    parser.add_argument('--gpu', default="0", type=str)
    parser.add_argument('--train', default=False, action="store_true")
    parser.add_argument('--test', default=False, action="store_true")
    parser.add_argument('--from_model_ckpt', default=None, type=str)
    parser.add_argument('--dataset', default=None, type=str)
    parser.add_argument('--shift', default=False, type=bool)   # whether use time-shifting setting  (Todo: fix issues)

    d = vars(parser.parse_args())
    option = Option(d)
    option.tag = time.strftime("%y-%m-%d-%H-%M")
    option.exps_dir = '../exps/'
    
    # We consider using different states for relation and temporal relation (TR). Found not necessary.
    option.different_states_for_rel_and_TR = False
    
    # We consider only using the shallow layers to accelerate the training and inference.
    # Set flag_acceleration to False to use the RNN structure.
    option.flag_acceleration = True
    
    # Use different scores for different lengths of rules. Found not necessary.
    option.flag_ruleLen_split_ver = False

    # Use the duration information. Found not necessary.
    option.flag_use_dur = False

    # We enhance the RNN structure by adding the shallow layers.
    # When flag_acceleration is True, we only use the shallow layers.
    # When both flag_acceleration and flag_state_vec_enhancement are False, we only use the RNN structure.
    # When flag_acceleration is False and flag_state_vec_enhancement is True, we use both the RNN and shallow layers.
    option.flag_state_vec_enhancement = True
    
    # If an event satisfies multiple rules, choose the max or mean probability of the query time during training.
    # During inference, we always choose the mean probability since we don't have the prior.
    option.prob_selection_for_training = ['max', 'mean'][0]
    
    # If we use both RNN and shallow layers, it is better to sample the training data for efficiency.
    # option.num_samples_per_rel = -1 means no sampling for the training data.
    option.num_samples_per_rel = -1 if option.flag_acceleration else 200


    # Setting for the model. No need to change.
    option.seed = 33
    option.num_layer = 1
    option.rnn_state_size = 128
    option.query_embed_size = 128
    option.no_norm = False
    option.thr = 1e-20
    option.dropout = 0.
    option.max_epoch = 30 if option.flag_acceleration else 10
    option.min_epoch = 5
    option.create_log = False


    os.environ["CUDA_VISIBLE_DEVICES"] = option.gpu
    tf.logging.set_verbosity(tf.logging.ERROR)

    
    # For the first time, set preprocess_walk_res=True to preprocess and save the probabilities of query time.
    # After that, we can set it to False to skip this step.
    # We only do preprocessing for training data.
    processor = Data_Processor()
    data = processor.prepare_data(option, preprocess_walk_res=True)


    # Build the model.
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
            experiment.close_log_file()

        if option.test:
            # Prepare for testing:
            # In acceleration mode, we save the rule scores.
            # In normal mode, we save the state vectors and attention scores.
            print("Prepare for testing...")
            if option.flag_acceleration:
                experiment.save_rule_scores()
            else:
                experiment.save_state_vectors(data['test_idx_ls'])


    if option.test:
        print("Start testing...")        
        # After preparing, we don't need to load the model. Instead, we use a distributed online strategy to accelerate the inference.
        experiment = Experiment(None, None, option, None, data)
        
        num_batches = 24
        idx_ls = split_list_into_batches(data['test_idx_ls'], num_batches=num_batches)
        outputs = Parallel(n_jobs=num_batches)(
                delayed(experiment.test)(idx_batch) for idx_batch in idx_ls
                )
      
        eval_aeIOU, eval_TAC = [], []
        for output in outputs:
            eval_aeIOU += output[0]
            eval_TAC += output[1]

        print('aeIOU: ', np.mean(eval_aeIOU), len(eval_aeIOU))
        print('TAC: ', np.mean(eval_TAC), len(eval_TAC))

    
    




if __name__ == "__main__":
    main()