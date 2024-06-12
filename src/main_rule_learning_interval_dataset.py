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
from utlis import *
from joblib import Parallel, delayed




def save_rule_scores(data, option, experiment):
    rule_scores, refType_scores = experiment.get_rule_scores()

    path_suffix = '_' if not option.shift else '_time_shifting_'
    if not os.path.exists('../output/' + option.dataset + path_suffix[:-1]):
        os.mkdir('../output/' + option.dataset + path_suffix[:-1])

    for rel in range(data['num_rel']):
        output = {}
        output['rule_scores'] = rule_scores[rel, :].tolist()
        output['refType_scores'] = [scores[rel, :].tolist() for scores in refType_scores]
    
        cur_path = '../output/' + option.dataset + path_suffix[:-1] + '/' + option.dataset + path_suffix
        if option.shift and rel >= data['num_rel']//2:
            cur_path += 'fix_ref_time_'
        cur_path += 'rule_scores_rel_' + str(rel) + '.json'
                
        with open(cur_path, 'w') as file:
            json.dump(output, file)




def main():
    parser = argparse.ArgumentParser(description="Experiment setup")
    parser.add_argument('--gpu', default="0", type=str)
    parser.add_argument('--train', default=False, action="store_true")
    parser.add_argument('--test', default=False, action="store_true")
    parser.add_argument('--from_model_ckpt', default=None, type=str)
    parser.add_argument('--dataset', default=None, type=str)
    parser.add_argument('--shift', default=0, type=int)

    d = vars(parser.parse_args())
    option = Option(d)
    option.tag = time.strftime("%y-%m-%d-%H-%M")
    option.exps_dir = '../exps/'
    option.seed = 33
    
    option.num_layer = 1
    option.rnn_state_size = 128
    option.query_embed_size = 128
    option.no_norm = False
    option.thr = 1e-20
    option.dropout = 0.
    option.max_epoch = 30
    option.min_epoch = 5
    option.create_log = False

    option.different_states_for_rel_and_TR = False
    option.flag_acceleration = True   # only shallow layers are used
    option.flag_ruleLen_split_ver = False
    option.flag_use_dur = False
    option.flag_state_vec_enhancement = False
    option.prob_type_for_training = ['max', 'mean'][0]
    
    # If we use both RNN and shallow layers, it is better to sample the training data for efficiency.
    option.num_samples_per_rel = -1 if option.flag_acceleration else 500

    os.environ["CUDA_VISIBLE_DEVICES"] = option.gpu
    tf.logging.set_verbosity(tf.logging.ERROR)

    processor = Data_preprocessor()
    # For the first time, set process_walk_res=True to generate the walk results.
    # After that, set it to False to load the walk results directly.
    # Once we change the setting, we need to re-generate the walk results.
    data = processor.prepare_data(option, process_walk_res=True)


    if option.train:
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
           
            print("Start training...")
            experiment.train()
            save_rule_scores(data, option, experiment)
        
            experiment.close_log_file()
    else:
        experiment = Experiment(None, None, option, None, data)
        print("Experiment created.")

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