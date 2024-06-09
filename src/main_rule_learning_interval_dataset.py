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
from multiprocessing import Process, Queue



def run_experiment(queue, gpu, train, test, from_model_ckpt, dataset, num_step, num_layer, rnn_state_size, query_embed_size,
                   batch_size, print_per_batch, max_epoch, min_epoch, learning_rate, no_norm, thr, dropout, accuracy, top_k, shift, idx_ls):
    # Set up the experiment configuration
    d = {
        'gpu': gpu,
        'train': train,
        'test': test,
        'from_model_ckpt': from_model_ckpt,
        'dataset': dataset,
        'num_step': num_step,
        'num_layer': num_layer,
        'rnn_state_size': rnn_state_size,
        'query_embed_size': query_embed_size,
        'batch_size': batch_size,
        'print_per_batch': print_per_batch,
        'max_epoch': max_epoch,
        'min_epoch': min_epoch,
        'learning_rate': learning_rate,
        'no_norm': no_norm,
        'thr': thr,
        'dropout': dropout,
        'accuracy': accuracy,
        'top_k': top_k,
        'shift': shift,
    }

    option = Option(d)
    option.tag = time.strftime("%y-%m-%d-%H-%M")
    option.exps_dir = '../exps/'
    option.seed = 33
    option.different_states_for_rel_and_TR = False
    option.flag_acceleration = True   # only shallow layers are used
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
            experiment.train(idx_ls)

        if option.test:
            eval_aeIOU, eval_TAC, _ = experiment.test(idx_ls)
            res = {'aeIOU': eval_aeIOU, 'TAC': eval_TAC}
            # print('aeIOU: ', np.mean(eval_aeIOU))
            # print('TAC: ', np.mean(eval_TAC))

    experiment.close_log_file()
    if queue is not None:
        queue.put(res)
    print("="*36 + "Finish" + "="*36)



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

    args = parser.parse_args()


    # Create a list of configurations for different experiments
    config = {}
    config.update(vars(args))
    config['idx_ls'] = None
    config['queue'] = None
    experiment_configs = [config]
    

    if args.test:
        processor = Data_preprocessor()
        data = processor.prepare_data(args, save_option=False)
        queue = Queue()
        experiment_configs = []
        # To accelerate the testing process, we split the test set into 20 pieces, and run the experiments in parallel.
        for idx_ls in split_list_into_pieces(data['test_idx_ls'], 20):
            config = {}
            config.update(vars(args))
            config['idx_ls'] = idx_ls
            config['queue'] = queue
            experiment_configs.append(config)


    processes = []
    for config in experiment_configs:
        p = Process(target=run_experiment, kwargs=config)
        processes.append(p)
        p.start()

    for p in processes:
        p.join()

    if args.test:
        results = []
        while not queue.empty():
            results.append(queue.get())
        aeIOU = np.concatenate([res['aeIOU'] for res in results])
        TAC = np.concatenate([res['TAC'] for res in results])
        print('aeIOU: ', np.mean(aeIOU))
        print('TAC: ', np.mean(TAC))


if __name__ == "__main__":
    main()