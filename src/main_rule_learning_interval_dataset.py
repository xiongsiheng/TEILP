import os
import argparse
import time
import sys
import json
import tensorflow as tf
import numpy as np
from model import Learner
from experiment import Experiment
from utils import *




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
    if option.dataset == 'YAGO':
        option.max_epoch = 80


    os.environ["CUDA_VISIBLE_DEVICES"] = option.gpu
    tf.logging.set_verbosity(tf.logging.ERROR)


    data = {}
    dataset_index = ['wiki', 'YAGO'].index(option.dataset)

    data['path'] = '../output/walk_res/'
    if option.shift:
        data['path'] = '../output/walk_res_time_shift/'

    data['dataset'] = ['WIKIDATA12k', 'YAGO11k'][dataset_index]
    if option.shift:
        data['dataset'] = 'difficult_settings/' + data['dataset'] + '_time_shifting'

    data['dataset_name'] = ['wiki', 'YAGO'][dataset_index]

    data['num_rel'] = [48, 20][dataset_index]
    data['num_entity'] = [40000, 40000][dataset_index]
    data['num_TR'] = 4

    data['train_edges'], data['valid_edges'], data['test_edges'] = obtain_all_data(data['dataset'], shuffle_train_set=False)

    data['timestamp_range'] = [np.arange(-3, 2024, 1), np.arange(-431, 2024, 1)][dataset_index]
    num_rel = data['num_rel']//2
    if option.shift:
        num_rel = data['num_rel'] # known time range change
    
    data['pattern_ls'], data['ts_stat_ls'], data['te_stat_ls'] = processing_stat_res(data['dataset_name'], num_rel, 
                                                                                     flag_with_ref_end_time=True, flag_time_shifting=option.shift)

    # print(data['pattern_ls'])
    # print(data['ts_stat_ls'])
    # print(data['te_stat_ls'])
    # sys.exit()

    data['num_samples_dist'] = [[32497, 4062, 4062], [16408, 2050, 2051]][dataset_index]
    data['train_idx_ls'] = list(range(data['num_samples_dist'][0]))
    data['valid_idx_ls'] = list(range(data['num_samples_dist'][0], data['num_samples_dist'][0] + data['num_samples_dist'][1]))
    data['test_idx_ls'] = list(range(data['num_samples_dist'][0] + data['num_samples_dist'][1], np.sum(data['num_samples_dist'])))

    if option.shift:
        data['train_idx_ls'] = [idx + np.sum(data['num_samples_dist']) for idx in data['train_idx_ls']]
        data['valid_idx_ls'] = [idx + np.sum(data['num_samples_dist']) for idx in data['valid_idx_ls']]
        data['test_idx_ls'] = [idx + np.sum(data['num_samples_dist']) for idx in data['test_idx_ls']]

    # print(data['train_idx_ls'])
    # print(data['valid_idx_ls'])
    # print(data['test_idx_ls'])

    data['rel_ls_no_dur'] = [[4, 16, 17, 20], [0, 7]][dataset_index]

    # todo: shift mode
    with open('../data/'+ data['dataset_name'] +'_time_pred_eval_rm_idx.json', 'r') as file:
        data['rm_ls'] = json.load(file)
    if option.shift:
        with open('../data/'+ data['dataset_name'] +'_time_pred_eval_rm_idx_shift_mode.json', 'r') as file:
            data['rm_ls'] = json.load(file)

    if option.flag_acceleration:
        data['mdb'], data['connectivity'], data['TEKG_nodes'] = None, None, None
    else:
        data['mdb'], data['connectivity'], data['TEKG_nodes'], data['num_entity'] = prepare_whole_TEKG_graph(data)

    # todo
    if option.flag_use_dur:
        with open("../output/"+ data['dataset_name'] + "_dur_preds.json", "r") as json_file:
            data['pred_dur'] = json.load(json_file)

    print("Data prepared.")


    option.this_expsdir = os.path.join(option.exps_dir, data['dataset_name'] + '_' + option.tag)
    if not os.path.exists(option.this_expsdir):
        os.makedirs(option.this_expsdir)
    option.ckpt_dir = os.path.join(option.this_expsdir, "ckpt")
    if not os.path.exists(option.ckpt_dir):
        os.makedirs(option.ckpt_dir)
    option.model_path = os.path.join(option.ckpt_dir, "model")


    option.num_step = [4, 6][dataset_index]
    option.num_rule = [1000, 6000][dataset_index]
    option.flag_interval = True

    option.savetxt = option.this_expsdir + '/intermediate_res.txt'
    option.save()

    print("Option saved.")

    learner = Learner(option, data)
    print("Learner built.")

    data['random_walk_res'] = None
    if option.train:
        data['random_walk_res'] = prepare_graph_random_walk_res(option, data, 'Train')
        print('Data preprocessed.')


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
            eval_aeIOU, eval_TAC, eval_MAE = experiment.test()
            res = {'aeIOU': np.mean(eval_aeIOU), 'TAC': np.mean(eval_TAC)}
            print('aeIOU: ', np.mean(eval_aeIOU))
            print('TAC: ', np.mean(eval_TAC))


    experiment.close_log_file()
    print("="*36 + "Finish" + "="*36)


if __name__ == "__main__":
    main()