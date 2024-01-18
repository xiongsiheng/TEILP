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

    os.environ["CUDA_VISIBLE_DEVICES"] = option.gpu



    data = {}
    dataset_index = ['icews14', 'icews05-15', 'gdelt100'].index(option.dataset)

    data['dataset'] = ['icews14', 'icews05-15', 'gdelt100'][dataset_index]
    data['dataset_name'] = ['icews14', 'icews05-15', 'gdelt100'][dataset_index]

    data['num_rel'] = [460, 502, 40][dataset_index]
    data['num_entity'] = [40000, 40000, 40000][dataset_index]
    data['num_TR'] = 4

    dataset_path = data['dataset']
    if option.shift:
        dataset_path = 'difficult_settings/' + dataset_path + '_time_shifting'

    data['train_edges'], data['valid_edges'], data['test_edges'] = obtain_all_data(dataset_path, shuffle_train_set=False)
    data['timestamp_range'] = [np.arange(0, 366, 1), np.arange(0, 4017, 1), np.arange(90, 456, 1)][dataset_index]
    data['num_samples_dist'] = [[72826, 8941, 8963], [368962, 46275, 46092], [390045, 48756, 48756]][dataset_index]


    # print(data['train_edges'])
    # print(data['valid_edges'])
    # print(data['test_edges'])
    # print(max(data['train_edges'][:, 1]))


    if option.dataset == 'gdelt100' and option.shift:
        with open('../data/gdelt100_sparse_edges.json') as json_file:
            edges = json.load(json_file)

        num_train = [16000, 2000]
        edges = np.array(edges)

        data['train_edges'] = edges[:num_train[0]]
        data['valid_edges'] = edges[num_train[0]:num_train[0] + num_train[1]]
        data['test_edges'] = edges[num_train[0] + num_train[1]:]

        data['num_samples_dist'] = [16000, 2000, 2000]

    # print(data['train_edges'])
    # print(data['valid_edges'])
    # print(data['test_edges'])
    # print(max(data['train_edges'][:, 1]))

    # sys.exit()


    data['train_idx_ls'] = list(range(data['num_samples_dist'][0]))
    data['valid_idx_ls'] = list(range(data['num_samples_dist'][0], data['num_samples_dist'][0] + data['num_samples_dist'][1]))
    data['test_idx_ls'] = list(range(data['num_samples_dist'][0] + data['num_samples_dist'][1], np.sum(data['num_samples_dist'])))


    # if option.shift:
    #     data['train_idx_ls'] += [idx + np.sum(data['num_samples_dist']) for idx in data['train_idx_ls']]
    #     data['valid_idx_ls'] += [idx + np.sum(data['num_samples_dist']) for idx in data['valid_idx_ls']]
    #     data['test_idx_ls'] += [idx + np.sum(data['num_samples_dist']) for idx in data['test_idx_ls']]


    data['rel_ls_no_dur'] = None

    data['mdb'], data['connectivity'], data['TEKG_nodes'] = None, None, None


    file_suffix = '_'
    if option.shift:
        file_suffix = '_time_shifting_'
        pattern_ls_fkt = {}
        stat_res_fkt = {}

    pattern_ls = {}
    stat_res = {}
    for rel in range(data['num_rel']):
        rel = str(rel)
        cur_path = '../output/'+ data['dataset'] + file_suffix[:-1] + '/' + data['dataset'] + file_suffix + 'pattern_ls_rel_'+ rel +'.json'
        if not os.path.exists(cur_path):
            pattern_ls[rel] = []
            stat_res[rel] = []
        else:
            with open(cur_path, 'r') as file:
                pattern_ls[rel] = json.load(file)
            with open('../output/'+ data['dataset'] + file_suffix[:-1] +'/' + data['dataset'] + file_suffix + 'stat_res_rel_'+ rel + '_Train.json', 'r') as file:
                stat_res[rel] = json.load(file)


        if option.shift:
            cur_path = '../output/'+ data['dataset'] + file_suffix[:-1] +'/' + data['dataset'] + file_suffix + 'fix_ref_time_pattern_ls_rel_'+ rel +'.json'
            if not os.path.exists(cur_path):
                pattern_ls_fkt[rel] = []
                stat_res_fkt[rel] = []
            else:
                with open(cur_path, 'r') as file:
                    pattern_ls_fkt[rel] = json.load(file)
                with open('../output/'+ data['dataset'] + file_suffix[:-1] +'/' + data['dataset'] + file_suffix + 'fix_ref_time_stat_res_rel_'+ rel + '_Train.json', 'r') as file:
                    stat_res_fkt[rel] = json.load(file)


    data['pattern_ls'] = pattern_ls
    data['stat_res'] = stat_res
    if option.shift:
        data['pattern_ls_fkt'] = pattern_ls_fkt
        data['stat_res_fkt'] = stat_res_fkt

    print("Data prepared.")


    option.this_expsdir = os.path.join(option.exps_dir, option.dataset + '_' + option.tag)
    if not os.path.exists(option.this_expsdir):
        os.makedirs(option.this_expsdir)
    option.ckpt_dir = os.path.join(option.this_expsdir, "ckpt")
    if not os.path.exists(option.ckpt_dir):
        os.makedirs(option.ckpt_dir)
    option.model_path = os.path.join(option.ckpt_dir, "model")


    option.num_step = [4, 4, 4][dataset_index]
    option.num_rule = [10000, 10000, 10000][dataset_index]
    option.flag_interval = False

    option.savetxt = option.this_expsdir + '/intermediate_res.txt'
    option.save()
    print("Option saved.")


    data['random_walk_res'] = None



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

            rule_scores = experiment.get_rule_scores()

            if not os.path.exists('../output/' + option.dataset + file_suffix[:-1]):
                os.mkdir('../output/' + option.dataset + file_suffix[:-1])

            for rel in range(data['num_rel']):
                cur_path = '../output/' + option.dataset + file_suffix[:-1] + '/' + option.dataset + file_suffix
                if option.shift and rel >= data['num_rel']//2:
                    cur_path += 'fix_ref_time_'
                cur_path += 'rule_scores_rel_' + str(rel) + '.json'
                with open(cur_path, 'w') as file:
                    json.dump(rule_scores[rel, :].tolist(), file)

        if option.test:
            eval_aeIOU, eval_TAC, eval_MAE = experiment.test()
            res = {'MAE': np.mean(eval_MAE)}
            print('MAE:', np.mean(eval_MAE))




    experiment.close_log_file()
    print("="*36 + "Finish" + "="*36)


if __name__ == "__main__":
    main()