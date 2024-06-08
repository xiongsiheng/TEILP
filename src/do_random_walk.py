from joblib import Parallel, delayed
from Walker import Walker
from tqdm import tqdm



def my_walk(i, num_queries, num_processes, rel_idx, ver, train_edges, para_ls_for_walker, 
                path_name='', pos_examples_idx=None, time_shift_mode=0, output_path=None, ratio=None,
                imbalanced_rel=None, exp_idx=None):
    num_rel, num_pattern, num_ruleLen, dataset_using, f_interval = para_ls_for_walker

    my_walker = Walker(num_rel, num_pattern, num_ruleLen, dataset_using, f_interval)
    my_walker.walk_in_batch(i, num_queries, num_processes, rel_idx, ver, train_edges, 
                              path_name, pos_examples_idx, time_shift_mode, output_path, 
                              ratio, imbalanced_rel, exp_idx)


def random_walk(rel_ls, train_edges, para_ls_for_walker, ver='fast', path_name='', 
                    pos_examples_idx=None, time_shift_mode=0, output_path=None, ratio=None,
                    imbalanced_rel=None, exp_idx=None, num_processes = 24):
    for cur_rel in tqdm(rel_ls, desc='Random Walk'):
        num_queries = (len(train_edges) // 2) // num_processes
        if time_shift_mode:
            num_queries = len(train_edges) // num_processes

        Parallel(n_jobs=num_processes)(
            delayed(my_walk)(i, num_queries, num_processes, cur_rel, ver, train_edges,
                            para_ls_for_walker, path_name, pos_examples_idx, time_shift_mode, 
                            output_path, ratio, imbalanced_rel, exp_idx) for i in range(num_processes)
        )