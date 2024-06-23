import numpy as np
import sys
import json
import pandas as pd





class Walker(object):
    def __init__(self, num_rel, num_pattern, num_ruleLen, dataset_using, f_interval):
        self.num_rel = num_rel
        self.num_pattern = num_pattern
        self.num_ruleLen = num_ruleLen
        self.dataset_using = dataset_using

        if self.dataset_using == 'wiki':
            self.num_entites = 12554
        elif self.dataset_using == 'YAGO':
            self.num_entites = 10623

        self.max_explore_len = num_ruleLen
        self.f_interval = f_interval


    def _BFS_mat(self, st_node, adj_mat, num_nodes, targ_node, max_len):
        node_st = np.zeros((num_nodes, 1))
        node_st[int(st_node)] = 1
        res = node_st.copy()


        new_nodes_ls =[[int(st_node)]]
        num_hops = []
        for i in range(max_len):
            res = np.dot(adj_mat, res)
            res[res>1] = 1

            idx_ls = np.where(res==1)[0]

            # cur_new_idx_ls = list(set(idx_ls)-set(idx_ls_old))
            cur_new_idx_ls = idx_ls.copy()
            if len(cur_new_idx_ls) > 0:
                new_nodes_ls.append(cur_new_idx_ls)

            if res[int(targ_node)] == 1:
                num_hops.append(i+1)
                # res[targ_node] = 0

        return num_hops, new_nodes_ls


    def _find_common_nodes(self, ls1, ls2):
        return [list(set(ls1[i]).intersection(set(ls2[i]))) for i in range(len(ls1))]


    def _get_inv_idx(self, num_dataset, idx):
        if isinstance(idx, int):
            if idx >= num_dataset:
                return idx - num_dataset
            else:
                return idx + num_dataset
        else:
            x = idx.copy()
            x[idx >= num_dataset] = x[idx >= num_dataset] - num_dataset
            x[idx < num_dataset] = x[idx < num_dataset] + num_dataset
            return x


    def _get_inv_rel_mat(self, rel_idx):
        rel_idx_copy = rel_idx.copy()
        rel_idx[rel_idx_copy < self.num_rel//2] += self.num_rel//2
        rel_idx[rel_idx_copy >= self.num_rel//2] -= self.num_rel//2
        return rel_idx


    def _get_walks(self, walk_edges, columns):
        df_edges = []
        df = pd.DataFrame(
            walk_edges[0],
            columns=[c + str(0) for c in columns] + ["entity_" + str(1)],
            dtype=int,
        )

        df_edges.append(df)
        df = df[0:0]

        for i in range(1, len(walk_edges)):
            df = pd.DataFrame(
                walk_edges[i],
                columns=[c + str(i) for c in columns] + ["entity_" + str(i+1)],
                dtype=int,
            )

            df_edges.append(df)
            df = df[0:0]

        rule_walks = df_edges[0]
        df_edges[0] = df_edges[0][0:0]

        for i in range(1, len(df_edges)):
            rule_walks = pd.merge(rule_walks, df_edges[i], on=["entity_" + str(i)])
            df_edges[i] = df_edges[i][0:0]

        return rule_walks


    def _create_res_path(self, output_path, idx, ratio=None, imbalanced_rel=None, exp_idx=None):
        cur_path = '../output/{}/{}/{}'.format(self.dataset_using, output_path, self.dataset_using + '_idx_' + str(idx))

        if ratio is not None:
            cur_path += '_ratio_' + str(ratio)
        elif imbalanced_rel is not None:
            cur_path += '_rel_' + str(imbalanced_rel)

        if exp_idx is not None:
            cur_path += '_exp_' + str(exp_idx)

        cur_path += '.json'
        
        return cur_path


    def _create_masked_facts(self, train_edges, idx, time_shift_mode):
        masked_facts = np.delete(train_edges, [idx, self._get_inv_idx(len(train_edges)//2, idx)], 0)
        sample = train_edges[idx]

        if time_shift_mode:
            prev_facts = masked_facts[(masked_facts[:, 0] == sample[0]) & (masked_facts[:, 3] <= sample[3])]
            if len(prev_facts) == 0:
                masked_facts = []
            else:
                cur_time = max(prev_facts[:, 3])
                masked_facts = masked_facts[masked_facts[:, 3] <= cur_time]
                masked_facts[:, 4] = np.minimum(masked_facts[:, 4], cur_time)
        return masked_facts, sample


    def _create_adj_mat(self, masked_facts):
        edges_simp = masked_facts[:,[0,2]]
        edges_simp = np.unique(edges_simp, axis=0)
        edges_simp = edges_simp.astype(int)
        pos = list(edges_simp)
        rows, cols = zip(*pos)

        adj_mat = np.zeros((self.num_entites, self.num_entites))
        adj_mat[rows, cols] = 1
        return adj_mat

    
    def _check_repetition(self, walk_res):
        edge_len = 3 + self.f_interval
        ind_repetition = np.zeros((walk_res.shape[0],)).astype(bool)
        for i in range((walk_res.shape[1]-1)//edge_len - 1):
            for j in range(i+1, (walk_res.shape[1]-1)//edge_len):
                ind_repetition = np.logical_or(ind_repetition, np.all(walk_res[:, edge_len*i:edge_len*(i+1)+1] \
                                            == walk_res[:, edge_len*j:edge_len*(j+1)+1], axis=1))
                edge_rel_copy = walk_res[:, edge_len*i+1:edge_len*i+2].copy()
                ind_repetition = np.logical_or(ind_repetition, np.all(np.hstack(
                                                [walk_res[:, edge_len*(i+1):edge_len*(i+1)+1], 
                                                    self._get_inv_rel_mat(edge_rel_copy),
                                                    walk_res[:, edge_len*i+2: edge_len*i+2 + (1+self.f_interval)], 
                                                    walk_res[:, edge_len*i:edge_len*i+1]]) \
                                                    == walk_res[:, edge_len*j:edge_len*(j+1)+1], axis=1))

        walk_res = walk_res[~ind_repetition]
        return walk_res


    def walk(self, train_edges, rel_idx=None, idx_ls=None, pos_examples_idx=None, time_shift_mode=0, output_path=None, ratio=None, imbalanced_rel=None, exp_idx=None):
        idx_ls = range(len(train_edges)) if idx_ls is None else idx_ls
        for idx in idx_ls:
            if pos_examples_idx is not None and idx not in pos_examples_idx:
                continue

            if rel_idx is not None and not train_edges[idx][1] == rel_idx:
                continue

            cur_path = self._create_res_path(output_path, idx, ratio, imbalanced_rel, exp_idx)
            masked_facts, sample = self._create_masked_facts(train_edges, idx, time_shift_mode)
            walk_dict = {'query': sample.tolist()}

            if len(masked_facts) == 0:    
                with open(cur_path, 'w') as f:
                    json.dump(walk_dict, f)
                continue

            adj_mat = self._create_adj_mat(masked_facts)
            cur_num_hops1, new_nodes_ls1 = self._BFS_mat(sample[0], adj_mat, self.num_entites, sample[2], self.max_explore_len)

            if len(cur_num_hops1) > 0:
                _, new_nodes_ls2 = self._BFS_mat(sample[2], adj_mat, self.num_entites, sample[0], self.max_explore_len)
                masked_facts = np.unique(masked_facts, axis=0)

                for num in cur_num_hops1:
                    path_ls = self._find_common_nodes(new_nodes_ls1[:num+1], new_nodes_ls2[:num+1][::-1])
                    walk_edges = []
                    for i in range(num):
                        related_facts = masked_facts[np.isin(masked_facts[:,0], path_ls[i]) & np.isin(masked_facts[:,2], path_ls[i+1])]

                        if self.f_interval:
                            walk_edges.append(related_facts[:,[0,1,3,4,2]])
                        else:
                            walk_edges.append(related_facts[:,[0,1,3,2]])

                    if self.f_interval:
                        cur_ent_walk_res = self._get_walks(walk_edges, ["entity_" , "rel_", "ts_", "te_"]).to_numpy()
                    else:
                        cur_ent_walk_res = self._get_walks(walk_edges, ["entity_" , "rel_", "t_"]).to_numpy()

                    cur_ent_walk_res = self._check_repetition(cur_ent_walk_res)
          
                    if len(cur_ent_walk_res.tolist())>0:
                        walk_dict[num] = cur_ent_walk_res.tolist()
        
            with open(cur_path, 'w') as f:
                json.dump(walk_dict, f)
        
        return 


    def _create_idx_in_batch(self, i, num_queries, num_processes, rel_idx, num_total):
        if rel_idx < self.num_rel//2:
            s = 0
        else:
            s = num_total

        n_t = num_total

        num_rest_queries = n_t - (i + 1) * num_queries
        if (num_rest_queries >= num_queries) and (i + 1 < num_processes):
            queries_idx = range(s+i*num_queries, s+(i+1)*num_queries)
        else:
            queries_idx = range(s+i*num_queries, s+n_t)

        return queries_idx


    def walk_in_batch(self, i, num_queries, num_processes, rel_idx, ver, train_edges, path_name='', 
                            pos_examples_idx=None, time_shift_mode=0, output_path=None, ratio=None, imbalanced_rel=None, exp_idx=None):
        num_total = len(train_edges) // 2
        queries_idx = self._create_idx_in_batch(i, num_queries, num_processes, rel_idx, num_total)

        self.walk(train_edges, rel_idx, queries_idx, pos_examples_idx, time_shift_mode, output_path, ratio, imbalanced_rel, exp_idx)
        return