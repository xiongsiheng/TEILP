import json
import numpy as np


class Grapher(object):
    def __init__(self, dataset_dir, num_entity, num_rel, timestamp_ls, flag_interval=False):
        """
        Store information about the graph (train/valid/test set).
        Add corresponding inverse quadruples to the data.

        Parameters:
            dataset_dir (str): path to the graph dataset directory

        Returns:
            None
        """

        self.flag_interval = flag_interval

        self.dataset_dir = dataset_dir

        # self.entity2id = json.load(open(dataset_dir + "entity2id.json"))

        self.entity2id = {}
        for idx in range(num_entity):
            self.entity2id[str(idx)] = idx

        # self.relation2id_ori = json.load(open(dataset_dir + "relation2id.json"))

        self.relation2id_ori = {}
        for idx in range(num_rel):
            self.relation2id_ori[str(idx)] = idx

        self.relation2id = self.relation2id_ori.copy()
        counter = len(self.relation2id_ori)
        for relation in self.relation2id_ori:
            self.relation2id["_" + relation] = counter  # Inverse relation
            counter += 1

        # self.ts2id = json.load(open(dataset_dir + "ts2id.json"))
        self.ts2id = {}
        for idx in timestamp_ls:
            self.ts2id[str(idx)] = idx


        self.id2entity = dict([(v, k) for k, v in self.entity2id.items()])
        self.id2relation = dict([(v, k) for k, v in self.relation2id.items()])
        self.id2ts = dict([(v, k) for k, v in self.ts2id.items()])

        self.inv_relation_id = dict()
        num_relations = len(self.relation2id_ori)
        for i in range(num_relations):
            self.inv_relation_id[i] = i + num_relations
        for i in range(num_relations, num_relations * 2):
            self.inv_relation_id[i] = i % num_relations

        self.train_idx = self.create_store("train.txt")
        self.valid_idx = self.create_store("valid.txt")
        self.test_idx = self.create_store("test.txt")
        self.all_idx = np.vstack((self.train_idx, self.valid_idx, self.test_idx))

        print("Grapher initialized.")

    def create_store(self, file):
        """
        Store the quadruples from the file as indices.
        The quadruples in the file should be in the format "subject\trelation\tobject\ttimestamp\n".

        Parameters:
            file (str): file name

        Returns:
            store_idx (np.ndarray): indices of quadruples
        """

        with open(self.dataset_dir + file, "r") as f:
            quads = f.readlines()
        store = self.split_quads(quads)
        store_idx = self.map_to_idx(store)
        store_idx = self.add_inverses(store_idx)

        return store_idx

    def split_quads(self, quads):
        """
        Split quadruples into a list of strings.

        Parameters:
            quads (list): list of quadruples
                          Each quadruple has the form "subject\trelation\tobject\ttimestamp\n".

        Returns:
            split_q (list): list of quadruples
                            Each quadruple has the form [subject, relation, object, timestamp].
        """

        split_q = []
        for quad in quads:
            split_q.append(quad[:-1].split("\t"))

        return split_q

    def map_to_idx(self, quads):
        """
        Map quadruples to their indices.

        Parameters:
            quads (list): list of quadruples
                          Each quadruple has the form [subject, relation, object, timestamp].

        Returns:
            quads (np.ndarray): indices of quadruples
        """

        subs = [self.entity2id[x[0]] for x in quads]
        rels = [self.relation2id[x[1]] for x in quads]
        objs = [self.entity2id[x[2]] for x in quads]
        tss = [self.ts2id[x[3]] for x in quads]

        if self.flag_interval:
            tes = [self.ts2id[x[4]] for x in quads]
            quads = np.column_stack((subs, rels, objs, tss, tes))

            arr = quads
            sorted_arr = arr.copy()
            mask = arr[:, 4] < arr[:, 3]
            sorted_arr[:, 3], sorted_arr[:, 4] = np.where(mask, [arr[:, 4], arr[:, 3]], [arr[:, 3], arr[:, 4]])
            quads = sorted_arr

        else:
            quads = np.column_stack((subs, rels, objs, tss))

        return quads

    def add_inverses(self, quads_idx):
        """
        Add the inverses of the quadruples as indices.

        Parameters:
            quads_idx (np.ndarray): indices of quadruples

        Returns:
            quads_idx (np.ndarray): indices of quadruples along with the indices of their inverses
        """

        subs = quads_idx[:, 2]
        rels = [self.inv_relation_id[x] for x in quads_idx[:, 1]]
        objs = quads_idx[:, 0]
        tss = quads_idx[:, 3]

        if self.flag_interval:
            tes = quads_idx[:, 4]
            inv_quads_idx = np.column_stack((subs, rels, objs, tss, tes))
        else:
            inv_quads_idx = np.column_stack((subs, rels, objs, tss))

        quads_idx = np.vstack((quads_idx, inv_quads_idx))

        return quads_idx




class Temporal_Walk(object):
    def __init__(self, learn_data, inv_relation_id, transition_distr, flag_interval=False):
        """
        Initialize temporal random walk object.

        Parameters:
            learn_data (np.ndarray): data on which the rules should be learned
            inv_relation_id (dict): mapping of relation to inverse relation
            transition_distr (str): transition distribution
                                    "unif" - uniform distribution
                                    "exp"  - exponential distribution

        Returns:
            None
        """

        self.learn_data = learn_data
        self.inv_relation_id = inv_relation_id
        self.transition_distr = transition_distr
        self.neighbors = store_neighbors(learn_data)
        self.edges = store_edges(learn_data)

        self.flag_interval = flag_interval



    def sample_start_edge(self, rel_idx):
        """
        Define start edge distribution.

        Parameters:
            rel_idx (int): relation index

        Returns:
            start_edge (np.ndarray): start edge
        """

        rel_edges = self.edges[rel_idx]
        start_edge = rel_edges[np.random.choice(len(rel_edges))]

        return start_edge

    def sample_next_edge(self, filtered_edges, cur_ts, transition_distr=None):
        """
        Define next edge distribution.

        Parameters:
            filtered_edges (np.ndarray): filtered (according to time) edges
            cur_ts (int): current timestamp

        Returns:
            next_edge (np.ndarray): next edge
        """

        if transition_distr is None:
            transition_distr = self.transition_distr

        if transition_distr == "unif":
            next_edge = filtered_edges[np.random.choice(len(filtered_edges))]
        elif transition_distr == "exp":
            tss = filtered_edges[:, 3]
            prob = np.exp(-np.abs(tss - cur_ts))

            if np.sum(prob) == 0:
                next_edge = filtered_edges[np.random.choice(len(filtered_edges))]
            else:
                try:
                    prob = prob / np.sum(prob)
                    next_edge = filtered_edges[
                        np.random.choice(range(len(filtered_edges)), p=prob)
                    ]
                except ValueError:  # All timestamps are far away
                    next_edge = filtered_edges[np.random.choice(len(filtered_edges))]

        return next_edge


    def transition_step(self, cur_node, cur_ts, prev_edge, start_node, step, L, 
                        cur_te=None, TR=None, rel=None, start_edge=None, start_ts=None, window=None, flag_time_shifting=0, ref_time=None):
        """
        Sample a neighboring edge given the current node and timestamp.
        In the second step (step == 1), the next timestamp should be smaller than the current timestamp.
        In the other steps, the next timestamp should be smaller than or equal to the current timestamp.
        In the last step (step == L-1), the edge should connect to the source of the walk (cyclic walk).
        It is not allowed to go back using the inverse edge.

        Parameters:
            cur_node (int): current node
            cur_ts (int): current timestamp
            prev_edge (np.ndarray): previous edge
            start_node (int): start node
            step (int): number of current step
            L (int): length of random walk

        Returns:
            next_edge (np.ndarray): next edge
        """

        if cur_node not in self.neighbors:
            next_edge = []
            return next_edge

        next_edges = self.neighbors[cur_node]

        if start_ts is not None and window is not None:
            next_edges = next_edges[(next_edges[:, 3] >= start_ts + window[0]) & (next_edges[:, 3] <= start_ts + window[1])]

        if (ref_time is not None) and flag_time_shifting:
            next_edges = next_edges[next_edges[:, 3] <= ref_time]


        if cur_te is not None:
            if TR == 'bf':
                filtered_edges = next_edges[next_edges[:, 4] < cur_ts]
            elif TR == 'af':
                filtered_edges = next_edges[next_edges[:, 3] > cur_te]
            elif TR == 'touch':
                filtered_edges = next_edges[(next_edges[:, 4] >= cur_ts) & (next_edges[:, 3] <= cur_te)]
            else:
                filtered_edges = next_edges
        else:
            if TR == 'bf':
                filtered_edges = next_edges[next_edges[:, 3] < cur_ts]
            elif TR == 'af':
                filtered_edges = next_edges[next_edges[:, 3] > cur_ts]
            elif TR == 'touch':
                filtered_edges = next_edges[next_edges[:, 3] == cur_ts]
            else:
                filtered_edges = next_edges


        inv_edge = [
            cur_node,
            self.inv_relation_id[prev_edge[1]],
            prev_edge[0],
            cur_ts,
        ]

        if cur_te is not None:
            inv_edge.append(cur_te)


        row_idx = np.where(np.all(filtered_edges == inv_edge, axis=1))
        filtered_edges = np.delete(filtered_edges, row_idx, axis=0)

        if start_edge is not None:
            row_idx = np.where(np.all(filtered_edges == start_edge, axis=1))
            filtered_edges = np.delete(filtered_edges, row_idx, axis=0)

            inv_edge = [
                start_edge[2],
                self.inv_relation_id[start_edge[1]],
                start_edge[0],
                start_edge[3],
            ]

            if cur_te is not None:
                inv_edge.append(start_edge[4])

            row_idx = np.where(np.all(filtered_edges == inv_edge, axis=1))
            filtered_edges = np.delete(filtered_edges, row_idx, axis=0)


        if rel is not None:
            filtered_edges = filtered_edges[filtered_edges[:,1] == rel]


        # if step == 1:  # The next timestamp should be smaller than the current timestamp
        #     # filtered_edges = next_edges[next_edges[:, 3] < cur_ts]
        #     # filtered_edges = next_edges
        #     filtered_edges = next_edges[next_edges[:, 3] < cur_ts]
        # else:  # The next timestamp should be smaller than or equal to the current timestamp
        #     filtered_edges = next_edges[next_edges[:, 3] <= cur_ts]
        #     # filtered_edges = next_edges
        #     # Delete inverse edge
        #     inv_edge = [
        #         cur_node,
        #         self.inv_relation_id[prev_edge[1]],
        #         prev_edge[0],
        #         cur_ts,
        #     ]

        #     if cur_te is not None:
        #         inv_edge.append(cur_te)

        #     row_idx = np.where(np.all(filtered_edges == inv_edge, axis=1))
        #     filtered_edges = np.delete(filtered_edges, row_idx, axis=0)

        if step == L - 1:  # Find an edge that connects to the source of the walk
            filtered_edges = filtered_edges[filtered_edges[:, 2] == start_node]

        if len(filtered_edges):
            if step == 1:
                next_edge = self.sample_next_edge(filtered_edges, cur_ts, 'unif')
            else:
                next_edge = self.sample_next_edge(filtered_edges, cur_ts)
        else:
            next_edge = []

        return next_edge



    def sample_walk(self, L, rel_idx, prev_edge=None, TR_ls=None, rel_ls=None, window=None, flag_time_shifting=False, fix_ref_time=False):
        """
        Try to sample a cyclic temporal random walk of length L (for a rule of length L-1).

        Parameters:
            L (int): length of random walk
            rel_idx (int): relation index

        Returns:
            walk_successful (bool): if a cyclic temporal random walk has been successfully sampled
            walk (dict): information about the walk (entities, relations, timestamps)
        """

        walk_successful = True
        walk = dict()
        if prev_edge is None:
            prev_edge = self.sample_start_edge(rel_idx)
        
        start_edge = prev_edge.copy()
        start_ts = start_edge[3]

        ref_time = None
        if flag_time_shifting:
            inv_edge = [
                        start_edge[2],
                        self.inv_relation_id[start_edge[1]],
                        start_edge[0],
                        start_edge[3],
                    ]

            filtered_edges = self.learn_data.copy()

            if fix_ref_time:
                prev_facts = filtered_edges[(filtered_edges[:, 2] == start_edge[2]) & (filtered_edges[:, 1] == start_edge[1]) & (filtered_edges[:, 3] <= start_edge[3])]
            else:
                prev_facts = filtered_edges[(filtered_edges[:, 0] == start_edge[0]) & (filtered_edges[:, 1] == start_edge[1]) & (filtered_edges[:, 3] <= start_edge[3])]

            row_idx = np.where(np.all(prev_facts == inv_edge, axis=1))
            prev_facts = np.delete(prev_facts, row_idx, axis=0)

            row_idx = np.where(np.all(prev_facts == start_edge, axis=1))
            prev_facts = np.delete(prev_facts, row_idx, axis=0)


            if len(prev_facts) == 0:
                walk_successful = False
                return walk_successful, walk, ref_time
            else:
                ref_time = max(prev_facts[:, 3])
                # print(prev_edge)
                # print(prev_facts)


        start_node = prev_edge[0]
        cur_node = prev_edge[2]
        cur_ts = prev_edge[3]
        walk["entities"] = [start_node, cur_node]
        walk["relations"] = [prev_edge[1]]
        walk["ts"] = [cur_ts]

        cur_te = None
        if self.flag_interval:
            cur_te = prev_edge[4]
            walk["te"] = [cur_te]

        for step in range(1, L):
            if TR_ls is not None:
                TR = TR_ls[step-1]
            else:
                TR = 'bf'

            if rel_ls is not None:
                rel = int(rel_ls[step-1])
            else:
                rel = None

            next_edge = self.transition_step(
                cur_node, cur_ts, prev_edge, start_node, step, L, cur_te = cur_te, TR = TR, rel=rel,
                start_edge = start_edge, start_ts=start_ts, window=window, flag_time_shifting=flag_time_shifting, ref_time=ref_time
            )

            if len(next_edge):
                cur_node = next_edge[2]
                cur_ts = next_edge[3]
                walk["relations"].append(next_edge[1])
                walk["entities"].append(cur_node)
                walk["ts"].append(cur_ts)

                cur_te = None
                if self.flag_interval:
                    cur_te = next_edge[4]
                    walk["te"].append(cur_te)

                prev_edge = next_edge
            else:  # No valid neighbors (due to temporal or cyclic constraints)
                walk_successful = False
                break

        return walk_successful, walk, ref_time


def store_neighbors(quads):
    """
    Store all neighbors (outgoing edges) for each node.

    Parameters:
        quads (np.ndarray): indices of quadruples

    Returns:
        neighbors (dict): neighbors for each node
    """

    neighbors = dict()
    nodes = list(set(quads[:, 0]))
    for node in nodes:
        neighbors[node] = quads[quads[:, 0] == node]

    return neighbors


def store_edges(quads):
    """
    Store all edges for each relation.

    Parameters:
        quads (np.ndarray): indices of quadruples

    Returns:
        edges (dict): edges for each relation
    """

    edges = dict()
    relations = list(set(quads[:, 1]))
    for rel in relations:
        edges[rel] = quads[quads[:, 1] == rel]

    return edges