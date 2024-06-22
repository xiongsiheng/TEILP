import numpy as np
import copy



def read_dataset_txt(path):
    edges = []
    with open(path, 'r') as f:
        lines=f.readlines()
        for l in lines:
            a=l.strip().split()
            a=[int(x) for x in a]
            b=copy.copy(a)
            if len(b)>4:
                b[3] = min(a[3], a[4])
                b[4] = max(a[3], a[4])
            edges.append(b)
    return edges


def obtain_inv_edges(edges, num_rel):
    if isinstance(edges, list):
        edges = np.array(edges)

    edges_ori = edges[edges[:, 1] < (num_rel//2)]
    edges_inv = edges[edges[:, 1] >= (num_rel//2)]
    edges_ori_inv = np.hstack([edges_ori[:, 2:3], edges_ori[:, 1:2] + num_rel//2, edges_ori[:, 0:1], edges_ori[:, 3:]])
    edges_inv_inv = np.hstack([edges_inv[:, 2:3], edges_inv[:, 1:2] - num_rel//2, edges_inv[:, 0:1], edges_inv[:, 3:]])

    edges = np.vstack((edges_ori, edges_inv_inv, edges_ori_inv, edges_inv))

    return edges


def obtain_dataset(dataset_name, num_rel):
    train_edges = read_dataset_txt('../data/' + dataset_name + '/train.txt')
    train_edges = obtain_inv_edges(train_edges, num_rel)

    valid_data = read_dataset_txt('../data/' + dataset_name + '/valid.txt')
    valid_data = np.array(valid_data)
    valid_data_inv = np.hstack([valid_data[:, 2:3], valid_data[:, 1:2] + num_rel//2, valid_data[:, 0:1], valid_data[:, 3:]])

    test_data = read_dataset_txt('../data/' + dataset_name + '/test.txt')
    test_data = np.array(test_data)
    test_data_inv = np.hstack([test_data[:, 2:3], test_data[:, 1:2] + num_rel//2, test_data[:, 0:1], test_data[:, 3:]])

    return train_edges, valid_data, valid_data_inv, test_data, test_data_inv