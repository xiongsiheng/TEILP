import sys
import numpy as np
import json


# dataset = 'YAGO11k'
# dataset_short_name = 'YAGO'

dataset = 'WIKIDATA12k'
dataset_short_name = 'wiki'



filename_path = '../data/' + dataset + '_ori/'
content = []
for mode in ['train', 'valid', 'test']:
    with open(filename_path + mode + '.txt') as file:
        content += file.readlines()


valid_int_id = []
data = ''
for (i, line) in enumerate(content):
    parts = line.split('\t')
    if parts[3][0] == '-':
        parts[3] = parts[3][1:]
    if parts[4][0] == '-':
        parts[4] = parts[4][1:]

    if '#' in parts[3].split('-')[0] or '#' in parts[4].split('-')[0]:
        continue

    valid_int_id.append(i)
    data += line


print(data)
print(len(valid_int_id))



filename_path = '../data/'+ dataset +'/'
content_processed = []
num_samples_dist = []
for mode in ['train', 'valid', 'test']:
    with open(filename_path + mode + '.txt') as file:
        lines = file.readlines()
        content_processed += lines
        num_samples_dist.append(len(lines))

content_processed = [[int(num) for num in line.strip().split('\t')] for line in content_processed]

edges = np.array(content_processed)
idx_ls = np.array(range(len(edges))).reshape((-1,1))
edges = np.hstack((edges, idx_ls))
# print(edges)

# edges = edges[edges[:, 3].argsort()]
edges = edges[np.lexsort((edges[:, -1], edges[:, 3]))]


filename_path = '../data/difficult_settings/'+ dataset +'_time_shifting/'
num_samples_dist = []
for mode in ['train', 'valid', 'test']:
    with open(filename_path + mode + '.txt') as file:
        lines = file.readlines()
        num_samples_dist.append(len(lines))


edges = edges[-num_samples_dist[2]:, :]

print(num_samples_dist)
print(edges)

# sys.exit()

rm_ls = []
for (i, line) in enumerate(edges):
    if not line[-1] in valid_int_id:
        rm_ls.append(i)


print(rm_ls)
print(len(rm_ls))

with open('../data/'+ dataset_short_name +'_time_pred_eval_rm_idx_shift_mode.json', 'w') as json_file:
    json.dump(rm_ls, json_file)