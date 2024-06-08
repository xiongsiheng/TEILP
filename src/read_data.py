import json



with open('/home/sxiong45/code/TEILP/output/YAGO/YAGO_stat_res.json', 'r') as file:
    # Load the JSON data into a Python dictionary
    data = json.load(file)


rel_names = ['wasBornIn', 'diedIn', 'worksAt', 'playsFor', 'hasWonPrize', 'isMarriedTo', 'owns', 'graduatedFrom', 'isAffiliatedTo', 'created']
idx = [0, 7, 1, 2, 3, 4, 5, 6, 8, 9]


for k in data['17'].keys():
    elements = k.split(' ')
    translated = []
    for e in elements:
        if e.isdigit():
            if int(e) in idx:
                translated.append(rel_names[idx.index(int(e))])
            else:
                translated.append(rel_names[idx.index(int(e) - len(idx))] + '-1')
        else:
            translated.append(e)
    print(translated)