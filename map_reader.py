import json
import os
import pickle


def load_data(data_file, data):
    with open(data_file, 'r', buffering=1) as f:
        final_dict = data
        curr_list = []
        prev = None
        for line in f:
            ex = json.loads(line)
            if prev is None:
                prev = ex["title"]
            if ex["title"] != prev:
                insert_dict = dict()
                insert_list = []
                for c in curr_list:
                    if c[0] in insert_dict:
                        insert_dict[c[0]] = max(insert_dict[c[0]], c[1])
                    else:
                        insert_dict[c[0]] = c[1]
                for item in insert_dict:
                    insert_list.append([item,insert_dict[item]])
                insert_list.sort(key = lambda x: -x[1])
                if len(insert_list)>20:
                    insert_list = insert_list[:20]
                final_dict[prev] = insert_list
                curr_list = []
                prev = ex["title"]
            for i in ex["anchored_et"]:
                curr_list.append([i[0], i[3]])
    return final_dict


# Doing for whole folder
data = dict()
path_to_json = 'map_data/enwiki_1101_anchor/AA/'
json_files = [pos_json for pos_json in os.listdir(path_to_json)]
for name in json_files:
    print("In progress: ", name)
    data = load_data(path_to_json + name, data)
path_to_json = 'map_data/enwiki_1101_anchor/AB/'
json_files = [pos_json for pos_json in os.listdir(path_to_json)]
for name in json_files:
    print("In progress: ", name)
    data = load_data(path_to_json + name, data)

with open('entity_link_score.pickle', 'wb') as handle:
    pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open('entity_link_dict.json', 'w') as f:
    json.dump(data, f)
