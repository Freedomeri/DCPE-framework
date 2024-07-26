'''Data Cleaning: Blocking'''
import json
#import recordlinkage as rl
import pandas as pd

#json_Dir = r'../prediction_result/result_myTest.json'
json_Dir = r'D:\0Dataset\POI\POI_Spline1+2_Results\result_cut.json'
outputClean_Dir = r'D:\0Dataset\POI\POI_Spline1+2_Results\result_cut_clean.json'

# set logging
#rl.logging.set_verbosity(rl.logging.INFO)


def check_similarity(left, right, similarity):
    set_length = [0, 0]
    set1 = set(left)
    set2 = set(right)
    set_length[0] = len(set1)
    set_length[1] = len(set2)
    intersection = len(set1.intersection(set2))
    shortOne = 0 if set_length[0] < set_length[1] else 1
    if set_length[0] != 0 and set_length[1] != 0:
        percentage = intersection / set_length[shortOne]
    else:
        percentage = 1
    if percentage > similarity:
        similar = 1
    else:
        similar = 0
    return similar, shortOne
def drop_similar(_data, similarity):
    data_length = len(_data)
    remove_list = [0] * data_length
    for i in range(data_length):
        for j in range(i + 1, i + 50 if (i + 50) < data_length else data_length):
            similarity_result, shortOne = check_similarity(_data[1][i], _data[1][j], similarity)
            if similarity_result == 1:
                remove_item = i if shortOne == 0 else j
                remove_list[remove_item] = 1
    for k in range(data_length - 1, -1, -1):
        if remove_list[k] == 1:
            _data = _data.drop(k)
    return _data
# load dataset
print("Loading data...")
with open(json_Dir, 'r', encoding='utf-8') as f:
    data = json.load(f)
    data = data["data"]
data = pd.DataFrame(list(data.items()))

data1 = drop_similar(data, similarity=0.7)
data1 = data1.reset_index()
#'''clean twice'''
#data2 = drop_similar(data1, similarity=0.8)
del data1['index']

'''generate json format'''
data2_json = {}
data2 = data1.values.tolist()
for i in range(len(data2)):
    data2_json[data2[i][0]] = data2[i][1]
data2_json_new = {"data": data2_json}

print("Clean Done")
json.dump(data2_json_new, open(outputClean_Dir, "w+", encoding="utf-8"), ensure_ascii=False, indent=1)