'''Data Cleaning: Blocking'''
import json
import pandas as pd
import os

workDir = os.path.dirname(os.getcwd())
os.chdir(workDir)

inputjson_Dir = r'D:\0Dataset\POI\POI_Spline1+2_Results\result_cut.json'
output_Dir = r'D:\0Dataset\POI\POI_Spline1+2_Results\Only_Attribute_cut.txt'
# load dataset
print("Loading json...")
with open(inputjson_Dir, 'r', encoding='utf-8') as f:
    data = json.load(f)
#data = pd.DataFrame(list(data.items()))

data = data["data"]
data_length = len(data)
line = ''
with open(output_Dir, 'w', encoding='utf-8') as file:
    count = 0
    for i, name in data.items():
        line += f'{count} {name}；'
        count += 1
    file.writelines(line)
print("write done!")