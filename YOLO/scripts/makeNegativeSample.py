import os
import glob
from pathlib import Path

workDir = os.path.dirname(os.getcwd())
os.chdir(workDir)

input_path = r"C:\Users\lc\Pictures\0Sun\FourDInternet\UE5Capture1\*.jpg"
target_txt_path = "./datasets/labels/finetuning/"

files = glob.glob(input_path)
for file in files:
    name = Path(file).stem
    with open(os.path.join(target_txt_path, '{}.txt'.format(name)), 'w', encoding='utf-8') as f:
        f.write('')