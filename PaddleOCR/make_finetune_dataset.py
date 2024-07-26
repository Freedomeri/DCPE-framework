from paddleocr import PaddleOCR, draw_ocr
import os
from pathlib import Path
from PIL import Image
import json
import argparse

parser = argparse.ArgumentParser(description='OCR')
parser.add_argument('--input', default="input", type=str, help='input dir')
parser.add_argument('--output', default="output", type=str, help='result dir')
args = parser.parse_args()
sourceImg_dir = args.input
outputImg_dir = args.output

# sourceImg_dir = r'D:\0PythonProjects\YOLO\datasets\images\cropped\Aoti_x'
# outputImg_dir = r'.\outputOCR\result_Aoti_x.json'
det_model_dir = r'.\models\ch_PP-OCRv4_det_server_infer'
rec_model_dir = r'.\models\ch_PP-OCRv4_rec_server_infer'
cls_model_dir = r'.\models\ch_ppocr_mobile_v2.0_cls_slim_infer'

# Paddleocr目前支持的多语言语种可以通过修改lang参数进行切换
# 例如`ch`, `en`, `fr`, `german`, `korean`, `japan`
ocr = PaddleOCR(use_angle_cls=True, lang="ch", det_model_dir = det_model_dir, rec_model_dir = rec_model_dir, cls_model_dir = cls_model_dir, use_gpu = True)  # need to run only once to download and load model into memory

imgNames = []
imgFiles = []
if os.path.isdir(sourceImg_dir):
    imgFiles = os.listdir(sourceImg_dir)
    for fileName in imgFiles:
        name, type = os.path.splitext(fileName)
        if type == ('.jpg' or '.png'):
            imgNames.append(name)
else:
    imgNames.append(Path(sourceImg_dir).stem)
json_file = {}
json_data = {}

lines = ''
with open(outputImg_dir, 'w', encoding='utf-8') as file:
    for n in range(len(imgNames)):
        detect_None = False
        #print(Path(sourceImg_dir)/f'{imgFiles[n]}')
        result = ocr.ocr(str(Path(sourceImg_dir)/f'{imgFiles[n]}'), cls=True)
        for idx in range(len(result)):
            res = result[idx]
            if res is not None:
                for line in res:
                    print(line)
            else:
                detect_None = True
        if not detect_None:
            # 显示结果
            result = result[0]
            txts = [line[1][0] for line in result]
            lines += f'{sourceImg_dir}/{imgFiles[n]}\t{"".join(txts)}\n'
        else:
            lines += f'{sourceImg_dir}/{imgFiles[n]}\n'

    file.writelines(lines)
print('write done!')