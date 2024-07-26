import os

from ultralytics import YOLO
from PIL import Image
import numpy as np
from pathlib import Path
from ultralytics.engine.results import Results
from ultralytics.utils.plotting import save_one_box

import torch
#from zoedepth.models.builder import build_model
#from zoedepth.utils.config import get_config

import re
import math
import argparse

from vidar.core.evaluator import Evaluator, sample_to_cuda
from vidar.utils.config import read_config
from vidar.utils.setup import setup_arch, setup_network

#os.environ["CUDA_VISIBLE_DEVICES"] = "1"



#directory
model_dir = r'models/best_detect_x.onnx'


imgW = 1280
imgH = 720
scale_factor_longi = 0.00005
scale_factor_lati = 0.00004

def compute_real_coord(xywh, depthArray, direction, pose, origin_coord):
    origin_longi, origin_lati = origin_coord.split(',')
    origin_longi = float(origin_longi)
    origin_lati = float(origin_lati)
    depth = depthArray[int(xywh[0][1])][int(xywh[0][0])]
    real_depth = (1.08 ** depth)-0.3
    #real_depth = (1.015 ** depth - 1.08) * depth if depth < 40 else (1.004 ** depth - 1) * depth
    offset = float((imgW/2 - xywh[0][0]) / imgW) * 1.6
    radians = math.radians(int(pose))
    if direction[0] == 'l':
        longi_center = math.sin(radians)
        lati_center = math.cos(radians)
        longi_offset = -1 * math.cos(radians) * offset
        lati_offset = math.sin(radians) * offset
        longitude = real_depth * scale_factor_longi * (longi_center + longi_offset) + origin_longi
        latitude =  real_depth * scale_factor_lati * (lati_center + lati_offset) + origin_lati
    else:
        longi_center = -1 * math.sin(radians)
        lati_center = -1 * math.cos(radians)
        longi_offset = math.cos(radians) * offset
        lati_offset = -1 * math.sin(radians) * offset
        longitude = real_depth * scale_factor_longi * (longi_center + longi_offset) + origin_longi
        latitude =  real_depth * scale_factor_lati * (lati_center + lati_offset) + origin_lati
    coord = str(round(longitude, 6)) + ',' + str(round(latitude, 6))
    return coord

def save_crop_as_coord(result, direction, pose, depth, height, save_dir, coordinate):
    for d in result.boxes:
        real_coord = compute_real_coord(d.xywh,depth,direction,pose,coordinate)
        save_one_box(d.xyxy,
                     result.orig_img.copy(),
                     file=Path(save_dir) / f'{real_coord}_{height}.jpg',
                     BGR=True)


if __name__ == '__main__':
    # predictImg_dir = r'C:\Users\lc\Pictures\0Sun\四维互联网\POI_PictureData\Spline1'
    # croppedImg_dir = r'C:\Users\lc\Pictures\0Sun\四维互联网\POI_PictureData\Spline1_cropped'
    parser = argparse.ArgumentParser(description='Detect')
    parser.add_argument('--input',default="input", type=str,help ='input dir')
    parser.add_argument('--output', default="output", type=str, help='result dir')
    args = parser.parse_args()
    predictImg_dir = args.input
    croppedImg_dir = args.output

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    '''Load Image Detection model: YOLO'''
    detect_model = YOLO(model_dir, task='detect')  # load a custom model
    #print(model.names)
    #detect_model.to(DEVICE)

    '''# Load Depth Model: ZoeD_K'''
    #conf = get_config("zoedepth", "infer", config_version="kitti",
    #                  pretrained_resource="local::./zoedepth/models/ZoeD_M12_K.pt")
    #model_zoe_n = build_model(conf)
    #zoe = model_zoe_n.to(DEVICE)
    ''' Load Depth Model: ZeroDepth_outdoor'''
    cfg = read_config("./vidar/configs/papers/zerodepth/hub_zerodepth.yaml")
    model = Evaluator(cfg)
    model = setup_network(cfg.networks.perceiver)
    model.eval()
    zerodepth_model = model.to(DEVICE)
    #zerodepth_model = model

    model_path = "./vidar/models/ZeroDepth_unified.ckpt"
    state_dict = torch.load(model_path, map_location="cuda:0")
    state_dict = {k.replace("module.networks.define.", ""): v for k, v in state_dict["state_dict"].items()}
    zerodepth_model.load_state_dict(state_dict, strict=True)


    intrinsics = torch.tensor(np.load('vidar/examples/ddad_intrinsics.npy')).unsqueeze(0)
    intrinsics = sample_to_cuda(intrinsics,0)

    '''#save the name of every image'''
    imgNames = []
    if os.path.isdir(predictImg_dir):
        imgFile = os.listdir(predictImg_dir)
        for fileName in imgFile:
            name, type = os.path.splitext(fileName)
            if type == ('.jpg' or '.png'):
                imgNames.append(name)
    else:
        imgNames.append(Path(predictImg_dir).stem)


    '''# Predict with the model'''
    results = detect_model.predict(predictImg_dir,classes=[0], imgsz=1280, device = 'cuda:0' , stream = True)  # predict on an image


    '''crop and save every Advertisement image'''
    pattern = '[lr]'
    pattern2 = '[h]'
    n = 0
    for r in results:
        #image1 = Image.open(predictImg_dir + '/' + '0000000005' + '.jpg')
        #image = np.array(image1)
        image = Image.open(predictImg_dir+ '/'+ imgNames[n]+'.jpg').convert(
            "RGB")  # load
        image = image.resize((640, 360))
        image = np.array(image)
        rgb = torch.tensor(image).permute(2, 0, 1).unsqueeze(0) / 255.
        rgb = sample_to_cuda(rgb, 0)
        with torch.no_grad():
            depth_pred = zerodepth_model(rgb, intrinsics)
        np_depth_pred = depth_pred.cpu().numpy()
        depth = np.resize(np_depth_pred, (720, 1280))  # resize to origin scale
        #depth = zoe.infer_pil(image)  # as numpy
        direction = re.findall(pattern, imgNames[n])
        height_letter = re.findall(pattern2, imgNames[n])
        height_index = imgNames[n].find(height_letter[0])
        direction_index = imgNames[n].find(direction[0])
        pose = imgNames[n][direction_index+1:height_index]
        height = imgNames[n][height_index+1:]
        origin_coord = imgNames[n][:direction_index]
        save_crop_as_coord(r, direction = direction, pose = pose, depth = depth, height = height, save_dir=croppedImg_dir, coordinate=origin_coord)
        n = n + 1
