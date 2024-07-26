import torch
import numpy as np
from cv2 import imread
from PIL import Image
from vidar.core.evaluator import Evaluator
from vidar.utils.config import read_config
from vidar.utils.setup import setup_arch, setup_network

#zerodepth_model = torch.hub.load("TRI-ML/vidar", "ZeroDepth", pretrained=True, trust_repo=True)
cfg = read_config("./configs/papers/zerodepth/hub_zerodepth.yaml")
model = Evaluator(cfg)
model = setup_network(cfg.networks.perceiver)
model.eval()

model_path = "./models/ZeroDepth_unified.ckpt"
state_dict = torch.load(model_path, map_location="cuda:0")
state_dict =  {k.replace("module.networks.define.", ""): v for k, v in state_dict["state_dict"].items()}
model.load_state_dict(state_dict, strict=True)
zerodepth_model = model


intrinsics = torch.tensor(np.load('examples/ddad_intrinsics.npy')).unsqueeze(0)
image = Image.open('./117.124368,36.686269l-30h85.jpg')
image = image.resize((640,360))
image = np.array(image)
rgb = torch.tensor(image).permute(2,0,1).unsqueeze(0)/255.

depth_pred = zerodepth_model(rgb, intrinsics)
np_depth_pred = depth_pred.detach().numpy()
new_np_depth_pred = np.resize(np_depth_pred,(720,1280))  #resize to origin scale