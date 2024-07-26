import torch
import numpy as np
from cv2 import imread
zerodepth_model = torch.hub.load("TRI-ML/vidar", "ZeroDepth", pretrained=True, trust_repo=True)

intrinsics = torch.tensor(np.load('examples/ddad_intrinsics.npy')).unsqueeze(0)
rgb = torch.tensor(imread('examples/ddad_sample.png')).permute(2,0,1).unsqueeze(0)/255.

depth_pred = zerodepth_model(rgb, intrinsics)