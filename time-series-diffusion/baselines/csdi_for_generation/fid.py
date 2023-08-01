import numpy as np 
from numpy import cov
from numpy import trace
from numpy import iscomplexobj
from numpy.random import random
from scipy.linalg import sqrtm
import argparse
import torch
import datetime
import json
import yaml
import os

from dataset_electricity import get_dataloader, Electricity_Dataset
from main_model import CSDI_PM25 
from utils import train, evaluate

parser = argparse.ArgumentParser(description="CSDI") 
parser.add_argument("--config", type=str, default="base.yaml")
parser.add_argument('--device', default='cuda:0', help='Device for Attack') 
parser.add_argument("--modelfolder", type=str, default="")
parser.add_argument("--horizon", type=int, default=48, help="synthesis horizon")
 
args = parser.parse_args()
print(args) 

path = "config/" + args.config 
with open(path, "r") as f:
    config = yaml.safe_load(f)

print(json.dumps(config, indent=4))

results_dir = 'save/electricity_20230727_013634'

model = CSDI_PM25(config, args.device, horizon=config["model"]["horizon"]).to(args.device)
model.load_state_dict(torch.load(os.path.join(results_dir, 'model_best.pth')))

model.eval()

num_batches = 1000
synthesized_points = torch.zeros((num_batches, config["model"]["target_dim"], config["model"]["horizon"]))
with torch.no_grad():
    for i in range(num_batches):
        print(i)
        output = model.synthesize()
        ts = output.detach().cpu()[0]
        synthesized_points[i] = ts

synthesized_points = synthesized_points.squeeze(1).numpy()

dataset = Electricity_Dataset(eval_length=config["model"]["horizon"]).dataset
dataset_points = dataset.squeeze(1).numpy()

print(synthesized_points.shape, dataset_points.shape)

def calculate_fid(act1, act2):
    mu1, sigma1 = act1.mean(axis=0), cov(act1, rowvar=False)
    mu2, sigma2 = act2.mean(axis=0), cov(act2, rowvar=False)
    ssdiff = np.sum((mu1 - mu2)**2.0)
    covmean = sqrtm(sigma1.dot(sigma2))
    if iscomplexobj(covmean):
        covmean = covmean.real
    
    fid = ssdiff + trace(sigma1 + sigma2 - 2.0 * covmean)
    return fid
 
# define two collections of activations
act1 = dataset_points
act2 = synthesized_points
# fid between act1 and act1
fid = calculate_fid(act1, act1)
print('FID (same): %.3f' % fid)
# fid between act1 and act2
fid = calculate_fid(act1, act2)
print('FID (different): %.3f' % fid)