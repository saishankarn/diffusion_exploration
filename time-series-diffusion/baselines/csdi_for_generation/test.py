import argparse
import torch
import datetime
import json
import yaml
import os

from dataset_electricity import get_dataloader, Electricity_Dataset
from main_model import CSDI_PM25
from utils import train, evaluate

import matplotlib.pyplot as plt
import numpy as np

parser = argparse.ArgumentParser(description="CSDI") 
parser.add_argument("--config", type=str, default="base.yaml")
parser.add_argument('--device', default='cuda:0', help='Device for Attack') 
parser.add_argument("--modelfolder", type=str, default="")
 
args = parser.parse_args()
print(args) 

path = "config/" + args.config 
with open(path, "r") as f:
    config = yaml.safe_load(f)

print(json.dumps(config, indent=4))

train_loader = get_dataloader(
    config["train"]["batch_size"], 
    horizon=config["model"]["horizon"]
)

results_dir = 'save/electricity_20230727_013634'

model = CSDI_PM25(config, args.device, horizon=config["model"]["horizon"]).to(args.device)
model.load_state_dict(torch.load(os.path.join(results_dir, 'model_best.pth')))

num_batches = 10
synthesized_points = torch.zeros((num_batches, config["model"]["target_dim"], config["model"]["horizon"]))
with torch.no_grad():
    for i in range(num_batches):
        print(i)
        output = model.synthesize()
        ts = output.detach().cpu()[0]
        synthesized_points[i] = ts

print(synthesized_points.shape)

dataset = Electricity_Dataset(eval_length=config["model"]["horizon"]).dataset

closest_points = torch.zeros_like(synthesized_points)
for i in range(synthesized_points.shape[0]):
    synthesized = synthesized_points[i]
    min_dist = 10000000000
    for j in range(dataset.shape[0]):
        data = dataset[j]
        dist = torch.cdist(synthesized.type(torch.float32), data.type(torch.float32), 2.0)
        if dist < min_dist:
            min_dist = dist
            closest_points[i] = data

synthesized_points = synthesized_points.squeeze(1).numpy()
closest_points = closest_points.squeeze(1).numpy()

qualitative_results_dir = os.path.join(results_dir, 'qualitative')
os.makedirs(qualitative_results_dir, exist_ok=True)

print(synthesized_points.shape, closest_points.shape)

for i in range(closest_points.shape[0]):
    generated = synthesized_points[i]
    closest = closest_points[i]
    x = np.arange(generated.shape[0])
    plt.plot(x, generated, label='generated')
    plt.plot(x, closest, label='dataset')
    plot_path = os.path.join(qualitative_results_dir, str(i)+'.png')
    leg = plt.legend(loc='upper center')
    plt.savefig(plot_path)
    plt.close()
