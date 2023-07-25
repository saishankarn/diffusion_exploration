import argparse
import torch
import datetime
import json
import yaml
import os

from dataset_electricity import get_dataloader
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

current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S") 
foldername = ("./save/electricity" + "_" + current_time + "/")

print('model folder:', foldername)
os.makedirs(foldername, exist_ok=True)
with open(foldername + "config.json", "w") as f:
    json.dump(config, f, indent=4)

train_loader = get_dataloader(
    config["train"]["batch_size"], 
    horizon=args.horizon
)
model = CSDI_PM25(config, args.device, horizon=args.horizon).to(args.device)
# print(model.embed_layer)

if args.modelfolder == "":
    train(
        model,
        config,
        train_loader,
        foldername=foldername,
)
# else:
#     model.load_state_dict(torch.load("./save/" + args.modelfolder + "/model.pth"))

# evaluate(
#     model,
#     test_loader,
#     nsample=args.nsample,
#     scaler=scaler,
#     mean_scaler=mean_scaler,
#     foldername=foldername,
# )