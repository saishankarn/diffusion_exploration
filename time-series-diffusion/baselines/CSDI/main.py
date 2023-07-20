import os
import json
import torch
import argparse
import datetime

# custom imports 
# from csdi import CSDIDiff
from dataset import get_dataloader_electricity
# from training import train
# from testing import evaluate

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config.json', \
                        help='JSON file for configuration')
    parser.add_argument('--device', default='cuda:0', \
                        help='Device for Attack')
    parser.add_argument("--modelfolder", type=str, default="")
    parser.add_argument("--nsample", type=int, default=100)

    args = parser.parse_args()

    with open(args.config) as f:
        data = f.read() 

    config = json.loads(data)
    print(config)

    print(json.dumps(config, indent=4))

    current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S") 
    foldername = ("./results/electricity" + "_" + current_time + "/")

    print('model folder:', foldername)
    os.makedirs(foldername, exist_ok=True)
    with open(foldername + "config.json", "w") as f:
        json.dump(config, f, indent=4)

    train_loader = get_dataloader_electricity(config["train"]["batch_size"], \
                    config["model"]["ts_dim"], \
                    config["train"]["batch_size"], \
                    device=args.device
                )
    
    # model = CSDIDiff(config, args.device).to(args.device)

    # if args.modelfolder == "":
    #     train(model, \
    #         config["train"], \
    #         train_loader, \
    #         valid_loader=valid_loader, \
    #         foldername=foldername
    #     )
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