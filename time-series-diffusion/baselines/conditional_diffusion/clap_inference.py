import os
import numpy as np

import torch
from torch.utils.data import DataLoader  

from clap_transformer import Sinusoidal, GetDataset, CLTSPModel, CLTSPConfig, DatasetConfig 

dataset_config = DatasetConfig() 
cltsp_config = CLTSPConfig()
cltsp_config.pretrained_loc = 'save/sine_20230814_110133/results/model_best.pth'

sine_dataset_obj = Sinusoidal(config=dataset_config)
train_dataset = GetDataset(data=sine_dataset_obj.train_data, labels=sine_dataset_obj.train_labels)
test_dataset = GetDataset(data=sine_dataset_obj.test_data, labels=sine_dataset_obj.test_labels)

train_dataloader = DataLoader(train_dataset, batch_size=cltsp_config.train_batch_size, num_workers=0, shuffle=True)
val_dataloader = DataLoader(test_dataset, batch_size=cltsp_config.val_batch_size, num_workers=0, shuffle=True)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = CLTSPModel(model_config=cltsp_config, device=device)
model = model.to(device)
if cltsp_config.pretrained_loc != '':
    model.load_state_dict(torch.load(cltsp_config.pretrained_loc))

model.eval()
val_labels = sine_dataset_obj.test_labels
val_labels = val_labels.to(device)

val_parameters_latent = model.parameters_encoder(val_labels)
num_labels = val_parameters_latent.shape[0]
top1_acc = 0
top3_acc = 0
top5_acc = 0
            
for batch_no, val_batch in enumerate(val_dataloader):
    print("------------------------------------------------------------------------------------------")
    print("sample no : %d"%batch_no)
    seq_true = val_batch["observed_data"].to(device)
    seq_pred, timeseries_latent, parameters_latent = model(val_batch)

    scores = torch.matmul(val_parameters_latent, timeseries_latent.T).squeeze(-1)
    query_label = val_batch["labels"][0].to(device)
    print("query label : ", query_label)

    # top 1
    # print("top 1")
    top1_indices = torch.topk(scores, k=1).indices
    top1_corresponding_labels = val_labels[top1_indices]
    print("most similar label : ")
    print(top1_corresponding_labels)
                    
    query_label_exists_in_top1 = torch.any(torch.all(top1_corresponding_labels == query_label, dim=1))
    top1_acc += query_label_exists_in_top1

    # top 3
    top3_indices = torch.topk(scores, k=3).indices
    top3_corresponding_labels = val_labels[top3_indices]
    print("most similar three labels : ")
    print(top3_corresponding_labels)

    query_label_exists_in_top3 = torch.any(torch.all(top3_corresponding_labels == query_label, dim=1))
    top3_acc += query_label_exists_in_top3
                
    # top 5
    top5_indices = torch.topk(scores, k=5).indices
    top5_corresponding_labels = val_labels[top5_indices]
    print("most similar five labels : ") 
    print(top5_corresponding_labels)

    query_label_exists_in_top5 = torch.any(torch.all(top5_corresponding_labels == query_label, dim=1))
    top5_acc += query_label_exists_in_top5

top1_acc = top1_acc / num_labels
top3_acc = top3_acc / num_labels
top5_acc = top5_acc / num_labels
print("top1 accuracy = %f, top3 accuracy = %f,  top5 accuracy = %f"%(top1_acc, top3_acc, top5_acc))