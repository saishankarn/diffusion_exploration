import os 
os.system("unset LD_LIBRARY_PATH")
import torch
from torch.utils.data import DataLoader

from config import DatasetConfig, VAEConfig
from dataset import GetDataset, generate_dataset_obj
from autoencoder import TimeSeriesVariationalAutoEncoder

def encode_sinusoidal(autoencoder, dataloader):
    autoencoder.train()
    mean_dataset = []
    log_var_dataset = []
    labels_dataset = []
    ts_dataset = []
    with torch.no_grad():
        for val_batch in dataloader:
            mean, log_var, _ = model(val_batch)
            mean_dataset.append(mean.unsqueeze(1))
            log_var_dataset.append(log_var.unsqueeze(1))
            labels_dataset.append(val_batch["labels"])
            ts_dataset.append(val_batch["observed_data"])

    mean_dataset = torch.cat(mean_dataset, 0)
    log_var_dataset = torch.cat(log_var_dataset, 0)
    labels_dataset = torch.cat(labels_dataset, 0)
    ts_dataset = torch.cat(ts_dataset, 0)
    print(mean_dataset.shape, log_var_dataset.shape)

    return mean_dataset, log_var_dataset, labels_dataset, ts_dataset

if __name__ == "__main__":
    dataset_config = DatasetConfig() 
    model_config = VAEConfig()

    sine_dataset_obj = generate_dataset_obj(dataset_config=dataset_config)
    train_dataset = GetDataset(dataset_obj=sine_dataset_obj, dataset_config=dataset_config, train=True)
    train_dataloader = DataLoader(train_dataset, batch_size=model_config.train_batch_size, num_workers=0, shuffle=True)
    test_dataset = GetDataset(dataset_obj=sine_dataset_obj, dataset_config=dataset_config, train=False)    
    val_dataloader = DataLoader(test_dataset, batch_size=model_config.val_batch_size, num_workers=0, shuffle=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = TimeSeriesVariationalAutoEncoder(model_config=model_config, device=device)
    model = model.to(device)
    if model_config.pretrained_dir != '':
        pretrained_loc = os.path.join(model_config.pretrained_dir, 'results/model_best.pth')
        model.load_state_dict(torch.load(pretrained_loc))

    mean_train_dataset, log_var_train_dataset, labels_train_dataset, ts_train_dataset = encode_sinusoidal(autoencoder=model, dataloader=train_dataloader)
    mean_val_dataset, log_var_val_dataset, labels_val_dataset, ts_val_dataset = encode_sinusoidal(autoencoder=model, dataloader=val_dataloader)
    
    # storing the train dataset
    mean_train_dataset_loc = os.path.join(model_config.pretrained_dir, 'results/mean_train.pth')
    torch.save(mean_train_dataset, mean_train_dataset_loc)
    log_var_train_dataset_loc = os.path.join(model_config.pretrained_dir, 'results/log_var_train.pth')
    torch.save(log_var_train_dataset, log_var_train_dataset_loc)
    labels_train_dataset_loc = os.path.join(model_config.pretrained_dir, 'results/labels_train.pth')
    torch.save(labels_train_dataset, labels_train_dataset_loc)
    ts_train_dataset_loc = os.path.join(model_config.pretrained_dir, 'results/ts_train.pth')
    torch.save(ts_train_dataset, ts_train_dataset_loc)

    # storing the val dataset
    mean_val_dataset_loc = os.path.join(model_config.pretrained_dir, 'results/mean_val.pth')
    torch.save(mean_val_dataset, mean_val_dataset_loc)
    log_var_val_dataset_loc = os.path.join(model_config.pretrained_dir, 'results/log_var_val.pth')
    torch.save(log_var_val_dataset, log_var_val_dataset_loc)
    labels_val_dataset_loc = os.path.join(model_config.pretrained_dir, 'results/labels_val.pth')
    torch.save(labels_val_dataset, labels_val_dataset_loc)
    ts_val_dataset_loc = os.path.join(model_config.pretrained_dir, 'results/ts_val.pth')
    torch.save(ts_val_dataset, ts_val_dataset_loc)


    

