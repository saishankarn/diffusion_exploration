import os 
os.system("unset LD_LIBRARY_PATH")
import tqdm
import torch
from torch.utils.data import Dataset

class GetDiffusionDataset(Dataset):
    def __init__(self, train_dataloader, vae, device):
        self.dataloader = train_dataloader
        self.model = vae 
        self.device = device 
        self.mean_dataset, self.log_var_dataset = self.get_latents()

    def get_latents(self):
        self.model.eval()
        mean_dataset = []
        log_var_dataset = []
        print("obtaining the latents")

        for train_batch in self.dataloader:
            mean, log_var, _ = self.model(train_batch)
                    
        dataset = torch.cat(dataset, 0)
        return dataset
    
    def __getitem__(self, org_index):
        return self.dataset[org_index]

    def __len__(self):
        return self.dataset.shape[0]