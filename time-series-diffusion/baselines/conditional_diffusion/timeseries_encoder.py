import torch
import torch.nn as nn
from model import CSDIEncoder, CSDIDecoder


class TimeSeriesEncoder(nn.Module):
    def __init__(self, model_config, device):
        super().__init__()
        self.horizon = model_config.max_seq_len
        self.emb_time_dim = model_config.dim_val
        self.device = device

        input_dim = 1
        self.encoder = CSDIEncoder(model_config, input_dim)
        self.decoder = CSDIDecoder(model_config, input_dim)

    def time_embedding(self, pos, d_model=128):
        pe = torch.zeros(pos.shape[0], pos.shape[1], d_model).to(self.device)
        position = pos.unsqueeze(2)
        div_term = 1 / torch.pow(10000.0, torch.arange(0, d_model, 2).to(self.device) / d_model)
        pe[:, :, 0::2] = torch.sin(position * div_term)
        pe[:, :, 1::2] = torch.cos(position * div_term)
        return pe

    def get_side_info(self, observed_tp, cond_mask):
        B, K, L = cond_mask.shape
        time_embed = self.time_embedding(observed_tp, self.emb_time_dim)  # (B,L,emb)
        time_embed = time_embed.unsqueeze(2).expand(-1, -1, K, -1) # (B, L, K, emb)
        side_info = time_embed.permute(0, 3, 2, 1)  # (B,*,K,L)
        return side_info
    
    def reparameterization(self, mean, var):
        epsilon = torch.randn_like(var)        
        z = mean + var*epsilon
        return z

    def forward(self, batch):
        observed_data = batch["observed_data"].to(self.device).float()
        observed_tp = batch["timepoints"].to(self.device).float()
        side_info = self.get_side_info(observed_tp, observed_data)
        total_input = observed_data.unsqueeze(1) # process the input before forward pass to the neural network
        encoded = self.diffmodel(total_input, side_info)  # (B,K,L)
        return encoded, pred