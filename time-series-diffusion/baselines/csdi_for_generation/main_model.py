import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from diff_models import diff_CSDI


class CSDI_base(nn.Module):
    def __init__(self, target_dim, horizon, config, device):
        super().__init__()
        self.device = device
        self.target_dim = target_dim
        self.horizon = horizon
 
        self.emb_time_dim = config["model"]["timeemb"] # 128
        self.emb_feature_dim = config["model"]["featureemb"] # 16
        self.is_unconditional = config["model"]["is_unconditional"] # 1
        self.target_strategy = config["model"]["target_strategy"]

        # print(self.emb_time_dim, self.emb_feature_dim)
        self.emb_total_dim = self.emb_time_dim + self.emb_feature_dim
        if self.is_unconditional == False:
            self.emb_total_dim += 1  # for conditional mask
        self.embed_layer = nn.Embedding(
            num_embeddings=self.target_dim, embedding_dim=self.emb_feature_dim 
        )

        config_diff = config["diffusion"]
        config_diff["side_dim"] = self.emb_total_dim 

        input_dim = 1 if self.is_unconditional == True else 2
        self.diffmodel = diff_CSDI(config_diff, input_dim)

        # parameters for diffusion models
        self.num_steps = config_diff["num_steps"]
        if config_diff["schedule"] == "quad":
            self.beta = np.linspace(
                config_diff["beta_start"] ** 0.5, config_diff["beta_end"] ** 0.5, self.num_steps
            ) ** 2
        elif config_diff["schedule"] == "linear":
            self.beta = np.linspace(
                config_diff["beta_start"], config_diff["beta_end"], self.num_steps
            )

        self.alpha_hat = 1 - self.beta
        self.alpha = np.cumprod(self.alpha_hat)
        self.alpha_torch = torch.tensor(self.alpha).float().to(self.device).unsqueeze(1).unsqueeze(1)

    def time_embedding(self, pos, d_model=128):
        pe = torch.zeros(pos.shape[0], pos.shape[1], d_model).to(self.device)
        position = pos.unsqueeze(2)
        div_term = 1 / torch.pow(
            10000.0, torch.arange(0, d_model, 2).to(self.device) / d_model
        )
        pe[:, :, 0::2] = torch.sin(position * div_term)
        pe[:, :, 1::2] = torch.cos(position * div_term)
        return pe

    def get_side_info(self, observed_tp, cond_mask):
        B, K, L = cond_mask.shape
        # print(B,K,L)

        time_embed = self.time_embedding(observed_tp, self.emb_time_dim)  # (B,L,emb)
        time_embed = time_embed.unsqueeze(2).expand(-1, -1, K, -1)
        feature_embed = self.embed_layer(
            torch.arange(self.target_dim).to(self.device)
        )  # (K,emb)
        feature_embed = feature_embed.unsqueeze(0).unsqueeze(0).expand(B, L, -1, -1)
        # print(time_embed.shape, feature_embed.shape)

        side_info = torch.cat([time_embed, feature_embed], dim=-1)  # (B,L,K,*)
        # print(side_info.shape)
        side_info = side_info.permute(0, 3, 2, 1)  # (B,*,K,L)
        # print(side_info.shape)

        return side_info

    def calc_loss(
        self, observed_data, side_info, is_train, set_t=-1
    ):
        B, K, L = observed_data.shape # B is the batch size, K is the number of features for each time step, L is the total no of time steps
        if is_train != 1:  # for validation
            t = (torch.ones(B) * set_t).long().to(self.device)
        else:
            t = torch.randint(0, self.num_steps, [B]).to(self.device) # randomly sampling noise levels for each element in the batch 
        current_alpha = self.alpha_torch[t]  # (B,1,1)

        noise = torch.randn_like(observed_data)
        noisy_data = (current_alpha ** 0.5) * observed_data + (1.0 - current_alpha) ** 0.5 * noise
        # print(noisy_data.shape, observed_data.shape, cond_mask.shape)

        total_input = noisy_data.unsqueeze(1) # process the input before forward pass to the neural network

        predicted = self.diffmodel(total_input, side_info, t)  # (B,K,L)
        # print(predicted.shape, observed_data.shape)

        loss = F.mse_loss(predicted, noise)
        return loss

    def synthesize(self):
        B = 1
        K = self.target_dim
        L = self.horizon

        sample_output = torch.zeros(B,K,L).to(self.device) 

        timepoints = torch.tensor(np.arange(self.horizon)).to(self.device).unsqueeze(0)
        side_info = self.get_side_info(timepoints, sample_output)
        synthesized_output = torch.randn_like(sample_output) # step 1

        for tidx in reversed(range(self.num_steps)):            
            input = synthesized_output.unsqueeze(1)
            predicted_noise = self.diffmodel(input, side_info, torch.tensor([tidx]).to(self.device))

            coeff1 = 1 / self.alpha_hat[tidx] ** 0.5
            coeff2 = (1 - self.alpha_hat[tidx]) / (1 - self.alpha[tidx]) ** 0.5

            synthesized_output = coeff1 * (synthesized_output - coeff2 * predicted_noise)

            if tidx > 0:
                noise = torch.randn_like(sample_output)
                sigma = (
                    (1.0 - self.alpha[tidx - 1]) / (1.0 - self.alpha[tidx]) * self.beta[tidx]
                ) ** 0.5
                synthesized_output += sigma * noise

        return synthesized_output

    def forward(self, batch, is_train=1):
        observed_data = batch["observed_data"].to(self.device).float()
        observed_tp = batch["timepoints"].to(self.device).float()

        side_info = self.get_side_info(observed_tp, observed_data)
        # print(side_info.shape, observed_data.shape)

        loss_func = self.calc_loss if is_train == 1 else self.calc_loss_valid

        return loss_func(observed_data, side_info, is_train)

class CSDI_PM25(CSDI_base):
    def __init__(self, config, device, horizon=36, target_dim=1):
        super(CSDI_PM25, self).__init__(target_dim, horizon, config, device)

    def process_data(self, batch):
        observed_data = batch["observed_data"].to(self.device).float()
        observed_mask = batch["observed_mask"].to(self.device).float()
        observed_tp = batch["timepoints"].to(self.device).float()
        gt_mask = batch["gt_mask"].to(self.device).float()
        cut_length = batch["cut_length"].to(self.device).long()
        for_pattern_mask = batch["hist_mask"].to(self.device).float()

        observed_data = observed_data.permute(0, 2, 1)
        observed_mask = observed_mask.permute(0, 2, 1)
        gt_mask = gt_mask.permute(0, 2, 1)
        for_pattern_mask = for_pattern_mask.permute(0, 2, 1)

        return (
            observed_data,
            observed_mask,
            observed_tp,
            gt_mask,
            for_pattern_mask,
            cut_length,
        )