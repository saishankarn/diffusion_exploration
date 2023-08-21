import math
import numpy as np 
import torch
import torch.nn as nn
import torch.nn.functional as F

def get_torch_trans(heads=8, layers=1, channels=64):
    encoder_layer = nn.TransformerEncoderLayer(d_model=channels, nhead=heads, dim_feedforward=64, activation="gelu")
    return nn.TransformerEncoder(encoder_layer, num_layers=layers)

def Conv1d_with_init(in_channels, out_channels, kernel_size):
    layer = nn.Conv1d(in_channels, out_channels, kernel_size)
    nn.init.kaiming_normal_(layer.weight)
    return layer

class DiffusionEmbedding(nn.Module):  
    def __init__(self, num_steps, embedding_dim=128, projection_dim=None):
        super().__init__()
        if projection_dim is None:
            projection_dim = embedding_dim
        self.register_buffer(
            "embedding",
            self._build_embedding(num_steps, embedding_dim / 2),
            persistent=False,
        ) 
        self.projection1 = nn.Linear(embedding_dim, projection_dim)
        self.projection2 = nn.Linear(projection_dim, projection_dim)

    def forward(self, diffusion_step):
        x = self.embedding[diffusion_step]
        x = self.projection1(x)
        x = F.silu(x)
        x = self.projection2(x)
        x = F.silu(x)
        return x

    def _build_embedding(self, num_steps, dim=64):
        steps = torch.arange(num_steps).unsqueeze(1)  # (T,1)
        frequencies = 10.0 ** (torch.arange(dim) / (dim - 1) * 4.0).unsqueeze(0)  # (1,dim)
        table = steps * frequencies  # (T,dim)
        table = torch.cat([torch.sin(table), torch.cos(table)], dim=1)  # (T,dim*2)
        return table

class ResidualBlock(nn.Module):
    def __init__(self, channels, diffusion_embedding_dim, pos_emb_dim, nheads):
        super().__init__()
        self.diffusion_projection = nn.Linear(diffusion_embedding_dim, channels) # to project time step to the same dimension as noise
        self.mid_projection = Conv1d_with_init(channels, 2 * channels, 1)
        self.output_projection = Conv1d_with_init(channels, 2 * channels, 1)
        self.cond_projection = Conv1d_with_init(pos_emb_dim, 2 * channels, 1)

        self.time_layer = get_torch_trans(heads=nheads, layers=1, channels=channels)
        self.feature_layer = get_torch_trans(heads=nheads, layers=1, channels=channels)

    def forward_time(self, y, base_shape):
        B, channel, K, L = base_shape
        if L == 1:
            return y
        y = y.reshape(B, channel, K, L).permute(0, 2, 1, 3).reshape(B * K, channel, L)
        y = self.time_layer(y.permute(2, 0, 1)).permute(1, 2, 0)
        y = y.reshape(B, K, channel, L).permute(0, 2, 1, 3).reshape(B, channel, K * L)
        return y

    def forward_feature(self, y, base_shape):
        B, channel, K, L = base_shape
        if K == 1:
            return y
        y = y.reshape(B, channel, K, L).permute(0, 3, 1, 2).reshape(B * L, channel, K)
        y = self.feature_layer(y.permute(2, 0, 1)).permute(1, 2, 0)
        y = y.reshape(B, L, channel, K).permute(0, 2, 3, 1).reshape(B, channel, K * L)
        return y

    def forward(self, x, cond_info, diffusion_emb):
        # print(x.shape, cond_info.shape, diffusion_emb.shape)
        B, channel, K, L = x.shape
        base_shape = x.shape
        x = x.reshape(B, channel, K * L)

        diffusion_emb = self.diffusion_projection(diffusion_emb).unsqueeze(-1)  # (B,channel,1)
        # print("adding the input and timestep embedding : ", x.shape, diffusion_emb.shape)
        y = x + diffusion_emb
        # print(y.shape)

        y = self.forward_time(y, base_shape)
        y = self.forward_feature(y, base_shape)  # (B,channel,K*L)
        y = self.mid_projection(y)  # (B,2*channel,K*L)

        _, cond_dim, _, _ = cond_info.shape
        cond_info = cond_info.reshape(B, cond_dim, K * L)
        cond_info = self.cond_projection(cond_info)  # (B,2*channel,K*L)
        y = y + cond_info

        gate, filter = torch.chunk(y, 2, dim=1)
        y = torch.sigmoid(gate) * torch.tanh(filter)  # (B,channel,K*L)
        y = self.output_projection(y)

        residual, skip = torch.chunk(y, 2, dim=1)
        x = x.reshape(base_shape)
        residual = residual.reshape(base_shape)
        skip = skip.reshape(base_shape) 
        return (x + residual) / math.sqrt(2.0), skip

class Denoiser(nn.Module):
    def __init__(self, model_config, inputdim):
        super().__init__()
        self.channels = model_config.channels
        self.inputdim = inputdim
        self.num_features = model_config.n_features 
        self.horizon = model_config.max_seq_len
        self.num_steps = model_config.num_diffusion_steps
        self.diffusion_step_embedding_dim = model_config.diffusion_step_embedding_dim
        self.pos_emb_dim = model_config.dim_val

        self.diffusion_embedding = DiffusionEmbedding(num_steps=self.num_steps, embedding_dim=model_config.diffusion_step_embedding_dim)
        self.input_projection = Conv1d_with_init(self.inputdim, self.channels, 1)
        self.output_projection1 = Conv1d_with_init(self.channels, self.channels, 1)
        self.output_projection2 = Conv1d_with_init(self.channels, self.inputdim, 1)
        self.residual_layers = nn.ModuleList([ResidualBlock(channels=self.channels, diffusion_embedding_dim=self.diffusion_step_embedding_dim, pos_emb_dim=self.pos_emb_dim, nheads=model_config.n_heads) for _ in range(model_config.n_denoising_layers)])

    def forward(self, x, cond_info, diffusion_step):
        B = x.shape[0]
        x = x.reshape(B, self.inputdim, self.num_features*self.horizon)
        x = self.input_projection(x)
        x = F.relu(x)
        x = x.reshape(B, self.channels, self.num_features, self.horizon)
        diffusion_emb = self.diffusion_embedding(diffusion_step)

        skip = []
        for layer in self.residual_layers:
            x, skip_connection = layer(x, cond_info, diffusion_emb)
            skip.append(skip_connection)

        x = torch.sum(torch.stack(skip), dim=0) / math.sqrt(len(self.residual_layers))
        x = x.reshape(B, self.channels, self.num_features*self.horizon)
        x = self.output_projection1(x)  # (B,channel,K*L)
        x = F.relu(x)
        x = self.output_projection2(x)  # (B,1,K*L)
        x = x.reshape(B, self.num_features, self.horizon)
        return x
    
class UnConditionalLDM(nn.Module):
    def __init__(self, model_config, device):
        super().__init__()
        self.device = device
        
        # parameters for the denoiser model
        self.diffusion_step_embedding_dim = model_config.diffusion_step_embedding_dim # 96 
        self.latent_dim = model_config.latent_dim # 96
        self.side_dim = self.diffusion_step_embedding_dim
        self.input_dim = 1
        self.num_features = model_config.n_features
        self.horizon = model_config.max_seq_len
        self.pos_emb = model_config.dim_val

        self.diffmodel = Denoiser(model_config=model_config, inputdim=self.input_dim) 

        # parameters for diffusion models
        self.num_steps = model_config.num_diffusion_steps
        self.beta = np.linspace(model_config.beta_start, model_config.beta_end, self.num_steps)
        self.alpha_hat = 1 - self.beta
        self.alpha = np.cumprod(self.alpha_hat)
        self.alpha_torch = torch.tensor(self.alpha).float().to(self.device).unsqueeze(1).unsqueeze(1)

        # validation 
        self.num_plots = model_config.n_plots

    def pos_embedding(self, pos, d_model=128):
        pe = torch.zeros(pos.shape[0], pos.shape[1], d_model).to(self.device)
        position = pos.unsqueeze(2)
        div_term = 1 / torch.pow(10000.0, torch.arange(0, d_model, 2).to(self.device) / d_model)
        pe[:, :, 0::2] = torch.sin(position * div_term)
        pe[:, :, 1::2] = torch.cos(position * div_term)
        return pe

    def get_side_info(self, observed_tp):
        time_embed = self.pos_embedding(observed_tp, self.pos_emb)  # (B,L,emb)
        time_embed = time_embed.unsqueeze(2).expand(-1, -1, self.num_features, -1) # (B, L, K, emb)
        side_info = time_embed.permute(0, 3, 2, 1)  # (B,*,K,L)
        return side_info

    def synthesize(self):
        timepoints = torch.arange(self.horizon).unsqueeze(0).repeat(self.num_plots, 1).to(self.device).float()
        side_info = self.get_side_info(timepoints)
        synthesized_output = torch.randn((self.num_plots,1,self.horizon)).to(self.device) # step 1
        for tidx in reversed(range(self.num_steps)):            
            input = synthesized_output.unsqueeze(1)
            predicted_noise = self.diffmodel(input, side_info, torch.tensor([tidx]*self.num_plots).to(self.device))
            coeff1 = 1 / self.alpha_hat[tidx] ** 0.5
            coeff2 = (1 - self.alpha_hat[tidx]) / (1 - self.alpha[tidx]) ** 0.5
            synthesized_output = coeff1 * (synthesized_output - coeff2 * predicted_noise)
            if tidx > 0:
                noise = torch.randn_like(synthesized_output).to(self.device)
                sigma = ((1.0 - self.alpha[tidx - 1]) / (1.0 - self.alpha[tidx]) * self.beta[tidx]) ** 0.5
                synthesized_output += sigma * noise

        return synthesized_output

    def forward(self, batch):
        observed_data = batch["observed_data"].to(self.device).float()
        observed_tp = torch.arange(self.horizon).unsqueeze(0).repeat(observed_data.shape[0], 1).to(self.device).float() # (B x horizon)
        side_info = self.get_side_info(observed_tp)

        diffusion_steps = torch.randint(0, self.num_steps, [observed_data.shape[0]]).to(self.device)
        current_alpha = self.alpha_torch[diffusion_steps]
        noise = torch.randn_like(observed_data).to(self.device).float()
        noisy_data = (current_alpha ** 0.5) * observed_data + (1.0 - current_alpha) ** 0.5 * noise
        total_input = noisy_data.unsqueeze(1)
        predicted_noise = self.diffmodel(total_input, side_info, diffusion_steps)  # (B,K,L)
        loss = F.mse_loss(predicted_noise, noise)
        return loss
