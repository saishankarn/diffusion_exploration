import torch
import torch.nn as nn
import torch.nn.functional as F
import math


def get_torch_trans(heads=8, layers=1, channels=64):
    encoder_layer = nn.TransformerEncoderLayer(d_model=channels, nhead=heads, dim_feedforward=64, activation="gelu")
    return nn.TransformerEncoder(encoder_layer, num_layers=layers)

def Conv1d_with_init(in_channels, out_channels, kernel_size):
    layer = nn.Conv1d(in_channels, out_channels, kernel_size)
    nn.init.kaiming_normal_(layer.weight)
    return layer

class diff_CSDI(nn.Module):
    def __init__(self, model_config, inputdim=2):
        super().__init__()
        self.channels = model_config.channels

        self.input_projection = Conv1d_with_init(inputdim, model_config.channels, 1)
        self.output_projection1 = Conv1d_with_init(model_config.channels, model_config.channels, 1)
        self.output_projection2 = Conv1d_with_init(model_config.channels, inputdim, 1) 
        nn.init.zeros_(self.output_projection2.weight)

        self.residual_layers = nn.ModuleList(
            [
                ResidualBlock(
                    side_dim=model_config.dim_val,
                    channels=model_config.channels,
                    nheads=model_config.n_heads,
                )
                for _ in range(model_config.n_encoder_layers)
            ]
        )

    def forward(self, x, cond_info):
        B, inputdim, K, L = x.shape
        x = x.reshape(B, inputdim, K * L)
        x = self.input_projection(x)
        x = F.relu(x)
        x = x.reshape(B, self.channels, K, L)

        # print(x.shape, cond_info.shape, diffusion_emb.shape)

        skip = []
        for layer in self.residual_layers:
            x, skip_connection = layer(x, cond_info)
            skip.append(skip_connection)

        x = torch.sum(torch.stack(skip), dim=0) / math.sqrt(len(self.residual_layers))
        x = x.reshape(B, self.channels, K * L)
        x = self.output_projection1(x)  # (B,channel,K*L)
        x = F.relu(x)
        x = self.output_projection2(x)  # (B,1,K*L)
        x = x.reshape(B, K, L)
        return x


class ResidualBlock(nn.Module):
    def __init__(self, side_dim, channels, nheads):
        super().__init__()
        self.cond_projection = Conv1d_with_init(side_dim, 2 * channels, 1)
        self.mid_projection = Conv1d_with_init(channels, 2 * channels, 1)
        self.output_projection = Conv1d_with_init(channels, 2 * channels, 1)

        self.time_layer = get_torch_trans(heads=nheads, layers=1, channels=channels)
        self.feature_layer = get_torch_trans(heads=nheads, layers=1, channels=channels)

    def forward_time(self, y, base_shape):
        B, channel, K, L = base_shape
        y = y.reshape(B, channel, K, L).permute(0, 2, 1, 3).reshape(B * K, channel, L)
        y = self.time_layer(y.permute(2, 0, 1)).permute(1, 2, 0)
        y = y.reshape(B, K, channel, L).permute(0, 2, 1, 3).reshape(B, channel, K * L)
        return y

    def forward_feature(self, y, base_shape):
        B, channel, K, L = base_shape
        y = y.reshape(B, channel, K, L).permute(0, 3, 1, 2).reshape(B * L, channel, K)
        y = self.feature_layer(y.permute(2, 0, 1)).permute(1, 2, 0)
        y = y.reshape(B, L, channel, K).permute(0, 2, 3, 1).reshape(B, channel, K * L)
        return y

    def forward(self, x, cond_info):
        B, channel, K, L = x.shape
        base_shape = x.shape
        y = x.reshape(B, channel, K * L)

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
    


###########################################################################################################
########## from clap_transformer

def plot_results(model, val_dataloader, num_plots, log_dir, epoch, device):
    fig, axs = plt.subplots(nrows=1, ncols=num_plots, sharey=True, sharex=True, figsize=(32, 8))
    model = model.eval()
    for i, val_batch in enumerate(val_dataloader):
        if i >= num_plots:
            break
        seq_true = val_batch["observed_data"]
        predictions, _, _ = model(val_batch)
        pred = predictions[0].squeeze(0).cpu().detach().numpy()
        true = seq_true[0].squeeze(0).cpu().numpy()
        axs[i].plot(true, label='true')
        axs[i].plot(pred, label='reconstructed')
        
    fig.tight_layout()

    save_dir = os.path.join(log_dir, 'qualitative')
    os.makedirs(save_dir, exist_ok=True)
    save_loc = os.path.join(save_dir, str(epoch)+'.png')
    plt.savefig(save_loc)
    plt.close('all')