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

class EncoderBlock(nn.Module):
    def __init__(self, model_config):
        super().__init__()

        self.side_dim = model_config.positional_embedding_dim
        self.num_channels = model_config.channels
        self.n_heads = model_config.n_heads 
        
        self.cond_projection = Conv1d_with_init(self.side_dim, 2*self.num_channels, 1)
        self.mid_projection = Conv1d_with_init(self.num_channels, 2*self.num_channels, 1)
        self.output_projection = Conv1d_with_init(2*self.num_channels, self.num_channels, 1)

        self.time_layer = get_torch_trans(heads=self.n_heads, layers=1, channels=self.num_channels)
        self.feature_layer = get_torch_trans(heads=self.n_heads, layers=1, channels=self.num_channels)
 
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

        y = self.output_projection(y) # (B,channel,K*L)
        y = y.reshape(base_shape) # (B, channel, K, L)
        return y

class CSDIEncoder(nn.Module):
    def __init__(self, model_config, input_dim):
        super().__init__()
        self.inputdim = input_dim 
        self.num_features = model_config.num_input_features 
        self.horizon = model_config.horizon

        self.num_encoder_layers = model_config.n_encoder_layers
        self.channels = model_config.channels

        self.latent_dim = model_config.latent_dim

        self.input_projection = Conv1d_with_init(self.inputdim, self.channels, 1)
        self.encoder_layers = nn.ModuleList([EncoderBlock(model_config) for _ in range(self.num_encoder_layers)])

        self.mean_projection_conv = Conv1d_with_init(self.channels, self.horizon // self.latent_dim, 1)
        self.mean_projection_fc = nn.Linear(int(self.horizon*(self.horizon//self.latent_dim)), self.latent_dim)
        
        self.log_var_projection_conv = Conv1d_with_init(self.channels, self.horizon // self.latent_dim, 1)
        self.log_var_projection_fc = nn.Linear(int(self.horizon*(self.horizon//self.latent_dim)), self.latent_dim)

        self.act = torch.nn.ELU()

    def forward(self, x, cond_info):
        B = x.shape[0]
        
        # input projection
        encoded = x.reshape(B, self.inputdim, self.num_features*self.horizon)
        encoded = self.input_projection(encoded)
        # encoded = F.relu(encoded)
        encoded = self.act(encoded)
        
        # processing through the encoder layers (transformer encoders with self attention)
        encoded = encoded.reshape(B, self.channels, self.num_features, self.horizon)
        for layer in self.encoder_layers:
            encoded = layer(encoded, cond_info)
        encoded = encoded.reshape(B, self.channels, self.num_features*self.horizon)

        # mean and variance projection from the encodings
        mean = self.mean_projection_conv(encoded)
        mean = self.act(mean)
        mean = mean.reshape(B, int(self.horizon*(self.horizon//self.latent_dim)))
        mean = self.mean_projection_fc(mean)


        log_var = self.log_var_projection_conv(encoded)
        log_var = self.act(log_var)
        log_var = log_var.reshape(B, int(self.horizon*(self.horizon//self.latent_dim)))
        log_var = self.log_var_projection_fc(log_var)

        return mean, log_var

class DecoderBlock(nn.Module):
    def __init__(self, model_config):
        super().__init__()
        self.side_dim = model_config.positional_embedding_dim 
        self.num_channels = model_config.channels 
        self.n_heads = model_config.n_heads

        self.cond_projection = Conv1d_with_init(self.side_dim, 2*self.num_channels, 1)
        self.mid_projection = Conv1d_with_init(self.num_channels, 2*self.num_channels, 1)
        self.output_projection = Conv1d_with_init(2*self.num_channels, self.num_channels, 1)

        self.time_layer = get_torch_trans(heads=self.n_heads, layers=1, channels=self.num_channels)
        self.feature_layer = get_torch_trans(heads=self.n_heads, layers=1, channels=self.num_channels)

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

        y = self.output_projection(y) # (B,1,K*L)
        y = y.reshape(base_shape) # (B, channels, K, L)
        return y

class CSDIDecoder(nn.Module):
    def __init__(self, model_config, input_dim):
        super().__init__()
        self.inputdim = input_dim
        self.num_features = model_config.num_input_features
        self.horizon = model_config.horizon # L

        self.num_decoder_layers = model_config.n_decoder_layers
        self.channels = model_config.channels

        self.latent_dim = model_config.latent_dim # l

        self.num_features = model_config.num_input_features # K

        self.decoder_layers = nn.ModuleList([DecoderBlock(model_config) for _ in range(self.num_decoder_layers)])

        self.decompress_encoded_fc = nn.Linear(self.latent_dim, int(self.horizon*(self.horizon//self.latent_dim)))
        self.decompress_encoded_conv = Conv1d_with_init(self.horizon//self.latent_dim, self.channels, 1)

        self.output_projection = Conv1d_with_init(self.channels, self.inputdim, 1) 
        
        self.act = torch.nn.ELU()

    def forward(self, encoded, cond_info):
        B = encoded.shape[0]

        # decompress the encoded information
        decoded = self.decompress_encoded_fc(encoded) # B x (L/l)*K*L
        decoded = self.act(decoded)
        decoded = decoded.reshape(B, int(self.horizon//self.latent_dim), self.num_features*self.horizon) # B x (L/l) x K*L
        decoded = self.decompress_encoded_conv(decoded)

        # process the decoded info using transformer encoder layers 
        decoded = decoded.reshape(B, self.channels, self.num_features, self.horizon)
        for layer in self.decoder_layers:
            decoded = layer(decoded, cond_info)
        decoded = decoded.reshape(B, self.channels, self.num_features * self.horizon)
        # decoded = F.relu(decoded)
        decoded = self.act(decoded) 

        # output projection
        decoded = self.output_projection(decoded)  # (B,1,K*L)
        x_pred = decoded.reshape(B, self.num_features, self.horizon)
        
        return x_pred

class TimeSeriesVariationalAutoEncoder(nn.Module):
    def __init__(self, model_config, device):
        super().__init__()
        self.num_features = model_config.num_input_features
        self.horizon = model_config.horizon
        self.positional_embedding_dim = model_config.positional_embedding_dim
        self.latent_dim = model_config.latent_dim

        self.device = device
        
        input_dim = 1 
        """
        The input should be Batch_size x input_dim x num_features*horizon 
        To draw parallels to an image input, input_dim is the number of channels 
        """
        self.encoder = CSDIEncoder(model_config, input_dim)
        self.decoder = CSDIDecoder(model_config, input_dim)

    def pos_embedding(self, pos, pos_emb_dim=128):
        pe = torch.zeros(pos.shape[0], pos.shape[1], pos_emb_dim).to(self.device)
        position = pos.unsqueeze(2)
        div_term = 1 / torch.pow(10000.0, torch.arange(0, pos_emb_dim, 2).to(self.device) / pos_emb_dim)
        pe[:, :, 0::2] = torch.sin(position * div_term)
        pe[:, :, 1::2] = torch.cos(position * div_term)
        return pe

    def get_side_info(self, observed_tp):
        time_embed = self.pos_embedding(observed_tp, self.positional_embedding_dim)  # (B,L,emb)
        time_embed = time_embed.unsqueeze(2).expand(-1, -1, self.num_features, -1) # (B, L, K, emb)
        side_info = time_embed.permute(0, 3, 2, 1)  # (B,*,K,L)
        return side_info
    
    def reparameterization(self, mean, var):
        epsilon = torch.randn_like(var)        
        z = mean + var*epsilon
        return z

    def forward(self, batch):
        observed_data = batch.to(self.device).float()
        observed_tp = torch.arange(self.horizon).unsqueeze(0).repeat(observed_data.shape[0],1).to(self.device).float()
        side_info = self.get_side_info(observed_tp)
        total_input = observed_data.unsqueeze(1) # process the input before forward pass to the neural network 
        mean, log_var = self.encoder(total_input, side_info)
        z = self.reparameterization(mean=mean, var=log_var)
        pred_data = self.decoder(encoded=z, cond_info=side_info)
        return mean, log_var, pred_data 
    
    def decode(self, encoded):
        B = encoded.shape[0]
        observed_tp = torch.arange(self.horizon).unsqueeze(0).repeat(B, 1).to(self.device).float() # (B x horizon)
        side_info = self.get_side_info(observed_tp)
        pred_data = self.decoder(encoded=encoded, cond_info=side_info)
        return pred_data 