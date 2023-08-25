"""
refer to the CSDI paper for more details - https://arxiv.org/abs/2107.03502
"""

import numpy as np
from itertools import product

class DatasetConfig():
    def __init__(self):
        self.horizon = 96
        self.train_test_ratio = 0.8
        self.dataset_dir = '../../datasets/electricity/'
    
class VAEConfig():
    def __init__(self): 
        # train config parameters
        self.train_batch_size = 256
        self.val_batch_size = 256
        self.n_epochs = 2000
        self.learning_rate = 1e-4
        self.n_plots = 6
        self.val_epoch_interval = 5

        # autoencoder parameters
        # embedding dims
        self.num_input_features = 1 # 1 for electricity dataset as there is only one load demand time series per user per day (K in the paper)
        self.positional_embedding_dim = 128 
        self.horizon = 96 # the length of the time series (L in the paper)
        self.latent_dim = 48
        
        # size of layers
        self.n_heads = 8
        self.n_encoder_layers = 4
        self.n_decoder_layers = 4
        self.batch_first = True
        self.channels = 64

        self.dropout_pos_enc = 0.2

        # model initialization 
        self.pretrained_loc = 'save/electricity_20230824_202525/results/model_best.pth'