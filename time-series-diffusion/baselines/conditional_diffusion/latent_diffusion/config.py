import numpy as np
from itertools import product

class DatasetConfig():
    def __init__(self):
        self.horizon = 48
        self.start_time = 0
        self.end_time = 2*np.pi 

        self.minimum_amplitude = 0.1
        self.maximum_amplitude = 1.0
        self.amplitude_interval = 0.05 
        self.amplitudes = list(np.arange(self.minimum_amplitude, self.maximum_amplitude + self.amplitude_interval, self.amplitude_interval))
    
        self.minimum_frequency = 0.1
        self.maximum_frequency = 1 
        self.frequency_interval = 0.05
        self.frequencies = list(np.arange(self.minimum_frequency, self.maximum_frequency + self.frequency_interval, self.frequency_interval))

        self.minimum_phase = 0.0 
        self.maximum_phase = 2*np.pi
        self.phase_interval = 5*np.pi/180
        self.phases = list(np.arange(self.minimum_phase, self.maximum_phase, self.phase_interval))

        self.sine_params = list(product(self.amplitudes, self.frequencies, self.phases))

        self.train_test_ratio = 0.95
    
class VAEConfig():
    def __init__(self):
        # train config parameters
        self.train_batch_size = 64
        self.val_batch_size = 64
        self.n_epochs = 2000
        self.learning_rate = 1e-4
        self.n_plots = 6
        self.val_epoch_interval = 5

        # encoder parameters
        self.n_features = 1
        self.dim_val = 128
        self.dropout_pos_enc = 0.2 
        self.max_seq_len = 48
        self.n_heads = 8
        self.n_encoder_layers = 2
        self.n_decoder_layers = 1
        self.batch_first = True
        self.channels = 64

        # parameter encoder parameters 
        self.n_parameters = 3 
        self.clap_latent_dim = self.max_seq_len*2

        # model initialization 
        self.pretrained_dir = '../save/sine_20230818_051528/'