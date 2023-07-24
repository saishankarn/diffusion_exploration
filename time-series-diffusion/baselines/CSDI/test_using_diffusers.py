import os
import json

import torch
import numpy as np
import matplotlib.pyplot as plt

from diffusers import DDPMScheduler, UNet1DModel, DDPMPipeline

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

device="cpu"
config_loc = 'config.json'
with open(config_loc) as f:
    data = f.read() 
config = json.loads(data)
print(config)

num_diff_steps = config["diffusion"]["num_steps"]
beta_schedule = config["diffusion"]["schedule"]
noise_scheduler = DDPMScheduler(
    num_train_timesteps=num_diff_steps, 
    beta_schedule=beta_schedule
)

model = UNet1DModel(
    in_channels=1,
    out_channels=1,
    layers_per_block=2,
    block_out_channels=(128, 128, 256, 512),
    down_block_types=(
        "DownBlock1D",
        "DownBlock1D",
        "DownBlock1D",
        "AttnDownBlock1D",
    ),
    up_block_types=(
        "AttnUpBlock1D",
        "UpBlock1D",
        "UpBlock1D",
        "UpBlock1D",
    ),
).to(device)

model_ckpt = 'results/electricity/21-07-2023-22:50:59/checkpoints/checkpoint-7000'
model.load_state_dict(torch.load(model_ckpt))
model.eval()

# Sample noise that we'll add to the images
noisy_image = torch.randn(
    (1,1,96), dtype=(torch.float32)
).to(device)
print(noise_scheduler.timesteps)
for t in noise_scheduler.timesteps:
    print(t)
    # 1. predict noise model_output
    noise = model(noisy_image, t).sample

    # 2. compute previous image: x_t -> x_t-1
    denoised_image = noise_scheduler.step(noise, t, noisy_image).prev_sample
    noisy_image = denoised_image
    print(noisy_image.shape)

print(noisy_image)
ts = noisy_image.detach().numpy()[0][0]
print(ts)
x = np.arange(96)
plt.plot(x, ts)
plt.show()

