import argparse
import math
import os
import json
import datetime

import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

from diffusers import DDPMScheduler, UNet1DModel
from diffusers.optimization import get_scheduler

from dataset import get_dataloader_electricity

def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="electricity",
        help="The name of the Dataset."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.json",
        help="The config of the dataset, model, and the training, in JSON format.",
    )
    args = parser.parse_args()
    if args.dataset_name is None and args.train_data_dir is None:
        raise ValueError("You must specify either a dataset name from the hub or a train data directory.")
    
    return args

def main(config):
    logging_dir = os.path.join(
        config["store"]["output_dir"], 
        config["store"]["logging_dir"]
    )
    os.makedirs(logging_dir, exist_ok=True)

    checkpoint_dir = os.path.join(
        config["store"]["output_dir"], 
        config["store"]["checkpoint_dir"]
    )
    os.makedirs(checkpoint_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ## important training parameters
    batch_size = config["train"]["batch_size"]
    num_epochs = config["train"]["epochs"]

    # define the train data loader
    train_dataloader = get_dataloader_electricity(
        config["dataset"]["dataset_path"], 
        batch_size
    )

    # initialize the model
    model = UNet1DModel(
            in_channels=1,
            out_channels=1,
            layers_per_block=2,
            block_out_channels=(128, 256),
            down_block_types=(
                "DownBlock1D",
                "AttnDownBlock1D",
            ),
            up_block_types=(
                "AttnUpBlock1D",
                "UpBlock1D",
            ),
    ).to(device)


    # initialize the noise scheduler
    num_diff_steps = config["diffusion"]["num_steps"]
    beta_schedule = config["diffusion"]["schedule"]
    noise_scheduler = DDPMScheduler(
        num_train_timesteps=num_diff_steps, 
        beta_schedule=beta_schedule
    )

    # Initialize the optimizer
    lr = config["train"]["learning_rate"]
    beta1 = config["train"]["adam_beta1"]
    beta2 = config["train"]["adam_beta2"]
    awd = config["train"]["adam_weight_decay"]
    eps = config["train"]["adam_epsilon"]
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        betas=(beta1, beta2),
        weight_decay=awd,
        eps=eps
    )

    # Initialize the learning rate scheduler
    scheduler_type = config["train"]["lr_scheduler"]
    gradient_accumulation_steps = config["train"]["gradient_accumulation_steps"]
    num_warmup_steps = config["train"]["lr_warmup_steps"] * gradient_accumulation_steps
    lr_scheduler = get_scheduler(
        scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=(len(train_dataloader) * num_epochs),
    )

    global_step = 0
    first_epoch = 0
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / gradient_accumulation_steps)

    # Train!
    for epoch in range(first_epoch, num_epochs):
        model.train()
        progress_bar = tqdm(total=num_update_steps_per_epoch)
        progress_bar.set_description(f"Epoch {epoch}")
        for step, batch in enumerate(train_dataloader):
            clean_time_series = batch 
            clean_time_series = clean_time_series.to(device)

            # Sample noise that we'll add to the images
            noise = torch.randn(
                clean_time_series.shape, dtype=(torch.float32)
            ).to(clean_time_series.device)
            bsz = clean_time_series.shape[0]

            # Sample a random timestep for each image
            timesteps = torch.randint(
                0, noise_scheduler.config.num_train_timesteps, 
                (bsz,), device=clean_time_series.device
            ).long()

            # Add noise to the clean time series according to the noise magnitude at each timestep
            # (this is the forward diffusion process)
            noisy_time_series = noise_scheduler.add_noise(clean_time_series, noise, timesteps)

            # Predict the noise residual
            model_output = model(noisy_time_series, timesteps).sample

            # Calculate the loss
            loss = F.mse_loss(model_output, noise)  # this could have different weights!

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_scheduler.step()

            progress_bar.update(1)
            global_step += 1

            if global_step % config["store"]["checkpointing_steps"] == 0:
                checkpoint_loc = os.path.join(
                    checkpoint_dir, 
                    "checkpoint-%d"%global_step
                )
                torch.save(model.state_dict(), checkpoint_loc)

            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0], "step": global_step}
            progress_bar.set_postfix(**logs)

        progress_bar.close()

if __name__ == "__main__":
    args = parse_args()
    with open(args.config) as f:
        data = f.read() 
    config = json.loads(data)
    print(config)

    print(json.dumps(config, indent=4))

    current_time = datetime.datetime.now().strftime("%d-%m-%Y-%H:%M:%S") 
    results_dir = os.path.join(
        config["store"]["output_dir"],
        config["dataset"]["dataset_name"],
        current_time)
    print('model folder:', results_dir)
    config["store"]["output_dir"] = results_dir
    os.makedirs(results_dir, exist_ok=True)
    with open(os.path.join(results_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=4)
    
    main(config)