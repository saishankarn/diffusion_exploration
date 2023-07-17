from diffusers import UNet2DModel, DDIMScheduler, VQModel
import torch
import PIL.Image
import numpy as np
import tqdm

if __name__ == "__main__":
    batch_size = 16
    num_batches = 10
    num_images = int(batch_size * num_batches)
    num_inference_steps = 1000
    seeds = np.random.randint(0, 100, num_batches)

    # load all models
    unet = UNet2DModel.from_pretrained("CompVis/ldm-celebahq-256", subfolder="unet")
    vqvae = VQModel.from_pretrained("CompVis/ldm-celebahq-256", subfolder="vqvae")
    scheduler = DDIMScheduler.from_config("CompVis/ldm-celebahq-256", subfolder="scheduler")

    # set to cuda
    torch_device = "cuda" if torch.cuda.is_available() else "cpu"

    unet.to(torch_device)
    vqvae.to(torch_device)
    
    dataset = torch.zeros((num_batches, num_inference_steps, batch_size, unet.in_channels, unet.sample_size, unet.sample_size), device=torch_device)
    for batch_idx in range(num_batches):
        seed = seeds[batch_idx]
        generator = torch.manual_seed(seed)
        noise = torch.randn((batch_size, unet.in_channels, unet.sample_size, unet.sample_size), generator=generator).to(torch_device)
        scheduler.set_timesteps(num_inference_steps=num_inference_steps)

        image = noise
        for timestep_idx in tqdm.tqdm(range(num_inference_steps)):
            t = scheduler.timesteps[timestep_idx]
            dataset[batch_idx,timestep_idx] = image 
            with torch.no_grad():
                residual = unet(image, t)["sample"]
            prev_image = scheduler.step(residual, t, image, eta=0.0)["prev_sample"]
            image = prev_image

    dataset = dataset.view(-1, batch_size, unet.in_channels, unet.sample_size, unet.sample_size)
    # print(dataset.shape)
    torch.save(dataset, 'logs/inference_values.pt')

    # decode image with vae
    # with torch.no_grad():
        # image = vqvae.decode(image)

    # process image
    # image_processed = image.sample.cpu().permute(0, 2, 3, 1)
    # image_processed = (image_processed + 1.0) * 127.5
    # image_processed = image_processed.clamp(0, 255).numpy().astype(np.uint8)
    # image_pil = PIL.Image.fromarray(image_processed[0])

    # image_pil.save(f"generated_image_{seed}.png")