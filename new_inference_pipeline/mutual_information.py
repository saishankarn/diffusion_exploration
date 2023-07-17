import os 
import torch 

if "__name__" == "__main__":
    torch_device = "cuda" if torch.cuda.is_available() else "cpu"
    inference_values = torch.load('logs/inference_values.pt').to(torch_device)

    