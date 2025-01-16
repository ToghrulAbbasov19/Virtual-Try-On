import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from torch.utils.data import Dataset
import numpy as np
import torchvision.transforms
import torch.nn.functional as F
from diffusion.diffusion import Diffusion
from models.unet_cond import UNet
from train2 import train
from preprocess import TrainDataset
from preprocess.dataset import VTODataset
from models.new_unet import UNetModel, SuperResModel, EncoderUNetModel
import argparse
from diffusers import AutoencoderKL

def build_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="./data/train")
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--batch_size", "-bs",  type=int, default=16)
    parser.add_argument("--device", type=str, default='cuda')
    parser.add_argument("--save_epochs", type=int, default=5)
    parser.add_argument("--no_epochs", type=int, default=1000)
    parser.add_argument("--save_dir", type=str, default="./ckpts")
    parser.add_argument("--pair_path", type=str, default="./data/train_pairs.txt")
    
    args=parser.parse_args()
    return args

def main():
    args=build_args()
    size=(args.size, args.size)
    transform= torchvision.transforms.Compose([
        torchvision.transforms.Resize(size),
        torchvision.transforms.ToTensor(),
        lambda x: x * 255.0
    ])  # Used when processing with PIL
    vae = AutoencoderKL.from_pretrained(
        "diffusers/stable-diffusion-xl-1.0-inpainting-0.1",
        subfolder="vae",
        torch_dtype=torch.float32,
        # device=args.device
    ).to(args.device)
    print(vae.device)
    vae.to(args.device)
    vae.requires_grad_(False)
    # vae.to(args.device)
    data=VTODataset(args.data_path, size, args.pair_path, None, transform, args.device)
    loader=DataLoader(data, batch_size=args.batch_size)
    unet=UNet(c_in=20, c_out=4, device=args.device).to(args.device)
    model_diff=Diffusion(1000, 0.9999, 0.98, args.device)
    loss_func=nn.MSELoss()
    optim=torch.optim.AdamW(unet.parameters(), lr=args.lr)
    train(loader, model_diff, unet, vae, loss_func, optim, args.no_epochs, args.device, args.save_dir, args.save_epochs)


if __name__=="__main__":
    main()