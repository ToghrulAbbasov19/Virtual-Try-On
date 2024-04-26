import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from torch.utils.data import Dataset
import numpy as np
import torchvision.transforms
import torch.nn.functional as F
from diffusion.diffusion import Diffusion
from models.unet import UNet
from train import train
from preprocess.preprocess import TrainDataset
import argparse
from models.new_unet import UNetModel, SuperResModel, EncoderUNetModel

def build_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="../")
    parser.add_argument("--batch_size", "-bs",  type=int, default=16)
    parser.add_argument("--device", type=str, default='cpu')
    parser.add_argument("--save_epochs", type=int, default=10)
    parser.add_argument("--no_epochs", type=int, default=1000)
    parser.add_argument("--save_path", type=str, default="")
    parser.add_argument("--pair_path", type=str, default="")
    
    args=parser.parse_args()
    return args

def main():
    args=build_args()
    transform=transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize((64, 64)),
        torchvision.transforms.ToTensor(),
        lambda x: x * 255.0
    ])
    data=TrainDataset(args.data_path, (64, 64), args.pair_path, None, transform, args.device)
    loader=DataLoader(data, batch_size=args.batch_size)
    unet=UNet(c_in=15, device=args.device).to(args.device)
    model_diff=Diffusion(1000, 0.9999, 0.98, args.device)
    loss_func=nn.MSELoss()
    optim=torch.optim.AdamW(unet.parameters(), lr=3e-4)
    train(loader, model_diff, unet, loss_func, optim, 4000, args.device, args.save_path, args.save_epochs)


