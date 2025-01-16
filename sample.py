import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from torch.utils.data import Dataset
import numpy as np
import torchvision.transforms
import torch.nn.functional as F
from diffusion.diffusion import Diffusion
from models.unet_cond import UNet
from preprocess import TrainDataset
from models.new_unet import UNetModel
from preprocess.dataset import VTODataset
import cv2
from diffusers import AutoencoderKL
import argparse
import os 

def build_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="./data/test")
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--model_path", type=str, default="./ckpts/model_50.pt")
    parser.add_argument("--save_dir", type=str, default="./images")
    parser.add_argument("--device", type=str, default='cuda')
    parser.add_argument("--pair_path", type=str, default="./data/test_pairs.txt")
    
    args=parser.parse_args()
    return args

# device="cuda"
def main():
    args=build_args()
    device=args.device
    size=(args.size, args.size)
    model_diff=Diffusion(1000, 0.9999, 0.98, device)
    unet=UNet(c_in=20, c_out=4, device=device).to(device)
    unet.load_state_dict(torch.load(args.model_path))
    vae = AutoencoderKL.from_pretrained(
            "diffusers/stable-diffusion-xl-1.0-inpainting-0.1",
            subfolder="vae",
            torch_dtype=torch.float32,
        ).to(device)



    transform=torchvision.transforms.Compose([
            torchvision.transforms.Resize(size),
            # torchvision.transforms.RandomResizedCrop(64, scale=(0.8, 1.0)),
            torchvision.transforms.ToTensor(),
            lambda x: x * 255.0
            # torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    data=VTODataset(args.data_path, size, args.pair_path, None, transform, device)
    image_path=os.path.join(args.save_dir, "source_img.png")
    cloth_path=os.path.join(args.save_dir, "cloth.png")
    out_path=os.path.join(args.save_dir, "output.png")

    loader=DataLoader(data, batch_size=1, shuffle=False)
    # print(data.data)
    for i, total_img in enumerate(loader):
        # print(i)
        if i==9:
            m_clt=total_img[4].type(torch.float32).to(device)
        if i==30:
            image=total_img[0].type(torch.float32).to(device)
            cloth=m_clt
            img_to_save=image[0].detach().cpu().numpy()
            print(img_to_save.shape)
            clt_to_save=cloth[0].detach().cpu().numpy()
            img_to_save=np.transpose(img_to_save, (1, 2, 0))
            img_to_save=(img_to_save+1)*127.5
            clt_to_save=np.transpose(clt_to_save, (1, 2, 0))
            clt_to_save=(clt_to_save+1)*127.5
            print(img_to_save.shape)
            cv2.imwrite(image_path, img_to_save)
            cv2.imwrite(cloth_path, clt_to_save)
            image=vae.encode(image.type(torch.float32)).latent_dist.sample()
            cloth=vae.encode(cloth.type(torch.float32)).latent_dist.sample()
            agn=vae.encode(total_img[2].to(device).type(torch.float32)).latent_dist.sample()
            agn_mask=vae.encode(total_img[3].to(device).type(torch.float32)).latent_dist.sample()
            img_densepose=vae.encode(total_img[1].to(device).type(torch.float32)).latent_dist.sample()
            image_100=model_diff.add_noise(image, torch.tensor(999))
            add=torch.cat([agn, agn_mask, img_densepose, cloth], dim=1)
            image_restored=model_diff.sample(image_100[0], unet, torch.tensor(999), add, cloth)
            image_restored=vae.decode(image_restored)[0]
            res_to_save=image_restored[0].detach().cpu().numpy()
            res_to_save=np.transpose(res_to_save, (1, 2, 0))
            res_to_save=(res_to_save+1)*127.5
            cv2.imwrite(out_path, res_to_save)
            break
