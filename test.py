import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from torch.utils.data import Dataset
import numpy as np
import torchvision.transforms
import torch.nn.functional as F
from diffusion.diffusion import Diffusion
from model2.unet import UNet
from preprocess import TrainDataset
from models.new_unet import UNetModel
from preprocess_test import TestDataset
device="cpu"
model_diff=Diffusion(1000, 0.9999, 0.98, device)
unet=UNet(c_in=15, device=device).to(device)
unet.load_state_dict(torch.load("")) #path


transform=transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize((64, 64)),
        torchvision.transforms.ToTensor(),
        lambda x: x * 255.0
    ])

main_path=main_path=''
pair=''

data=TestDataset(main_path, (64, 64), pair, None, transform, device)

loader=DataLoader(data, batch_size=1, shuffle=False)
# print(data.data)
for i, total_img in enumerate(loader):
    print(i)
    if i==19:
        image=total_img[0]
        cloth=total_img[4]
        # image=image.to(device)
        # print(image.shape)
        torch.save(image[0], '')
        torch.save(cloth[0], '')

        image_100=model_diff.add_noise(image, torch.tensor(999))
        # print(image_500.shape)
        torch.save(image_100[0][0], '')
        add=torch.cat([total_img[2], total_img[3], total_img[1], total_img[4]], dim=1)
        image_restored=model_diff.sample(image_100[0], unet, torch.tensor(999), add)
        torch.save(image_restored[0], '')
        break
