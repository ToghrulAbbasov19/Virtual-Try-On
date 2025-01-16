import torch
from torch.utils.data import Dataset
import os
from PIL import Image
# import albumentations
import cv2
import numpy as np
types=["agnostic-mask", "agnostic-v3.2", "cloth", "cloth-mask", "gt_cloth_warped_mask", "image", "image-densepose"]


class VTODataset(Dataset):
  def __init__(self, main_path, size, pairs_path, types, transform, device):
    super().__init__()
    self.main_path=main_path
    self.types=types
    self.transform=transform
    self.size=size
    self.pairs_path=pairs_path
    self.pairs=self.get_pairs()
    self.device=device
    self.imgs=os.listdir(os.path.join(self.main_path, "image"))
    self.clths=os.listdir(os.path.join(self.main_path, "cloth"))

  def __len__(self):
    # return len(self.pairs["image"])
    return len(self.imgs)

  def get_pairs(self):
    pairs={}
    pairs["image"]=[]
    pairs["cloth"]=[]
    with open(self.pairs_path, "r") as file:
        for line in file.readlines():
            image, cloth=line.strip().split()
            pairs["image"].append(image)
            pairs["cloth"].append(cloth)
        file.close()
    return pairs


  def read(self, tipe, idx):
    # print("type")
    # print(type)
    path=os.path.join(self.main_path, tipe)
    path=os.path.join(path, idx)
    img=cv2.imread(path)
    img=cv2.resize(img, self.size)
    # img=img.reshape((3, self.size[0], self.size[1]))
    img=np.transpose(img, (2, 0, 1))
    img=img/127.5-1
    # img=torch.tensor(img)
    # img=img.permute(2, 1, 0)
    # print(img.shape)
    return img


  def __getitem__(self, index):
    image=self.imgs[index]
    cloth=self.clths[index]
    images=[]
    images.append(self.read("image", image))
    # print(1)
    images.append(self.read("image-densepose", image))
    # print(2)
    images.append(self.read("agnostic-v3.2", image))
    # print(3)
    # print(image[:-4]+"_mask.jpg")
    images.append(self.read("agnostic-mask", image[:-4]+"_mask.png"))
    # print(4)
    images.append(self.read("cloth", cloth))
    # print(5)
    images.append(self.read("cloth-mask", cloth))
    images.append(self.read("gt_cloth_warped_mask", image))
    # images=torch.tensor(images)
    


    return images