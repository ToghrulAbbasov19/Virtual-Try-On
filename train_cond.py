from tqdm import tqdm
import torch
import os
def train(train_loader, model_diff, unet, vae, loss_func, optim, epochs, device, save_dir, save_epoch):
  unet.train()
  os.makedirs(save_dir, exist_ok=True)
  for i in range(epochs):
    model_path=os.path.join(save_dir, "model.pt")
    optim_path=os.path.join(save_dir, "optim.pt")
    tk = tqdm(train_loader, desc='EPOCH' + '[TRAIN]' + str(i) + "/" + str(epochs))
    for j, total_img in enumerate(tk):
      image=total_img[0].to(device).type(torch.float32)
      cloth=total_img[4].to(device).type(torch.float32)

      image=vae.encode(image).latent_dist.sample()
      cloth=vae.encode(cloth).latent_dist.sample()
      agn=vae.encode(total_img[2].to(device).type(torch.float32)).latent_dist.sample()
      agn_mask=vae.encode(total_img[3].to(device).type(torch.float32)).latent_dist.sample()
      img_densepose=vae.encode(total_img[1].to(device).type(torch.float32)).latent_dist.sample()

      t=torch.randint(1, 1000, size=(image.shape[0], )).to(device)
      noised_image, noise=model_diff.add_noise(image, t)
      concat=torch.cat([noised_image, agn, agn_mask, img_densepose, cloth], dim=1)
      pred=unet(concat, t, cloth)
      
      model_loss=loss_func(noise, pred)
      optim.zero_grad()
      model_loss.backward()
      optim.step()
      print(f"Loss:{model_loss}")
    if (i+1)%save_epoch==0:
      torch.save(unet.state_dict(), model_path)
      torch.save(optim.state_dict(), optim_path)