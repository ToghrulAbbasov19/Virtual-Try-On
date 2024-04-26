from tqdm import tqdm
import torch
def train(train_loader, model_diff, unet, loss_func, optim, epochs, device, save_path, save_epochs):
  unet.train()
  # optim.zero_grad()
  for i in range(epochs):
    tk = tqdm(train_loader, desc='EPOCH' + '[TRAIN]' + str(i) + "/" + str(epochs))
    for j, total_img in enumerate(tk):
      image=total_img[0]
      t=torch.randint(1, 1000, size=(image.shape[0], )).to(device)
      noised_image, noise=model_diff.add_noise(image, t)
      concat=torch.cat([noised_image, total_img[2], total_img[3], total_img[1], total_img[4]], dim=1)
      # print(concat.shape)
      pred=unet(concat, t)
      model_loss=loss_func(noise, pred)
      # model_loss = torch.mean((noise - pred) ** 2, dim = list(range(1, len(noise.shape))))
      # model_loss = (model_loss*1/noise.shape[0]).mean()
      optim.zero_grad()
      model_loss.backward()
      optim.step()
      print(f"Loss:{model_loss}")
    if (i+1) % save_epochs==0:
      torch.save(unet.state_dict(), save_path)
      torch.save(optim.state_dict(), save_path[:-3]+"optim.pt")