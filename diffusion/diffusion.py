import torch
class Diffusion:
  def __init__(self, no_steps, alpha_start, alpha_end, device):
    self.steps=no_steps
    self.alpha_0=alpha_start
    self.alpha_T=alpha_end
    self.device=device
    self.alphas=self.list_alphas()
    self.alpha_hats=torch.cumprod(self.alphas, dim=0)




  def add_noise(self, x, t, noise=None):
    '''
    x ~ p(x_t | x_0)

    '''
    alpha_t=self.alpha_hats[t].reshape(-1, 1, 1, 1)
    # print(t.shape)
    # alpha_t=self.alpha_hats[t]
    # print(alpha_t)
    if noise is None:
      noise=torch.randn_like(x).to(self.device)
    # return (alpha_t**0.5)*x+((1-alpha_t)**0.5)*nosie, nosie
    return torch.sqrt(alpha_t)*x + torch.sqrt(1.0 - alpha_t)*noise, noise



  def list_alphas(self):
    return torch.linspace(self.alpha_0, self.alpha_T, self.steps).to(self.device)


  def sample(self, x, model, start, add):
    model.eval()
    with torch.no_grad():
      for i in range(start-1, 0, -1):
        if i>1:
          z=torch.randn_like(x).to(self.device)
        else:
          z=torch.zeros_like(x).to(self.device)
        # k=model(x, torch.tensor([i]).to(self.device), torch.tensor([1]).to(self.device).to(torch.int32))
        # print(x.shape)
        # print(add.shape)
        x1=torch.cat([x, add], dim=1)
        # print(x1.shape)
        k=model(x1, torch.tensor([i]*x.shape[0]).to(self.device))
        # k=k[:, :3, :, :]
        # print(k.shape)
        # k=k[0]
        # k=k[:3]
        # k=k.unsqueeze(0)
        # print(k.shape)
        # a1=self.alphas[i].unsqueeze(0).unsqueeze(0).unsqueeze(0)
        # b1=1-self.alphas[i].unsqueeze(0).unsqueeze(0).unsqueeze(0)
        # c1=self.alpha_hats[i].unsqueeze(0).unsqueeze(0).unsqueeze(0)
        # x=1/torch.sqrt(a1)*(x - ((1-a1)/torch.sqrt(1-c1))*k)+torch.sqrt(b1)*z
        
        x=(1/torch.sqrt(self.alphas[i]))*(x - ((1-self.alphas[i])/torch.sqrt(1-self.alpha_hats[i]))*k)+torch.sqrt(1-self.alphas[i])*z

    return x


  def q_mean_var(self, x_0, t):
    '''
    q(x_t|x_0)

    '''

    mean=torch.sqrt(self.alpha_hats[t])*x_0
    var=1-self.alpha_hats[t]
    return mean, var

  def q_sample(self, x_0, t):
    mean, var=self.q_mean_var(x_0, t)
    epsilon=torch.randn_like(x_0)
    sample=epsilon*torch.sqrt(var)+mean
    return sample


  def q_posterior(self, x_0, t):
    '''
    q(x_t-1 | x_t, x_0)

    '''
    x_t=self.q_sample(x_0, t)
    mean=(torch.sqrt(self.alpha_hats[t-1])*(1-self.alphas[t])/(1-self.alpha_hats[t]))*x_0 + (torch.sqrt(self.alphas[t])*(1-self.alpha_hats[t-1])/(1-self.alpha_hats[t]))*x_t
    var=((1-self.alpha_hats[t-1])/(1-self.alpha_hats[t]))*(1-self.alphas[t])
    return mean, var

  def p_mean_var(self, x_t, t, model):
    '''
    p(x_t-1 | x_t)
    
    '''
    mean=(1/torch.sqrt(self.alphas[t]))*(x_t - ((1-self.alphas[t])/torch.sqrt(1-self.alpha_hats[t]))*model(x_t, t))
    # var=((1-self.alpha_hats[t-1])/(1-self.alpha_hats[t]))*(1-self.alphas[t])
    var=1-self.alphas[t]
    return mean, var