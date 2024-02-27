import torch
from tqdm import tqdm

class Diffusion:
    def __init__(self, steps=1000, beta_begin=1e-4, beta_end=2e-2, img_size=256, device="cuda"):
        self.steps = steps
        self.beta = torch.linspace(beta_begin, beta_end, steps).to(device)
        self.image_size = img_size
        self.device = device
        self.alpha = 1 - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)

    def generate_noised_images(self, x, t):
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None, None, None] # match x's dimension
        sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat[t])[:, None, None, None]
        epsilon = torch.randn_like(x)
        return sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * epsilon, epsilon

    def sample_timesteps(self, n):
        return torch.randint(low=1, high=self.steps, size=(n,))
    
    def infer(self, model, n): # 这里的n似乎代表batch_size
        model.eval() # 切换为评估模式
        with torch.no_grad():
            x = torch.randn(n, 3, self.image_size, self.image_size).to(self.device)
            for i in tqdm(reversed(range(1, self.steps))):
                t = torch.full((n,), i, dtype=torch.long, device=self.device)
                predicted_noise = model(x, t)
                alpha = self.alpha[t][:, None, None, None]
                alpha_hat = self.alpha_hat[t][:, None, None, None]
                beta = self.beta[t][:, None, None, None]
                noise = torch.zeros_like(x) if i == 0 else torch.randn_like(x)
                x = 1 / torch.sqrt(alpha) * (x - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * predicted_noise) + torch.sqrt(beta) * noise 
        model.train()
        x = (x.clamp(-1, 1) + 1) / 2
        x = (x * 255).type(torch.uint8)
        return x 

        