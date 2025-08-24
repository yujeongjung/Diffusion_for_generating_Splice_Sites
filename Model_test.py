import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleUNet(nn.Module):
    '''
    U-net is used to denoise the given data during  reverse process
    It predicts the noise which is used to calculate the mean of the Gaussian distribution
    '''
    def __init__(self, input_dim, num_features):
        super(SimpleUNet, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(input_dim, num_features, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(num_features, num_features, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Conv1d(num_features, num_features, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(num_features, input_dim, kernel_size=3, padding=1),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


class DDPM(nn.Module):
    def __init__(self, betas, unet):
        super(DDPM, self).__init__()
        self.betas = torch.tensor(betas, dtype=torch.float32)
        self.alphas = 1 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)  # cumulative product of all alphas. This is used to calculate noise (probability q)
        self.alphas_cumprod_prev = torch.cat([torch.tensor([1.0]), self.alphas_cumprod[:-1]])

        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1 - self.alphas_cumprod)
        self.posterior_variance = self.betas * (1 - self.alphas_cumprod_prev) / (1 - self.alphas_cumprod)

        self.unet = unet

    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)
        return (
            self.sqrt_alphas_cumprod[t].view(-1, 1, 1).to(x_start.device) * x_start +
            self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1).to(x_start.device) * noise
        )  # math.sqrt(a)*x + math.sqrt(1-a)*noise

    def p_losses(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)
        x_noisy = self.q_sample(x_start, t, noise)  # assign noise as q_sample
        predicted_noise = self.unet(x_noisy)  # predict noise by using Unet
        return F.mse_loss(predicted_noise, noise)  # calculate mean squared error between predicted and calculated noise 

    @torch.no_grad()  # set model to not calculate gradient
    def p_sample(self, x, t):  # x: sequence tensor
        betas_t = self.betas[t].to(x.device)
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t].to(x.device)
        sqrt_recip_alphas_t = torch.sqrt(1 / self.alphas[t]).to(x.device)
        sqrt_recipm1_alphas_t = torch.sqrt(1 / (1 - self.alphas_cumprod[t])).to(x.device)

        model_mean = sqrt_recip_alphas_t * (x - betas_t * sqrt_recipm1_alphas_t * self.unet(x))
        model_var = betas_t * sqrt_one_minus_alphas_cumprod_t
        noise = torch.randn_like(x) if t > 0 else torch.zeros_like(x)

        return model_mean + torch.sqrt(model_var) * noise

    @torch.no_grad()
    def sample(self, seq_length, batch_size, device):
        x = torch.randn((batch_size, 4, seq_length), device=device)  # x: initial noise, 4: dimension of one-hot encoded sequence
        for t in reversed(range(len(self.betas))):  # reverse process
            x = self.p_sample(x, t)  # update x by using preduction
        return x

