import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn import L1Loss

class mib(nn.Module):
    def __init__(self, in_dim, compressed_dim, m_dim=1024, beta=1e-3):
        super().__init__()

        self.in_dim = in_dim
        self.compressed_dim = compressed_dim
        self.m_dim = m_dim
        self.beta = beta 

        # Encoder: Input -> Hidden -> Mu/Std
        self.encoder = nn.Sequential(
            nn.Linear(self.in_dim, self.m_dim),
            nn.ReLU(inplace=True)
        )

        self.fc_mu = nn.Linear(self.m_dim, self.compressed_dim)
        self.fc_std = nn.Linear(self.m_dim, self.compressed_dim)

        # Decoder: Compressed -> Prediction 
        self.decoder = nn.Linear(self.compressed_dim, 1)

    def encode(self, x):
        x = self.encoder(x)
        # softplus(x-5) 是为了保证 std 为正且数值稳定
        return self.fc_mu(x), F.softplus(self.fc_std(x)-5, beta=1)

    def decode(self, z):
        return self.decoder(z)

    def reparameterise(self, mu, std):
        if self.training:
            eps = torch.randn_like(std)
            return mu + std*eps
        else:
            # 推理时不采样，直接用均值
            return mu

    def loss_function(self, y_pred, y, mu, std):
        loss_fct = L1Loss()
        # 确保 y 的形状匹配
        CE = loss_fct(y_pred.view(-1), y.view(-1))
        # KL 散度: 0.5 * sum(mu^2 + sigma^2 - log(sigma^2) - 1)
        # 这里使用 mean 来保持 loss 规模与 CE 一致
        KL = 0.5 * torch.mean(mu.pow(2) + std.pow(2) - 2*std.log() - 1)
        return (self.beta * KL + CE)

    def forward(self, x, label_ids=None):
        mu, std = self.encode(x)
        z = self.reparameterise(mu, std) 
        output = self.decode(z)
        
        loss = None
        if label_ids is not None:
            loss = self.loss_function(output, label_ids, mu, std)
        
        return z, output, loss