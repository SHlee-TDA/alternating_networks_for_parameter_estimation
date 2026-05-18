import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.SiLU(), 
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.SiLU(),
            nn.Linear(dim, dim)
        )

    def forward(self, x):
        return x + self.net(x)

def build_mlp(input_dim, output_dim, hidden_dims):
    layers = []
    
    layers.append(nn.Linear(input_dim, hidden_dims[0]))
    
    for i in range(len(hidden_dims) - 1):
        assert hidden_dims[i] == hidden_dims[i+1]
        layers.append(ResidualBlock(hidden_dims[i]))
        
    layers.append(nn.LayerNorm(hidden_dims[-1]))
    layers.append(nn.SiLU())
    layers.append(nn.Linear(hidden_dims[-1], output_dim))
    
    return nn.Sequential(*layers)

class  BaselineCVAE(nn.Module):
    def __init__(self, x_dim, theta_dim, y_dim, latent_dim=16, hidden_dims=[256, 256, 256, 256]):
        super().__init__()
        self.latent_dim = latent_dim
        
        encoder_input_dim = y_dim + x_dim + theta_dim
        self.encoder_net = build_mlp(encoder_input_dim, hidden_dims[-1], hidden_dims[:-1])
        self.fc_mu = nn.Linear(hidden_dims[-1], latent_dim)
        self.fc_logvar = nn.Linear(hidden_dims[-1], latent_dim)
        
        decoder_input_dim = latent_dim + x_dim + theta_dim
        self.decoder_net = nn.Sequential(
            build_mlp(decoder_input_dim, theta_dim, hidden_dims),
            nn.Tanh()
        )
    def encode(self, x, y, theta):
        inputs = torch.cat([x, y, theta], dim=-1)
        h = self.encoder_net(inputs)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z, x, theta):
        inputs = torch.cat([z, x, theta], dim=-1)
        theta_hat = self.decoder_net(inputs)
        return theta_hat
    
    def forward(self, x, y, theta):
        mu, logvar = self.encode(x, y, theta)
        z = self.reparameterize(mu, logvar)
        theta_hat = self.decode(z, x, theta)
        return theta_hat, mu, logvar
    
    def sample(self, x, theta, num_samples=1):
        batch_size = x.size(0)
        z = torch.randn(num_samples * batch_size, self.latent_dim).to(x.device)
        
        x_repeated = x.repeat_interleave(num_samples, dim=0)
        theta_repeated = theta.repeat_interleave(num_samples, dim=0)
        
        theta_sampled = self.decode(z, x_repeated, theta_repeated)
        return theta_sampled.view(batch_size, num_samples, -1)
    


class HiddenStateCVAE(nn.Module):
    def __init__(self, x_dim, theta_dim, y_dim, latent_dim=16, hidden_dims=[256, 256, 256, 256]):
        super().__init__()
        self.latent_dim = latent_dim
        
        encoder_input_dim = y_dim + x_dim + theta_dim
        self.encoder_net = build_mlp(encoder_input_dim, hidden_dims[-1], hidden_dims[:-1])
        self.fc_mu = nn.Linear(hidden_dims[-1], latent_dim)
        self.fc_logvar = nn.Linear(hidden_dims[-1], latent_dim)
        
        decoder_input_dim = latent_dim + x_dim + theta_dim
        self.decoder_net = nn.Sequential(
            build_mlp(decoder_input_dim, y_dim, hidden_dims),
            nn.Sigmoid()
        )
    def encode(self, x, y, theta):
        inputs = torch.cat([x, y, theta], dim=-1)
        h = self.encoder_net(inputs)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, x, theta):
        inputs = torch.cat([z, x, theta], dim=-1)
        y_hat = self.decoder_net(inputs)
        return y_hat

    def forward(self, x, y, theta):
        mu, logvar = self.encode(x, y, theta)
        z = self.reparameterize(mu, logvar)
        y_hat = self.decode(z, x, theta)
        return y_hat, mu, logvar

    def sample(self, x, theta, num_samples=1):
        batch_size = x.size(0)
        z = torch.randn(num_samples * batch_size, self.latent_dim).to(x.device)
        
        x_repeated = x.repeat_interleave(num_samples, dim=0)
        theta_repeated = theta.repeat_interleave(num_samples, dim=0)
        
        y_sampled = self.decode(z, x_repeated, theta_repeated)
        return y_sampled.view(batch_size, num_samples, -1)
    

class ParameterCVAE(nn.Module):
    def __init__(self, x_dim, y_dim, theta_dim, latent_dim=8, hidden_dims=[256, 256, 256, 256]):
        super().__init__()
        self.latent_dim = latent_dim
        
        encoder_input_dim = theta_dim + x_dim + y_dim
        self.encoder_net = build_mlp(encoder_input_dim, hidden_dims[-1], hidden_dims[:-1])
        self.fc_mu = nn.Linear(hidden_dims[-1], latent_dim)
        self.fc_logvar = nn.Linear(hidden_dims[-1], latent_dim)
        
        decoder_input_dim = latent_dim + x_dim + y_dim
        self.decoder_net = nn.Sequential(
            build_mlp(decoder_input_dim, theta_dim, hidden_dims),
            nn.Tanh()
        )
    def encode(self, x, y, theta):
        inputs = torch.cat([x, y, theta], dim=-1)
        h = self.encoder_net(inputs)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, x, y):
        inputs = torch.cat([z, x, y], dim=-1)
        theta_hat = self.decoder_net(inputs)
        return theta_hat

    def forward(self, x, y, theta):
        mu, logvar = self.encode(x, y, theta)
        z = self.reparameterize(mu, logvar)
        theta_hat = self.decode(z, x, y)
        return theta_hat, mu, logvar

    def sample(self, x, y, num_samples=1):
        batch_size = x.size(0)
        z = torch.randn(num_samples * batch_size, self.latent_dim).to(x.device)
        
        x_repeated = x.repeat_interleave(num_samples, dim=0)
        y_repeated = y.repeat_interleave(num_samples, dim=0)
        
        theta_sampled = self.decode(z, x_repeated, y_repeated)
        return theta_sampled.view(batch_size, num_samples, -1)
    
def elbo_loss(recon_x, x, mu, logvar, beta=1.0):
    recon_loss = F.mse_loss(recon_x, x, reduction='none').sum(dim=1).mean()
    kld_loss = -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))
    
    total_loss = recon_loss + beta * kld_loss
    return total_loss, recon_loss, kld_loss