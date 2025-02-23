import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class GANParams:
    def __init__(self, latent_dim=100, hidden_dim=256, output_dim=784, lr=0.0002, beta1=0.5):
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.lr = lr
        self.beta1 = beta1

class Generator(nn.Module):
    def __init__(self, latent_dim, hidden_dim, output_dim):
        super().__init__()
        self.main = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 784),
            nn.Tanh()
        )

    def forward(self, x):
        return self.main(x)

class Discriminator(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.main = nn.Sequential(
            nn.Linear(784, 512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.main(x)

class GAN:
    def __init__(self, params=None):
        if params is None:
            params = GANParams()
        self.latent_dim = params.latent_dim
        self.hidden_dim = params.hidden_dim
        self.output_dim = params.output_dim
        self.lr = params.lr
        self.beta1 = params.beta1
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # 初始化生成器和判别器
        self.generator = Generator(self.latent_dim, self.hidden_dim, self.output_dim).to(self.device)
        self.discriminator = Discriminator(self.output_dim, self.hidden_dim).to(self.device)

        # 优化器
        self.g_optimizer = optim.Adam(self.generator.parameters(), lr=self.lr, betas=(self.beta1, 0.999))
        self.d_optimizer = optim.Adam(self.discriminator.parameters(), lr=self.lr, betas=(self.beta1, 0.999))

        # 损失函数
        self.criterion = nn.BCELoss()

    def train(self, real_data, num_epochs=200, batch_size=64):
        history = {
            'g_losses': [],
            'd_losses': []
        }

        real_data = torch.tensor(real_data, dtype=torch.float32).to(self.device)
        dataset = torch.utils.data.TensorDataset(real_data)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

        for epoch in range(num_epochs):
            for batch in dataloader:
                batch_size = batch[0].size(0)
                real_samples = batch[0]

                # 训练判别器
                self.d_optimizer.zero_grad()
                label_real = torch.ones(batch_size, 1).to(self.device)
                label_fake = torch.zeros(batch_size, 1).to(self.device)

                output_real = self.discriminator(real_samples)
                d_loss_real = self.criterion(output_real, label_real)

                noise = torch.randn(batch_size, self.latent_dim).to(self.device)
                fake_samples = self.generator(noise)
                output_fake = self.discriminator(fake_samples.detach())
                d_loss_fake = self.criterion(output_fake, label_fake)

                d_loss = d_loss_real + d_loss_fake
                d_loss.backward()
                self.d_optimizer.step()

                # 训练生成器
                self.g_optimizer.zero_grad()
                output_fake = self.discriminator(fake_samples)
                g_loss = self.criterion(output_fake, label_real)
                g_loss.backward()
                self.g_optimizer.step()

            history['d_losses'].append(d_loss.item())
            history['g_losses'].append(g_loss.item())

        return history

    def generate(self, num_samples):
        self.generator.eval()
        with torch.no_grad():
            noise = torch.randn(num_samples, self.latent_dim).to(self.device)
            generated_samples = self.generator(noise)
            return generated_samples.cpu().numpy()

    def discriminate(self, samples):
        self.discriminator.eval()
        samples_tensor = torch.tensor(samples, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            predictions = self.discriminator(samples_tensor)
            return predictions.cpu().numpy()

    def dispose(self):
        self.generator = None
        self.discriminator = None