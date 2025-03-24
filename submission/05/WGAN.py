import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torchvision.utils as vutils
import os

batch_size = 128
lr = 0.00005  # WGAN采用较小的学习率
noise_dim = 100
epochs = 20
channel_size = 1
critic_iter = 5  # 判别器训练次数
weight_clip = 0.01  # 权重裁剪范围

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs("output_wgan", exist_ok=True)

# 数据预处理
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),  # 修改默认的图像通道数
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# 使用 ImageFolder 读取数据
dataset = datasets.ImageFolder(root='data/mnist_jpg', transform=transform)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)


class Generator(nn.Module):
    """
    基于卷积层的生成器
    和 DCGAN 相同
    """
    def __init__(self, noise_dim, channel_size):
        super().__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(noise_dim, 128, kernel_size=7, stride=1, padding=0),
            nn.BatchNorm2d(128),
            nn.ReLU(True),

            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),

            nn.ConvTranspose2d(64, channel_size, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )

    def forward(self, input):
        return self.main(input)


class Discriminator(nn.Module):
    """
    基于卷积层的判别器
    和 DCGAN 相同
    """
    def __init__(self, channel_size):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(channel_size, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.flatten = nn.Flatten()
        self.fc = nn.Linear(128 * 7 * 7, 1)

    def forward(self, input):
        x = self.main(input)
        x = self.flatten(x)
        output = self.fc(x)
        return output


# 模型,优化器
netG = Generator(noise_dim, channel_size).to(device)
netD = Discriminator(channel_size).to(device)

# 原论文建议使用 RMSprop 优化器
optimizerD = optim.RMSprop(netD.parameters(), lr=lr)
optimizerG = optim.RMSprop(netG.parameters(), lr=lr)

# 训练过程
for epoch in range(epochs):
    for i, (data, _) in enumerate(dataloader):
        real_imgs = data.to(device)
        batch_size = real_imgs.size(0)

        # 判别器训练
        for _ in range(critic_iter):
            netD.zero_grad()
            noise = torch.randn(batch_size, noise_dim, 1, 1, device=device)
            fake_imgs = netG(noise)

            # 计算 Wasserstein 损失
            lossD = -netD(real_imgs).mean() + netD(fake_imgs.detach()).mean()
            lossD.backward()
            optimizerD.step()

            # 权重裁剪
            for p in netD.parameters():
                p.data.clamp_(-weight_clip, weight_clip)

        # 生成器训练
        netG.zero_grad()
        fake_imgs = netG(noise)

        # 计算 Wasserstein 损失
        lossG = -netD(fake_imgs).mean()
        lossG.backward()
        optimizerG.step()

        if i % 100 == 0:
            print(f"Epoch [{epoch + 1}/{epochs}] Batch {i}/{len(dataloader)} Loss_D: {lossD.item():.4f} Loss_G: {lossG.item():.4f}")

    # 保存生成结果
    with torch.no_grad():
        fixed_noise = torch.randn(16, noise_dim, 1, 1, device=device)
        fake = netG(fixed_noise)
    vutils.save_image(fake, f"output_wgan/fake_samples_epoch_{epoch + 1}.png", normalize=True)
