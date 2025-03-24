import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torchvision.utils as vutils
import os

batch_size = 128
lr = 0.0002
noise_dim = 100
epochs = 20
channel_size = 1

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs("output_dcgan", exist_ok=True)

# 数据预处理
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# 使用 ImageFolder 读取数据
dataset = datasets.ImageFolder(root='data/mnist_jpg', transform=transform)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)


class Generator(nn.Module):
    def __init__(self, noise_dim, channel_size):
        """
        基于卷积层的生成器
        实现生成器的若干卷积层的叠加
        :param noise_dim: 输入的噪音维度
        :param channel_size: 目标图像的通道数
        """
        super().__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(noise_dim, 64 * 2, kernel_size=7, stride=1, padding=0), # (batch_size, noise_dim, 1, 1) -> (batch_size, 128, 7, 7)
            nn.BatchNorm2d(64 * 2),
            nn.ReLU(True),

            nn.ConvTranspose2d(64 * 2, 64, kernel_size=4, stride=2, padding=1), # (batch_size, 128, 7, 7) -> (batch_size, 64, 14, 14)
            nn.BatchNorm2d(64),
            nn.ReLU(True),

            nn.ConvTranspose2d(64, channel_size, kernel_size=4, stride=2, padding=1), # (batch_size, 64, 14, 14) -> (batch_size, channel_size, 28, 28)
            nn.Tanh()
        )

    def forward(self, input):
        """完成前向传播"""
        return self.main(input)


class Discriminator(nn.Module):
    def __init__(self, channel_size):
        """
        基于卷积层的判别器
        实现判别器的若干卷积层的叠加
        :param channel_size: 欲判别的图像通道数
        """
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(channel_size, 64, kernel_size=4, stride=2, padding=1), # (batch_size, channel_size, 28, 28)->(batch_size, 64, 14, 14)
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(64, 64 * 2, kernel_size=4, stride=2, padding=1), # (batch_size, 64, 14, 14)->(batch_size, 128, 7, 7)
            nn.BatchNorm2d(64 * 2),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.flatten = nn.Flatten()

        self.fc = nn.Sequential(
            nn.Linear(128 * 7 * 7, 1),
            nn.Sigmoid()
        )

    def forward(self, input):
        """
        完成前向传播
        :param input: 欲判别的图像数据
        :return: 返回分类结果
        """
        x = self.main(input)
        x = self.flatten(x)
        output = self.fc(x)
        return output

# 模型,优化器,损失函数
netG = Generator(noise_dim, channel_size).to(device)
netD = Discriminator(channel_size).to(device)

criterion = nn.BCELoss()

optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(0.5, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(0.5, 0.999))

# 训练过程
for epoch in range(epochs):
    for i, (data, _) in enumerate(dataloader):
        # 训练判别器
            # 使 D_model 对真实数据集里的真实图像进行分类判断，将 label 视作 1
        netD.zero_grad()
        real_imgs = data.to(device)
        batch_size = real_imgs.size(0)

        label_real = torch.full((batch_size, 1), 1.0, device=device)
        output_real = netD(real_imgs)
        lossD_real = criterion(output_real, label_real)

            # 使 D_model 对 G_model 生成的虚假图像进行分类判断，将 label 视作 0
        noise = torch.randn(batch_size, noise_dim, 1, 1, device=device)
        fake_imgs = netG(noise)
        label_fake = torch.full((batch_size, 1), 0.0, device=device)
        output_fake = netD(fake_imgs.detach())
        lossD_fake = criterion(output_fake, label_fake)
            # 通过真实图像上的损失和虚假图像上的损失相加，得到原论文中的损失表达，可以衡量模型在真假图形分类上的表现
        lossD = lossD_real + lossD_fake
        lossD.backward()
        optimizerD.step()

        # 训练生成器
        netG.zero_grad()
        label_gen = torch.full((batch_size, 1), 1.0, device=device)   # 生成器希望判别器将假样本判为真实,故标签设置为 1
        output_gen = netD(fake_imgs)
        lossG = criterion(output_gen, label_gen)
        lossG.backward()
        optimizerG.step()

        if i % 100 == 0:
            print(f"Epoch [{epoch + 1}/{epochs}] Batch {i}/{len(dataloader)} Loss_D: {(lossD_real + lossD_fake).item():.4f} Loss_G: {lossG.item():.4f}")

    # 保存每个 epoch 的生成结果
    with torch.no_grad():
        fixed_noise = torch.randn(16, noise_dim, 1, 1, device=device)
        fake = netG(fixed_noise)
    vutils.save_image(fake, f"output_dcgan/fake_samples_epoch_{epoch + 1}.png", normalize=True)