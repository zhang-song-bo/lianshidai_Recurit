import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torchvision.utils as vutils
import os

# 设置参数
batch_size = 128
lr = 0.0002
noise_dim = 100
epochs = 20

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs("output_ganpro", exist_ok=True)

# 数据预处理
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),  # 修改处理的默认的图像通道数
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))  # 将图像归一化到 [-1, 1]
])

# 使用 ImageFolder 读取数据
dataset = datasets.ImageFolder(root='data/mnist_jpg', transform=transform)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)


class Generator(nn.Module):
    def __init__(self, noise_dim):
        """
        生成器，将输入的噪声通过 MLP
        :param noise_dim: 输入的噪声维度
        """
        super().__init__()
        self.main = nn.Sequential(
            # MLP
            nn.Linear(noise_dim, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2, True),

            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2, True),

            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2, True),

            nn.Linear(1024, 28 * 28),
            nn.Tanh()
        )

    def forward(self, input):
        """
        :param input: 输入的噪声数据
        :return: 通过 MLP 生成的图像
        """
        output = self.main(input)
        output = output.view(-1, 1, 28, 28)
        return output

class Discriminator(nn.Module):
    def __init__(self):
        """
        判别器，将输入的图像通过 MLP 进行二分类
        """
        super().__init__()
        self.main = nn.Sequential(
            # MLP
            nn.Linear(28 * 28, 1024),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),

            # 使用 Sigmoid 映射到 [0,1]
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, input):
        output = self.main(input.view(-1, 28 * 28))
        return output

# 模型,优化器,损失函数
netG = Generator(noise_dim).to(device)
netD = Discriminator().to(device)

criterion = nn.BCELoss()

optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(0.5, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(0.5, 0.999))

# 训练过程
for epoch in range(epochs):
    for i, (data, _) in enumerate(dataloader):
        # 训练辨别器
            # 使 D_model 对真实数据集里的真实图像进行分类判断，将 label 视作 1
        netD.zero_grad()
        real_images = data.to(device)
        batch_size = real_images.size(0)

        label_real = torch.full((batch_size, 1), 1.0, device=device)  # 真实标签为 1
        output_real = netD(real_images)
        lossD_real = criterion(output_real, label_real)

            # 使 D_model 对 G_model 生成的虚假图像进行分类判断，将 label 视作 0
        noise = torch.randn(batch_size, noise_dim, device=device)
        fake_images = netG(noise)
        label_fake = torch.full((batch_size, 1), 0.0, device=device)  # 假图像标签为 0
        output_fake = netD(fake_images.detach())  # detach 防止梯度流向生成器
        lossD_fake = criterion(output_fake, label_fake)

        # 更新判别器参数
        lossD = lossD_real + lossD_fake
        lossD.backward()
        optimizerD.step()


        # 训练生成器
        netG.zero_grad()
        label_gen = torch.full((batch_size, 1), 1.0, device=device)
        output_gen = netD(fake_images)
        lossG = criterion(output_gen, label_gen)
        lossG.backward()
        optimizerG.step()

        # 输出训练信息
        if i % 100 == 0:
            print(f"Epoch [{epoch + 1}/{epochs}] Batch {i}/{len(dataloader)} "
                  f"Loss_D: {(lossD_real + lossD_fake).item():.4f} Loss_G: {lossG.item():.4f}")

    # 每个 epoch 结束后保存一批生成的图片
    with torch.no_grad():
        fixed_noise = torch.randn(16, noise_dim, device=device)
        fake = netG(fixed_noise).detach()
    vutils.save_image(fake, f"output_ganpro/fake_samples_epoch_{epoch + 1}.png", normalize=True)