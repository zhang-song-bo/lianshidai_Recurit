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
lambda_gp = 10  # 梯度惩罚系数
critic_iterations = 5  # 判别器训练次数

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs("output_wgan_gp", exist_ok=True)

# 数据预处理
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

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


class Critic(nn.Module):
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
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(128 * 7 * 7, 1)

    def forward(self, input):
        x = self.main(input)
        x = self.flatten(x)
        return self.fc(x)


def compute_gradient_penalty(critic, real_samples, fake_samples):
    """
    计算梯度惩罚
    """
    # 随机数 alpha作为插值的权重, interpolates 是在真实样本和假样本之间的插值数据
    alpha = torch.rand(real_samples.size(0), 1, 1, 1, device=device)
    interpolates = (alpha * real_samples + (1 - alpha) * fake_samples).requires_grad_(True)

    # 得到判别器的结果
    critic_interpolates = critic(interpolates)

    # torch.autograd.grad用于计算导数，将梯度计算的结果存储在 gradients 变量中
    gradients = torch.autograd.grad(outputs=critic_interpolates, inputs=interpolates,
                                    grad_outputs=torch.ones_like(critic_interpolates),
                                    create_graph=True, retain_graph=True, only_inputs=True)[0]

    # 将梯度展平 (Batch, channels, height, width) -> (Batch, channels * height * width)
    gradients = gradients.view(gradients.size(0), -1)

    # 计算梯度惩罚, 公式为((梯度的L2范数 - 1) ^ 2)的均值
        # 理想情况下，Lipschitz常数应该为1，因此梯度的L2范数（gradients.norm(2, dim=1)）应该接近1。
        # 如果它大于1或小于1，都会给模型带来惩罚，以倒逼判别器的梯度符合要求。
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty


# 模型,优化器
netG = Generator(noise_dim, channel_size).to(device)
netC = Critic(channel_size).to(device)

optimizerC = optim.Adam(netC.parameters(), lr=lr, betas=(0.5, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(0.5, 0.999))

# 训练过程
for epoch in range(epochs):
    for i, (data, _) in enumerate(dataloader):
        real_imgs = data.to(device)
        batch_size = real_imgs.size(0)

        # 训练判别器
        for _ in range(critic_iterations):
            netC.zero_grad()
            noise = torch.randn(batch_size, noise_dim, 1, 1, device=device)
            fake_imgs = netG(noise)

            # 计算 Wasserstein 损失
            lossC_real = -netC(real_imgs).mean() + netC(fake_imgs.detach()).mean()
            # 计算梯度惩罚
            gradient_penalty = compute_gradient_penalty(netC, real_imgs, fake_imgs.detach())
            lossC = lossC_real + lambda_gp * gradient_penalty # 增加梯度惩罚

            lossC.backward()
            optimizerC.step()

        # 训练生成器
        netG.zero_grad()
        fake_imgs = netG(noise)

        # 计算 Wasserstein 损失
        lossG = -netC(fake_imgs).mean()
        lossG.backward()
        optimizerG.step()

        if i % 100 == 0:
            print(f"Epoch [{epoch + 1}/{epochs}] Batch {i}/{len(dataloader)} Loss_C: {lossC.item():.4f} Loss_G: {lossG.item():.4f}")

    # 保存每个 epoch 的生成结果
    with torch.no_grad():
        fixed_noise = torch.randn(16, noise_dim, 1, 1, device=device)
        fake = netG(fixed_noise)
    vutils.save_image(fake, f"output_wgan_gp/fake_samples_epoch_{epoch + 1}.png", normalize=True)