"""
使用 CycleGAN 实现莫奈风格图像转化为真实照片风格图片
在同一根目录下需要：
    datasets/monet2photo/trainA 文件夹存放莫奈风格画像
    datasets/monet2photo/trainB 文件夹存放真实风景图片
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
import os

# 设置参数
batch_size = 1
lr = 0.0002
epochs = 1000
input_channel_size = 3
output_channel_size = 3

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs("output_cyclegan", exist_ok=True)

# 数据预处理
transform = transforms.Compose([
    transforms.Resize((256, 256)),  # 统一图片大小，方便处理
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # RGB 图片归一化
])

# 加载数据集
dataset_A = ImageFolder(root="datasets/monet2photo/trainA", transform=transform)  # 莫奈画像
dataset_B = ImageFolder(root="datasets/monet2photo/trainB", transform=transform)  # 真实照片


dataloader_A = DataLoader(dataset_A, batch_size=batch_size, shuffle=True)
dataloader_B = DataLoader(dataset_B, batch_size=batch_size, shuffle=True)


# 残差块
class ResnetBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.ReflectionPad2d(1), # 反射填充，缓解填充带来的差异
            nn.Conv2d(dim, dim, kernel_size=3, padding=0, bias=False),
            nn.InstanceNorm2d(dim), # 实例归一化
            nn.ReLU(True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, kernel_size=3, padding=0, bias=False),
            nn.InstanceNorm2d(dim)
        )

    def forward(self, x):
        return x + self.conv_block(x)


# 生成器
class Generator(nn.Module):
    def __init__(self, input_channel_size, output_channel_size):
        super().__init__()
        self.main = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_channel_size, 64, kernel_size=7, padding=0, bias=False), # (4,3,256,256)->(4,64,256,256) 感受野（7*7）
            nn.InstanceNorm2d(64),
            nn.ReLU(True),

            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),# (4,64,256,256)->(4,128,128,128) 感受野（9*9）
            nn.InstanceNorm2d(128),
            nn.ReLU(True),

            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=False),# (4,128,128,128)->(4,256,64,64) 感受野（13*13）
            nn.InstanceNorm2d(256),
            nn.ReLU(True),

            ResnetBlock(256),
            ResnetBlock(256),
            ResnetBlock(256),
            ResnetBlock(256), # (4,256,64,64)->(4,256,64,64) 感受野（13*13）

            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False),# (4,256,64,64)->(4,128,128,128) 感受野（17*17）
            nn.InstanceNorm2d(128),
            nn.ReLU(True),

            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False),# (4,128,128,128)->(4,64,256,256) 感受野（25*25）
            nn.InstanceNorm2d(64),
            nn.ReLU(True),

            nn.ReflectionPad2d(3),
            nn.Conv2d(64, output_channel_size, kernel_size=7, padding=0),# (4,64,128,128)->(4,3,256,256) 感受野（31*31）
            nn.Tanh()
        )

    def forward(self, input):
        return self.main(input)


# 判别器
class Discriminator(nn.Module):
    def __init__(self, input_channel_size):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(input_channel_size, 64, kernel_size=4, stride=2, padding=1), # (4,3,256,256)->(4,64,128,128)
            nn.LeakyReLU(0.2, True),

            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=False), # (4,64,128,128)->(4,128,64,64)
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(0.2, True),

            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1, bias=False), # (4,128,64,64)->(4,256,32,32)
            nn.InstanceNorm2d(256),
            nn.LeakyReLU(0.2, True),

            nn.Conv2d(256, 1, kernel_size=4, stride=2, padding=1) # (4,256,32,32)->(4,1,16,16)
            # 使用更为高级的 PatchGAN 判别器而不是普通的判别器设计，这样设计能让模型更加关注局部区域的真实程度，这正对应了图像风格转化的任务特性
        )

    def forward(self, input):
        return self.model(input)


netG = Generator(input_channel_size, output_channel_size).to(device) # A->B
netF = Generator(output_channel_size, input_channel_size).to(device) # B->A
netD_A = Discriminator(input_channel_size).to(device) # 检验图像是否是真实的A
netD_B = Discriminator(output_channel_size).to(device) # 检验图像是否是真实的B

criterion_GAN = nn.MSELoss()
criterion_cycle = nn.L1Loss()

optimizer_GF = optim.Adam(list(netG.parameters()) + list(netF.parameters()), lr=lr, betas=(0.5, 0.999)) # 由于循环一致性的设计，需要将 G 和 F 两个模型绑定在一起联合优化，保证相同的优化速率和效果
optimizer_D_A = optim.Adam(netD_A.parameters(), lr=lr, betas=(0.5, 0.999))
optimizer_D_B = optim.Adam(netD_B.parameters(), lr=lr, betas=(0.5, 0.999))

# 训练循环
for epoch in range(epochs):
    for i, ((real_A, _), (real_B, _)) in enumerate(zip(dataloader_A, dataloader_B)):
        real_A = real_A.to(device)
        real_B = real_B.to(device)

        # 训练生成器
        optimizer_GF.zero_grad()
            # 生成转化之后的图像
        fake_B = netG(real_A)
        fake_A = netF(real_B)
        valid_B = torch.ones_like(netD_B(fake_B))
        valid_A = torch.ones_like(netD_A(fake_A))
            # 计算损失
                # 计算对抗性损失
        loss_G_GAN = criterion_GAN(netD_B(fake_B), valid_B)
        loss_F_GAN = criterion_GAN(netD_A(fake_A), valid_A)
                # 计算循环一致性损失
        recovered_A = netF(fake_B) # 将生成出的虚假图像 B 转化回去
        recovered_B = netG(fake_A) # 将生成出的虚假图像 A 转化回去
        loss_cycle_A = criterion_cycle(recovered_A, real_A)
        loss_cycle_B = criterion_cycle(recovered_B, real_B)
        total_cycle_loss = (loss_cycle_A + loss_cycle_B) * 10 # 10 是权重
            # 计算总损失值
        total_G_loss = loss_G_GAN + loss_F_GAN + total_cycle_loss
        total_G_loss.backward()
        optimizer_GF.step()


        # 训练判别器
            # 训练判别器 A
        real_output_A = netD_A(real_A) # 来自真实的图像
        fake_output_A = netD_A(fake_A.detach()) # 来自虚假的图像，细节阻断梯度传播，防止波及生成fake_A 的 netF
        loss_D_real_A = criterion_GAN(real_output_A, torch.ones_like(real_output_A)) # 应接近于 1
        loss_D_fake_A = criterion_GAN(fake_output_A, torch.zeros_like(fake_output_A)) # 应接近于 0
        loss_D_A = 0.5 * (loss_D_real_A + loss_D_fake_A) # 0.5 代表取均值
        loss_D_A.backward()
        optimizer_D_A.step()
            # 训练判别器 B
            # 与 A 类似
        real_output_B = netD_B(real_B)
        fake_output_B = netD_B(fake_B.detach())
        loss_D_real_B = criterion_GAN(real_output_B, torch.ones_like(real_output_B))
        loss_D_fake_B = criterion_GAN(fake_output_B, torch.zeros_like(fake_output_B))
        loss_D_B = 0.5 * (loss_D_real_B + loss_D_fake_B)
        loss_D_B.backward()
        optimizer_D_B.step()


        if i % 100 == 0:
            print(f"Epoch [{epoch + 1}/{epochs}] Batch {i} G_loss: {total_G_loss.item():.4f} D_A: {loss_D_A.item():.4f} D_B: {loss_D_B.item():.4f}")
            # 保存生成器参数，方便加载使用
            torch.save(netG.state_dict(), "generator_monet2real.pth")  # monet->real 参数
            torch.save(netF.state_dict(), "generator_real2monet.pth")  # real->monet 参数