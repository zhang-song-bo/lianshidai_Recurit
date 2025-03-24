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
num_classes = 10  # 数据集类别数

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs("output_acgan", exist_ok=True)

# 数据预处理和加载
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

dataset = datasets.ImageFolder(root='data/mnist_jpg', transform=transform)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)


class Generator(nn.Module):
    def __init__(self, noise_dim, num_classes, channel_size):
        """
        基于卷积层的生成器
        卷积层的部分和DCGAN完全相同，只是增加了类别嵌入，以学习到类别信息
        """
        super().__init__()

        # 将离散类别标签映射到连续向量空间
        self.label_emb = nn.Embedding(num_classes, noise_dim)

        self.main = nn.Sequential(
            nn.ConvTranspose2d(noise_dim * 2, 128, kernel_size=7, stride=1, padding=0),
            nn.BatchNorm2d(128),
            nn.ReLU(True),

            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),

            nn.ConvTranspose2d(64, channel_size, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )

    def forward(self, noise, labels):
        # 将类别嵌入
        label_embedding = self.label_emb(labels).unsqueeze(2).unsqueeze(3) # 变形以匹配噪声维度
        input = torch.cat([noise, label_embedding], dim=1)
        return self.main(input)


class Discriminator(nn.Module):
    def __init__(self, channel_size, num_classes):
        """
        基于卷积层的判别器
        卷积层的部分和DCGAN完全相同
        新增加类别分类头
        """
        super().__init__()

        self.main = nn.Sequential(
            nn.Conv2d(channel_size, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.flatten = nn.Flatten()
        self.fc_real_fake = nn.Linear(128 * 7 * 7, 1)  # 真假分类
        self.fc_class = nn.Linear(128 * 7 * 7, num_classes)  # 分类

    def forward(self, input):
        x = self.main(input)
        x = self.flatten(x)

        real_fake = torch.sigmoid(self.fc_real_fake(x)) # 需要添加 softmax，因为 BCELoss 需要输入概率值
        class_output = self.fc_class(x)  # 分类输出（不加 softmax，交叉熵损失自带sigmoid）
        return real_fake, class_output

# 模型,优化器,损失函数
netG = Generator(noise_dim, num_classes, channel_size).to(device)
netD = Discriminator(channel_size, num_classes).to(device)

criterion_gan = nn.BCELoss()
criterion_class = nn.CrossEntropyLoss()

optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(0.5, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(0.5, 0.999))

# 训练过程
for epoch in range(epochs):
    for i, (data, labels) in enumerate(dataloader):
        batch_size = data.size(0)
        real_imgs, labels = data.to(device), labels.to(device)

        real_labels = torch.ones(batch_size, 1, device=device)
        fake_labels = torch.zeros(batch_size, 1, device=device)

        # 训练判别器
        netD.zero_grad()
            # 真实数据损失
        real_out, real_class = netD(real_imgs)
        lossD_real = criterion_gan(real_out, real_labels)
        lossD_real_class = criterion_class(real_class, labels)
            # 计算虚假数据损失
                # 生成假样本
        noise = torch.randn(batch_size, noise_dim, 1, 1, device=device)
        fake_labels_input = torch.randint(0, num_classes, (batch_size,), device=device)
        fake_imgs = netG(noise, fake_labels_input)
                # 假样本损失
        fake_out, fake_class = netD(fake_imgs.detach())
        lossD_fake = criterion_gan(fake_out, fake_labels)
        lossD_fake_class = criterion_class(fake_class, fake_labels_input)
            # 总判别器损失
        lossD = lossD_real + lossD_fake + lossD_real_class + lossD_fake_class
        lossD.backward()
        optimizerD.step()

        # 训练生成器
        netG.zero_grad()
        fake_out, fake_class = netD(fake_imgs)
            # 生成器希望判别器把假样本判为真实
        lossG_fake = criterion_gan(fake_out, real_labels)
        lossG_classification = criterion_class(fake_class, fake_labels_input)
            # 总生成器损失
        lossG = lossG_fake + lossG_classification
        lossG.backward()
        optimizerG.step()

        if i % 100 == 0:
            print(f"Epoch [{epoch + 1}/{epochs}] Batch {i}/{len(dataloader)} "
                  f"Loss_D: {lossD.item():.4f} Loss_G: {lossG.item():.4f}")

    # 保存生成结果
    with torch.no_grad():
        fixed_noise = torch.randn(16, noise_dim, 1, 1, device=device)
        fixed_labels = torch.randint(0, num_classes, (16,), device=device)
        fake = netG(fixed_noise, fixed_labels)
    vutils.save_image(fake, f"output_acgan/fake_samples_epoch_{epoch + 1}.png", normalize=True)
