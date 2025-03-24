import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision import datasets
import matplotlib.pyplot as plt
import os

class CNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),# (32*32)->(32,32),感受野:3*3
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),# (32*32)->(32,32),感受野:5*5
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),# (32*32)->(16,16),感受野:6*6

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),# (16,16)->(16,16),感受野:10*10
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),# (16,16)->(16,16),感受野:14*14
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),# (16,16)->(8,8),感受野:16*16

            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),# (8,8)->(8,8),感受野:24*24
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=2),# (8,8)->(8,8),感受野:32*32
            nn.BatchNorm2d(256),
            nn.ReLU()
        )
        self.aap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(256, 10)

    def forward(self, x):
        x = self.conv(x) # (Batch_size, 256, 8, 8)
        x = self.aap(x)  # (Batch_size, 256, 1, 1)
        x = x.view(x.size(0), -1) # (Batch_size, 256)
        x = self.fc(x) # (Batch_size, 10)
        return x

lr = 0.005
epochs = 50
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
train_accuracies = []
test_accuracies = []

# 对于训练集，需要应用数据增强，让模型更好的学习到图像的特征
train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),# 50% 概率水平翻转
    transforms.RandomCrop(32, padding=4),  # 随机裁剪
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # 颜色抖动
    transforms.RandomRotation(15),  # 随机顺时针或逆时针旋转 15 度之内
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
# 对于测试集，不应用数据增强，测试模型是否能识别复杂情景下的图像
test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# 加载数据
train_dataset = datasets.CIFAR10(root='data', train=True, download=True, transform=train_transform)
test_dataset = datasets.CIFAR10(root='data', train=False, download=True, transform=test_transform)
train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)

# 模型、优化器、损失函数和学习率调度器
model = CNN().to(device)
optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
criterion = nn.CrossEntropyLoss()
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer,
    max_lr=lr,
    steps_per_epoch=len(train_loader),
    epochs=epochs,
)

# 执行训练
def train(model, train_loader, optimizer, criterion, device, scheduler):
    model.train()
    total_loss = 0.0
    for i, (inputs, labels) in enumerate(train_loader):
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()

    average_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch + 1}/{epochs}, Loss: {average_loss:.4f}")

# 执行评估
def evaluate(model, data_loader, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, dim=1) # torch.max返回元组(values, indices),这里只需要索引信息(即对应的标签类别)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    acc = 100 * correct / total
    return acc


# 开始训练
for epoch in range(epochs):
    train(model, train_loader, optimizer, criterion, device, scheduler)

    # 计算训练集和测试集的准确率
    train_acc = evaluate(model, train_loader, device)
    test_acc = evaluate(model, test_loader, device)

    train_accuracies.append(train_acc)
    test_accuracies.append(test_acc)

    print(f"Epoch {epoch + 1}/{epochs}: Train Acc: {train_acc:.2f}%, Test Acc: {test_acc:.2f}%")

    # 保存模型
    os.makedirs("checkpoint", exist_ok=True)
    torch.save(model.state_dict(), f'checkpoint/checkpoint_epoch_{epoch + 1}.pth')


# 绘制准确率图像
plt.figure(figsize=(10, 6))
plt.plot(range(1, epochs+1), train_accuracies, label='Train Accuracy', marker='o')
plt.plot(range(1, epochs+1), test_accuracies,    label='Test Accuracy', marker='o')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training and Test Accuracy Curves')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.xticks(range(1, epochs+1, 5))
plt.show()