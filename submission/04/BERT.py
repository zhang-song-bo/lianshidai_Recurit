import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import OneCycleLR  # type: ignore
from transformers import BertTokenizer, BertForSequenceClassification
from torchtext.datasets import IMDB
import matplotlib.pyplot as plt

# 设备设置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 加载 BERT 分词器
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")


def text_pipeline(text):
    """
    :param text: 原始文本
    :return: 处理后的 token id 和 attention mask
    """
    return tokenizer(text, padding="max_length", truncation=True, max_length=100, return_tensors="pt")


def label_pipeline(label):
    """
    :param label: 电影评论标签（pos/neg）
    :return: 1 表示正面，0 表示负面
    """
    return 1 if label == "pos" else 0


class IMDBDataset(Dataset):
    """自定义 IMDB 数据集类"""
    def __init__(self, data_iter):
        """
        :param data_iter: IMDB 数据集迭代器
        """
        self.data = []
        for label, text in data_iter:
            encoding = text_pipeline(text)
            self.data.append((
                encoding["input_ids"].squeeze(0),
                encoding["attention_mask"].squeeze(0),
                label_pipeline(label)
            ))

    def __len__(self):
        """返回数据集大小"""
        return len(self.data)

    def __getitem__(self, idx):
        """获取数据集中的单个样本"""
        return self.data[idx]


# 加载 IMDB 数据集
train_iter, test_iter = IMDB(split="train"), IMDB(split="test")
train_data, test_data = IMDBDataset(train_iter), IMDBDataset(test_iter)


def collate_fn(batch):
    """
    处理批量数据
    :param batch: 输入数据
    :return: 处理后的 input_ids, attention_masks, labels
    """
    input_ids, attention_masks, labels = zip(*batch)
    return torch.stack(input_ids), torch.stack(attention_masks), torch.tensor(labels, dtype=torch.long)


# 创建数据加载器
train_loader = DataLoader(train_data, batch_size=64, shuffle=True, collate_fn=collate_fn)
test_loader = DataLoader(test_data, batch_size=64, shuffle=False, collate_fn=collate_fn)

# 加载预训练 BERT 模型
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=1).to(device)

# 只训练 BERT 的最后 4 层
for param in model.bert.encoder.layer[:-4].parameters():
    param.requires_grad = False

# 定义优化器、损失函数和学习率调度器
optimizer = optim.AdamW(model.parameters(), lr=5e-5, weight_decay=1e-2)
criterion = nn.BCEWithLogitsLoss().to(device)

epochs = 3
scheduler = OneCycleLR(
    optimizer,
    max_lr=5e-5,
    total_steps=epochs * len(train_loader),
    pct_start=0.3,
    anneal_strategy='cos',
    div_factor=10,
    final_div_factor=100
)

# 训练模型
def train(model, loader, optimizer, criterion, scheduler):
    model.train()
    epoch_loss = 0
    for input_ids, attention_masks, labels in loader:
        input_ids, attention_masks, labels = input_ids.to(device), attention_masks.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask=attention_masks)
        logits = outputs.logits.squeeze(1)

        loss = criterion(logits, labels.float())
        loss.backward()
        optimizer.step()
        scheduler.step()

        epoch_loss += loss.item()
    return epoch_loss / len(loader)

# 评估模型
def evaluate(model, loader, criterion):
    model.eval()
    epoch_loss, correct, total = 0, 0, 0
    with torch.no_grad():
        for input_ids, attention_masks, labels in loader:
            input_ids, attention_masks, labels = input_ids.to(device), attention_masks.to(device), labels.to(device)

            outputs = model(input_ids, attention_mask=attention_masks)
            logits = outputs.logits.squeeze(1)

            loss = criterion(logits, labels.float())
            epoch_loss += loss.item()

            preds = torch.sigmoid(logits) > 0.5
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return epoch_loss / len(loader), correct / total


# 训练过程
train_losses, test_losses, test_accs = [], [], []
for epoch in range(epochs):
    train_loss = train(model, train_loader, optimizer, criterion, scheduler)
    test_loss, test_acc = evaluate(model, test_loader, criterion)
    train_losses.append(train_loss)
    test_losses.append(test_loss)
    test_accs.append(test_acc)
    print(f'Epoch: {epoch + 1:02}, Train Loss: {train_loss:.3f}, Test Loss: {test_loss:.3f}, Test Acc: {test_acc:.2%}')

# 保存模型
torch.save(model.state_dict(), "bert_imdb.pth")

# 绘制训练曲线
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(train_losses, label="Train Loss")
plt.plot(test_losses, label="Test Loss")
plt.legend()
plt.title("Loss Curve")
plt.subplot(1, 2, 2)
plt.plot(test_accs, label="Test Accuracy")
plt.legend()
plt.title("Accuracy Curve")
plt.tight_layout()
plt.show()