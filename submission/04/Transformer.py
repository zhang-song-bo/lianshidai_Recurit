import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import OneCycleLR  # type: ignore
from torchtext.datasets import IMDB
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
import numpy as np


# 定义文本分词器，使用 spaCy 进行英文分词
tokenizer = get_tokenizer("spacy", language="en_core_web_sm")

def collect_tokens(data_iter):
    """
    :param data_iter: IMDB数据集
    :return: 返回单词列表
    """
    all_tokens = []
    for _, text in data_iter:
        tokens = tokenizer(text)
        all_tokens.append(tokens)
    return all_tokens  # 返回所有 tokens 的列表

# 加载 IMDB 数据集
train_iter = IMDB(split="train")
test_iter = IMDB(split="test")

# 构建词汇表
vocab = build_vocab_from_iterator(collect_tokens(train_iter), max_tokens=20000, specials=["<pad>", "<unk>"])
vocab.set_default_index(vocab["<unk>"])  # 设置未记录的词用 <unk> 代替

def text_pipeline(text):
    """
    :param text: 欲处理的文本列表
    :return: 文本对应的索引列表
    """
    tokens = tokenizer(text)
    max_length = 100 # 只需要前 100 长度
    tokens = tokens[:max_length]
    index = []
    for token in tokens:
        index.append(vocab[token])
    return index

def label_pipeline(label):
    """
    :param label: 情感标签 pos 或 neg
    :return:  将 pos 或 neg 映射为 1 或 0
    """
    return 1 if label == "pos" else 0

class IMDBDataset(Dataset):
    """自定义 IMDB 数据集类，用于加载 IMDB 电影评论数据"""
    def __init__(self, data_iter):
        """
        初始化
        :param data_iter: 同时包含 （label，text） 的数据迭代器
        """
        self.data = []
        for label, text in data_iter:
            self.data.append((text_pipeline(text), label_pipeline(label)))

    def __len__(self):
        """
        :return: 返回数据集 self.data 长度
        """
        return len(self.data)

    def __getitem__(self, idx):
        """
        :param idx: 索引
        :return: 索引对应的单词
        """
        return self.data[idx]

# 创建训练集和测试集
train_data = IMDBDataset(train_iter)
test_data = IMDBDataset(test_iter)

def collate_fn(batch):
    """
    处理批量数据，对文本进行填充使其长度一致
    :param batch: 批量数据
    :return: 填充处理之后的结果
    """
    texts, labels = zip(*batch)

    lengths = []
    for text in texts:
        lengths.append(len(text))

    max_len = max(lengths)

    padded_texts = []
    for text in texts:
        padded_texts.append(text + [vocab["<pad>"]] * (max_len - len(text)))

    return torch.tensor(padded_texts), torch.tensor(labels, dtype=torch.float)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 创建数据加载器
train_loader = DataLoader(train_data, batch_size=64, shuffle=True, collate_fn=collate_fn)
test_loader = DataLoader(test_data, batch_size=64, shuffle=False, collate_fn=collate_fn)

class PositionalEncoding(nn.Module):
    """ 定义位置编码类，为 Transformer 提供位置信息 """
    def __init__(self, d_model, dropout):
        """
        :param d_model: 词向量的维度
        :param dropout: Dropout 的概率，用于防止过拟合。
        """
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        max_len = 5000 # 最大序列长度
        position_code = torch.zeros(max_len, d_model) # 储存位置编码

        position_item = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1) # 生成位置索引 形状是(max_len, 1)

        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))  # 即 10000 ^ (-2i / d_model)

        position_code[:, 0::2] = torch.sin(position_item * div_term) # 偶数位置
        position_code[:, 1::2] = torch.cos(position_item * div_term) # 奇数位置

        self.register_buffer('position_code', position_code) # 将位置编码矩阵注册为模型不参与训练的缓冲区

    def forward(self, x):
        """
        前向传播
        :param x: 输入张量 (batch_size, seq_len, d_model)
        :return:  添加位置编码之后的张量
        """
        x = x + self.position_code[:x.size(1)] # 只取前 seq_len 个位置编码
        return self.dropout(x)

class TransformerClassifier(nn.Module):
    """ 定义 Transformer 分类模型 """
    def __init__(self, vocab_size, embed_dim, num_heads, num_layers, hidden_dim, dropout):
        """
        初始化
        :param vocab_size: 词汇表的大小
        :param embed_dim: 词嵌入的维度
        :param num_heads: Multi-Head 的头数
        :param num_layers: Encoder 的层数
        :param hidden_dim: FNN的隐藏层维度
        :param dropout: dropout 率
        """
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embed_dim) # 进行词嵌入，完成离散的索引->可学习的向量
        self.pos_encoder = PositionalEncoding(embed_dim, dropout) # 完成位置信息的添加

        encoder_layers = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dim_feedforward=hidden_dim, dropout=dropout) # 编码器层，传入若干所需的参数

        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers) # 完成若干个编码器层堆叠
        self.fc = nn.Linear(embed_dim, 1) # 全连接层，完成分类
        self.dropout = nn.Dropout(dropout) # Dropout 层

    def forward(self, text):
        """
        前向传播，完成计算流程
        :param text: 输入的文本张量 形状为 (batch_size, seq_len)
        :return:分类结果，形状为 (batch_size)
        """
        embedded = self.embedding(text) # 完成词嵌入映射，得到 (batch_size, seq_len, embed_dim)形状
        embedded = self.dropout(embedded) # dropout

        embedded = self.pos_encoder(embedded) # 添加位置编码

        embedded = embedded.transpose(0, 1) # 调整张量的维度为 (seq_len, batch_size, embed_dim)

        output = self.transformer_encoder(embedded) # 通过 Transformer 编码器
        output = output.mean(dim=0) # 对序列维度（seq_len）取均值，得到句子级别的表示， 形状变为 (batch_size, embed_dim)

        return self.fc(output).squeeze(1) # 通过全连接层，得到分类结果，形状为 (batch_size, 1)，然后去除多余的维度，返回的形状为 (batch_size)

# 设置模型参数
embed_dim = 512
num_heads = 4
num_layers = 4
hidden_dim = 512
dropout = 0.2
epochs = 10

# 初始化模型
model = TransformerClassifier(
    vocab_size=len(vocab),
    embed_dim=embed_dim,
    num_heads=num_heads,
    num_layers=num_layers,
    hidden_dim=hidden_dim,
    dropout=dropout
).to(device)


# 定义优化器和损失函数
optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-2)
scheduler = OneCycleLR(
    optimizer,
    max_lr=1e-4,         # 训练中最高学习率
    total_steps=epochs * len(train_loader),  # 总训练步数 = epoch数 * 每个epoch的batch数
    pct_start=0.3,       # warmup 30% 训练步数
    anneal_strategy='cos',  # 余弦退火策略
    div_factor=10,       # 初始学习率 = max_lr / 10
    final_div_factor=100 # 结束学习率 = max_lr / 100
)

criterion = nn.BCEWithLogitsLoss().to(device)

# 训练模型
def train(model, loader, optimizer, criterion, scheduler):
    model.train()
    epoch_loss = 0
    for text, label in loader:
        text, label = text.to(device), label.to(device)
        predictions = model(text)
        loss = criterion(predictions, label)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        epoch_loss += loss.item()
    return epoch_loss / len(loader)

# 评估模型
def evaluate(model, loader, criterion):
    model.eval()
    epoch_loss = 0
    correct, total = 0, 0
    with torch.no_grad():
        for text, label in loader:
            text, label = text.to(device), label.to(device)
            predictions = model(text)

            loss = criterion(predictions, label)
            epoch_loss += loss.item()
            preds = torch.sigmoid(predictions) > 0.5
            correct += (preds == label).sum().item()
            total += label.size(0)
    return epoch_loss / len(loader), correct / total

# 训练过程
train_losses, test_losses, test_accs = [], [], []

# 开始训练
for epoch in range(epochs):
    train_loss = train(model, train_loader, optimizer, criterion, scheduler)
    test_loss, test_acc = evaluate(model, test_loader, criterion)
    train_losses.append(train_loss)
    test_losses.append(test_loss)
    test_accs.append(test_acc)
    print(f'Epoch: {epoch + 1:02}, Train Loss: {train_loss:.3f}, Test Loss: {test_loss:.3f}, Test Acc: {test_acc:.2%}')

# 保存训练后的参数信息
torch.save(model.state_dict(), "model.pth")
# 保存词汇表信息
torch.save(vocab, "vocab.pth")

# 绘制模型训练和评估的可视化图表
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Train Loss')
plt.plot(test_losses, label='Test Loss')
plt.legend()
plt.title('Loss Curve')
plt.subplot(1, 2, 2)
plt.plot(test_accs, label='Test Accuracy')
plt.legend()
plt.title('Accuracy Curve')
plt.tight_layout()
plt.show()