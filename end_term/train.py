import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd
import matplotlib.pyplot as plt
import jieba
from end_term.model.model import Model

# 读取数据集
# df = pd.read_excel('D:/a_zzw/a_code/github/大数据/code/end_term/dateSet/data.xls')
df = pd.read_excel('D:/a_zzw/a_code/github/大数据/code/end_term/dateSet/shuffled_data.xls')
# 划分训练集和验证集
train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

# 分词函数
def tokenize_text(text):
    if isinstance(text, str):
        return list(jieba.cut(text))
    else:
        return []

# 定义数据集类
class DepressionDataset(Dataset):
    def __init__(self, dataframe, max_len):
        self.data = dataframe
        self.max_len = max_len
    def __len__(self):
        return len(self.data)
    def __getitem__(self, index):
        text = str(self.data.iloc[index]['text'])
        label = int(self.data.iloc[index]['label'])
        tokenized_text = tokenize_text(text)[:self.max_len]
        return {
            'text': tokenized_text,
            'label': torch.tensor(label, dtype=torch.long)
        }

# 构建词汇表
vocab = set()
for text in df['text']:
    tokens = tokenize_text(text)
    vocab.update(tokens)

word_to_idx = {word: idx + 1 for idx, word in enumerate(vocab)}
word_to_idx['<PAD>'] = 0


# 定义数据预处理函数
def collate_fn(batch):
    texts = [torch.tensor([word_to_idx.get(word, 0) for word in sample['text']]) for sample in batch]
    labels = torch.tensor([sample['label'] for sample in batch], dtype=torch.long)
    return {
        'texts': torch.nn.utils.rnn.pad_sequence(texts, batch_first=True),
        'labels': labels
    }

# 初始化数据集和数据加载器
max_len = 50  # 根据实际情况调整
train_dataset = DepressionDataset(train_df, max_len)
val_dataset = DepressionDataset(val_df, max_len)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)

# 初始化模型、损失函数和优化器
vocab_size = len(word_to_idx)
embedding_dim = 100
num_filters = 100
filter_sizes = [2, 3, 4]
output_dim = 2  # 二分类任务，输出维度为2
dropout = 0.5
model = Model(vocab_size, embedding_dim, num_filters, filter_sizes, output_dim, dropout)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


# 训练模型
num_epochs = 20
train_losses = []
val_losses = []

for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0
    for batch in train_loader:
        texts = batch['texts']
        labels = batch['labels']

        optimizer.zero_grad()

        outputs = model(texts)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    train_loss /= len(train_loader)
    train_losses.append(train_loss)

    # 验证模型
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for batch in val_loader:
            texts = batch['texts']
            labels = batch['labels']

            outputs = model(texts)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

    val_loss /= len(val_loader)
    val_losses.append(val_loss)

    print(f'Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')


# 模型评估
model.eval()
all_preds = []
all_labels = []

with torch.no_grad():
    for batch in val_loader:
        texts = batch['texts']
        labels = batch['labels']

        outputs = model(texts)
        preds = torch.argmax(outputs, dim=1)

        all_preds.extend(preds.tolist())
        all_labels.extend(labels.tolist())

# 计算准确率和分类报告
accuracy = accuracy_score(all_labels, all_preds)
classification_rep = classification_report(all_labels, all_preds)

print(f'Accuracy: {accuracy:.4f}')
print('Classification Report:')
print(classification_rep)

# 数据可视化
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

# 保存训练好的模型
torch.save(model.state_dict(), 'D:/a_zzw/a_code/github/大数据/code/end_term/model/model.pth')

