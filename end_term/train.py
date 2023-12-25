import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns

from end_term.dateSet.process import word_to_idx, DepressionDataset, train_df, val_df, collate_fn
from end_term.model.model import Model


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
filter_sizes = [2, 3]
output_dim = 2  # 二分类任务，输出维度为2
dropout = 0.5
model = Model(vocab_size, embedding_dim, num_filters, filter_sizes, output_dim, dropout)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


# 训练模型
num_epochs = 10
train_losses = []
val_losses = []

for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0
    # 迭代训练数据加载器中的每个批次
    for batch in train_loader:
        # 提取批次中的输入文本和对应标签
        texts = batch['texts']
        labels = batch['labels']
        # 梯度清零，以防累积先前迭代的梯度
        optimizer.zero_grad()
        # 前向传播：计算模型对输入文本的预测
        outputs = model(texts)
        # 计算模型预测与实际标签之间的损失
        loss = criterion(outputs, labels)
        # 反向传播：计算损失对模型参数的梯度
        loss.backward()
        # 更新模型参数，使用指定的优化算法
        optimizer.step()
        # 累积当前批次的训练损失
        train_loss += loss.item()

    # 计算所有批次的平均训练损失
    train_loss /= len(train_loader)
    # 将平均训练损失添加到训练损失列表，用于监控训练过程
    train_losses.append(train_loss)

    # 验证模型
    # 设置模型为评估模式，不进行梯度计算和参数更新
    model.eval()

    # 初始化验证损失为零
    val_loss = 0.0

    # 使用torch.no_grad()上下文管理器，禁用梯度计算
    with torch.no_grad():
        # 遍历验证数据加载器中的每个批次
        for batch in val_loader:
            # 提取当前批次中的输入文本和相应标签
            texts = batch['texts']
            labels = batch['labels']
            # 前向传播：计算模型对输入文本的预测
            outputs = model(texts)
            # 计算模型预测与实际标签之间的损失
            loss = criterion(outputs, labels)
            # 累积验证损失
            val_loss += loss.item()

    # 计算所有验证批次的平均验证损失
    val_loss /= len(val_loader)
    # 将平均验证损失添加到验证损失列表，用于监控模型在验证集上的性能
    val_losses.append(val_loss)

    print(f'Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')


# 模型评估
# 设置模型为评估模式，不进行梯度计算和参数更新
model.eval()
# 初始化用于存储所有预测值和标签的列表
all_preds = []
all_labels = []
# 使用torch.no_grad()上下文管理器，禁用梯度计算
with torch.no_grad():
    # 遍历验证数据加载器中的每个批次
    for batch in val_loader:
        # 提取当前批次中的输入文本和相应标签
        texts = batch['texts']
        labels = batch['labels']
        # 前向传播：计算模型对输入文本的预测
        outputs = model(texts)
        # 使用argmax获取每个样本的预测类别
        preds = torch.argmax(outputs, dim=1)

        # 将当前批次的预测值和标签添加到总体列表中
        all_preds.extend(preds.tolist())
        all_labels.extend(labels.tolist())

# 计算准确率和分类报告
accuracy = accuracy_score(all_labels, all_preds)
classification_rep = classification_report(all_labels, all_preds)
print(f'textCNN Accuracy: {accuracy:.4f}')
print('textCNN Classification Report:')
print(classification_rep)

# 数据可视化
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

# 画测试集和验证集结果混淆矩阵
conf_matrix_val = confusion_matrix(all_labels, all_preds)

# 绘制混淆矩阵热图
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix_val, annot=True, fmt='d', cmap='Blues',
            xticklabels=['0', '1'], yticklabels=['0', '1'])
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Val Confusion Matrix - textCNN')
plt.show()


# 保存训练好的模型
torch.save(model.state_dict(), 'D:/a_zzw/a_code/github/大数据/code/end_term/model/model.pth')

