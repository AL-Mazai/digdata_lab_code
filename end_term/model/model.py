import torch
import torch.nn as nn


'''
嵌入层 (self.embedding)：将整数编码的标记转换为密集向量表示。
卷积层 (self.conv_layers)：通过多个卷积层捕捉输入文本中的不同n-gram特征。每个卷积层使用不同大小的卷积核（由filter_sizes指定），以便捕捉不同大小的n-gram特征。
全连接层 (self.fc)：将卷积层的输出进行分类，输出最终的分类结果。
Dropout (self.dropout)：在卷积层的输出上应用dropout，以防止过拟合。
Forward 方法 (forward)：描述了模型的前向传播过程，其中包括嵌入、卷积、ReLU激活、最大池化、dropout等操作。
'''
class Model(nn.Module):
    def __init__(self, vocab_size, embedding_dim, num_filters, filter_sizes, output_dim, dropout):
        # 初始化模型，继承自nn.Module
        super(Model, self).__init__()
        # 定义一个嵌入层，将整数编码的标记转换为密集向量
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        # 创建一个包含指定滤波器大小的1维卷积层的列表
        self.conv_layers = nn.ModuleList([
            nn.Conv1d(embedding_dim, num_filters, fs) for fs in filter_sizes
        ])
        # 用于分类的全连接层
        self.fc = nn.Linear(len(filter_sizes) * num_filters, output_dim)
        # 对卷积层输出应用dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # 将输入通过嵌入层传递以获得密集表示
        x = self.embedding(x)
        # 调整维度以适应Conv1d期望的输入形状
        x = x.permute(0, 2, 1)
        # 对每个卷积层的输出应用ReLU激活
        conved = [nn.functional.relu(conv(x)) for conv in self.conv_layers]
        # 对每个卷积层的输出执行1D最大池化
        pooled = [nn.functional.max_pool1d(conv, conv.shape[2]) for conv in conved]
        # 对最大池化向量的连接输出应用dropout
        cat = self.dropout(torch.cat(pooled, dim=1))
        # 压缩最后一个维度并通过全连接层
        return self.fc(cat.squeeze(2))
