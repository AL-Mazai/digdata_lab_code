import torch
from end_term.dateSet.process import tokenize_text, word_to_idx
from end_term.model.model import Model

# if __name__ == "__main__":
#     # 加载已训练的模型
#     loaded_model = Model(vocab_size=len(word_to_idx), embedding_dim=100, num_filters=100, filter_sizes=[2, 3], output_dim=2, dropout=0.5)
#     loaded_model.load_state_dict(torch.load('D:/a_zzw/a_code/github/大数据/code/end_term/model/model.pth'))
#     loaded_model.eval()
#
#     # 文本预处理函数
#     def preprocess_text(text, word_to_idx, max_len):
#         tokens = tokenize_text(text)
#         indexed_tokens = [word_to_idx.get(word, 0) for word in tokens]
#         padded_tokens = indexed_tokens[:max_len] + [0] * max(0, max_len - len(indexed_tokens))
#         return torch.tensor(padded_tokens).unsqueeze(0)  # 添加批次维度
#
#     # 输入文本进行测试
#     while True:
#         text_to_predict = input("请输入要进行预测的文本（输入exit结束测试）: ")
#         if text_to_predict.lower() == 'exit':
#             break
#
#         preprocessed_input = preprocess_text(text_to_predict, word_to_idx, max_len=50)
#         with torch.no_grad():
#             output = loaded_model(preprocessed_input)
#             predicted_class = torch.argmax(output, dim=1).item()
#             print(f'预测类别: {predicted_class}')

import pandas as pd
import seaborn as sns
import torch
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from torch.utils.data import DataLoader

from end_term.dateSet.process import DepressionDataset, collate_fn
from end_term.train import model

if __name__ == "__main__":

    max_len = 50  # 根据实际情况调整
    # 加载测试数据集
    test_df = pd.read_excel('D:/a_zzw/a_code/github/大数据/code/end_term/dateSet/test_data_2.xls')  # 请替换为你的测试集路径

    # 初始化测试数据集和数据加载器
    test_dataset = DepressionDataset(test_df, max_len)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)

    # 设置模型为评估模式
    model.eval()

    # 初始化用于存储所有预测值和标签的列表
    all_preds = []
    all_labels = []

    # 使用torch.no_grad()上下文管理器，禁用梯度计算
    with torch.no_grad():
        # 遍历测试数据加载器中的每个批次
        for batch in test_loader:
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

    print(f'Test Accuracy: {accuracy:.4f}')
    print('Test Classification Report:')
    print(classification_rep)

    # 数据可视化
    # 这里可以按照你的需求进行可视化，例如混淆矩阵等
    # 计算混淆矩阵
    conf_mat = confusion_matrix(all_labels, all_preds)

    # 可视化混淆矩阵
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_mat, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Class 0', 'Class 1'],
                yticklabels=['Class 0', 'Class 1'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()