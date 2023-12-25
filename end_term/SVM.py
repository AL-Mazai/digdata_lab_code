import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, roc_auc_score


#########################################训练###################################
# 加载数据集
df = pd.read_excel('D:/a_zzw/a_code/github/大数据/code/end_term/dateSet/train_data_1_3500.xls')
train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

# 加载停用词列表
def load_stopwords(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        stopwords = [line.strip() for line in file]
    return list(stopwords)
# 停用词列表
stopwords_path = 'D:/a_zzw/a_code/github/大数据/code/end_term/dateSet/cn_stopwords.txt'
stopwords = load_stopwords(stopwords_path)
# 将NaN值替换为空字符串
train_df['text'].fillna('', inplace=True)
val_df['text'].fillna('', inplace=True)
# 创建 TF-IDF 特征提取器，设置停用词列表
vectorizer = TfidfVectorizer(max_features=5000, stop_words=stopwords)
X_train = vectorizer.fit_transform(train_df['text'])
X_val = vectorizer.transform(val_df['text'])

# 获取标签
y_train = train_df['label']
y_val = val_df['label']
# 初始化 SVM 分类器
svm_classifier = SVC(kernel='rbf')
# 训练 SVM 分类器
svm_classifier.fit(X_train, y_train)
# 在验证集上进行预测
val_preds = svm_classifier.predict(X_val)

# 评估模型
accuracy = accuracy_score(y_val, val_preds)
classification_rep = classification_report(y_val, val_preds)
print(f'SVM准确率: {accuracy:.4f}')
print('SVM分类报告:')
print(classification_rep)


#########################################测试###################################
# 加载测试数据集
test_df = pd.read_excel('D:/a_zzw/a_code/github/大数据/code/end_term/dateSet/test_data_1_1430.xls')
# 将NaN值替换为空字符串
test_df['text'].fillna('', inplace=True)
test_df['text'] = test_df['text'].astype(str)
# 使用相同的向量化器转换测试数据
X_test = vectorizer.transform(test_df['text'])
y_test = test_df['label']

# 在测试集上进行预测
test_preds = svm_classifier.predict(X_test)
test_accuracy = accuracy_score(y_test, test_preds)
test_classification_rep = classification_report(y_test, test_preds)

print(f'SVM测试准确率: {test_accuracy:.4f}')
print('SVM测试分类报告:')
print(test_classification_rep)


###################################################可视化#########################################3
# 计算混淆矩阵
conf_matrix_val = confusion_matrix(y_val, val_preds)
conf_matrix_test = confusion_matrix(y_test, test_preds)

# 使用Seaborn绘制混淆矩阵热图
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix_test, annot=True, fmt='d', cmap='Blues', xticklabels=['0', '1'], yticklabels=['0', '1'])
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Test Confusion Matrix - SVM')
plt.show()

plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix_val, annot=True, fmt='d', cmap='Blues', xticklabels=['0', '1'], yticklabels=['0', '1'])
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Val Confusion Matrix - SVM')
plt.show()



# 在验证集上计算ROC曲线
val_probs = svm_classifier.decision_function(X_val)
fpr_val, tpr_val, _ = roc_curve(y_val, val_probs)
roc_auc_val = roc_auc_score(y_val, val_probs)

# 在测试集上计算ROC曲线
test_probs = svm_classifier.decision_function(X_test)
fpr_test, tpr_test, _ = roc_curve(y_test, test_probs)
roc_auc_test = roc_auc_score(y_test, test_probs)

# 绘制ROC曲线
plt.figure(figsize=(8, 6))
plt.plot(fpr_val, tpr_val, color='darkorange', lw=2, label=f'Val ROC curve (AUC = {roc_auc_val:.2f})')
plt.plot(fpr_test, tpr_test, color='navy', lw=2, label=f'Test ROC curve (AUC = {roc_auc_test:.2f})')
plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--', label='Random')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve - SVM')
plt.legend(loc='lower right')
plt.show()
