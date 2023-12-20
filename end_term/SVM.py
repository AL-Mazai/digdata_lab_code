import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

# 加载数据集
df = pd.read_excel('D:/a_zzw/a_code/github/大数据/code/end_term/dateSet/train_data_1_shuffled.xls')
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
# 将文本数据转换为 TF-IDF 特征
X_train = vectorizer.fit_transform(train_df['text'])
X_val = vectorizer.transform(val_df['text'])
# 获取标签
y_train = train_df['label']
y_val = val_df['label']

# 初始化 SVM 分类器
svm_classifier = SVC(kernel='rbf')
# svm_classifier = SVC(kernel='linear')
# 训练 SVM 分类器
svm_classifier.fit(X_train, y_train)

# 在验证集上进行预测
val_preds = svm_classifier.predict(X_val)

# 评估模型
accuracy = accuracy_score(y_val, val_preds)
classification_rep = classification_report(y_val, val_preds)

print(f'准确率: {accuracy:.4f}')
print('分类报告:')
print(classification_rep)


#########################################测试###################################3
# 加载测试数据集
test_df = pd.read_excel('D:/a_zzw/a_code/github/大数据/code/end_term/dateSet/test_data_1.xls')

# 将NaN值替换为空字符串
test_df['text'].fillna('', inplace=True)

# 使用相同的向量化器转换测试数据
X_test = vectorizer.transform(test_df['text'])
y_test = test_df['label']

# 在测试集上进行预测
test_preds = svm_classifier.predict(X_test)

# 在测试集上评估模型
test_accuracy = accuracy_score(y_test, test_preds)
test_classification_rep = classification_report(y_test, test_preds)

print(f'测试准确率: {test_accuracy:.4f}')
print('测试分类报告:')
print(test_classification_rep)

