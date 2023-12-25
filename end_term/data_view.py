import pandas as pd
import matplotlib.pyplot as plt

# 加载数据集
file_path = 'D:/a_zzw/a_code/github/大数据/code/end_term/dateSet/train_data_1_3500.xls'
df = pd.read_excel(file_path)
# 将 'text' 列中的浮点数值转换为字符串
df['text'] = df['text'].astype(str)
# 绘制散点图
plt.figure(figsize=(10, 6))
scatter = plt.scatter(df['rid'], df['text'].apply(len), c=df['label'], cmap='viridis', alpha=0.7)
plt.title('Scatter Plot of Text Lengths by Label')
plt.xlabel('rid')
plt.ylabel('Text Length')
plt.colorbar(scatter, label='Label')

plt.show()
