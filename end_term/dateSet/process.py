import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
import pandas as pd
import jieba

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

# 读取数据集
df = pd.read_excel('D:/a_zzw/a_code/github/大数据/code/end_term/dateSet/train_data_1_3500.xls')
# df = pd.read_excel('D:/a_zzw/a_code/github/大数据/code/end_term/dateSet/train_data_1_shuffled.xls')
# 划分训练集和验证集
train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)
# 加载停用词列表
def load_stopwords(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        stopwords = [line.strip() for line in file]
    return set(stopwords)
stopwords_path = 'D:/a_zzw/a_code/github/大数据/code/end_term/dateSet/cn_stopwords.txt'
stopwords_set = load_stopwords(stopwords_path)

# 分词
def tokenize_text(text):
    if isinstance(text, str):
        # 使用jieba分词
        words = list(jieba.cut(text))
        # 过滤停用词
        filtered_words = [word for word in words if word not in stopwords_set]
        return filtered_words
    else:
        return []

# 定义数据集类
# 构建词汇表，将每个词映射为唯一的整数标识符，使得模型可以使用整数来表示文本数据
vocab = set()
for text in df['text']:
    tokens = tokenize_text(text)
    vocab.update(tokens)
word_to_idx = {word: idx + 1 for idx, word in enumerate(vocab)}
word_to_idx['<PAD>'] = 0

# 定义数据预处理函数
def collate_fn(batch):
    texts = [torch.tensor([word_to_idx.get(word, 0) for word in sample['text']], dtype=torch.long) for sample in batch]
    labels = torch.tensor([sample['label'] for sample in batch], dtype=torch.long)
    return {
        'texts': torch.nn.utils.rnn.pad_sequence(texts, batch_first=True),
        'labels': labels
    }

