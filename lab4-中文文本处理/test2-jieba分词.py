import jieba
import jieba.posseg as pseg
import jieba.analyse
import time

# 记录程序开始时间
start_time = time.time()

# 获取文本数据
file_path = "数据集-三国演义-清洗结果.txt"
with open(file_path, "r", encoding="utf-8") as file:
    text = file.read()

'''1-分词：将文本按照词语进行切分，得到一个词语列表'''
#分词
words = jieba.lcut(text)
# 将分词保存到文件中
output_file = "分词表.txt"
with open(output_file, "w", encoding="utf-8") as file:
    file.write(" ".join(words))

'''2-去除停用词：根据特定的停用词列表'''
words = jieba.lcut(text)
# 加载停用词列表
stopwords_path = "停用词列表.txt"
with open(stopwords_path, "r", encoding="utf-8") as file:
    stopwords = [line.strip() for line in file]
# 过滤停用词
filtered_words = [word for word in words if word not in stopwords and word != '\n' and word != ' ']
# 将去除停用次后的分词保存到文件中
output_file = "分词表-去除停用词.txt"
with open(output_file, "w", encoding="utf-8") as file:
    file.write(" ".join(filtered_words))

'''3-根据词语的频率进行关键词提取'''
keywords = jieba.analyse.extract_tags(text, topK=30)
print(keywords)


# 记录程序结束时间
end_time = time.time()

# 计算程序运行时间
run_time = end_time - start_time
print("程序运行时间：", run_time, "秒")