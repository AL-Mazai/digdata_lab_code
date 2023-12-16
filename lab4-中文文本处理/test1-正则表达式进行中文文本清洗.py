import re

# 读取文件内容
file_path = "数据集-三国演义.txt"
with open(file_path, "r", encoding="utf-8") as file:
    text = file.read()

# 去除重复的标点符号
cleaned_text = re.sub(r'([，。！？；：“”‘’])\1+', r'\1', text)

# 去掉空段
cleaned_text = re.sub(r'\n+', '\n', cleaned_text)

# 去除段首多余空格，保留首行缩进两个空格
cleaned_text = re.sub(r'^(?!\n)( +)', r'    ', cleaned_text, flags=re.MULTILINE)

# 重构文本段落
cleaned_text = '\n\n'.join(cleaned_text.split('\n\n'))

# 保存清洗后的文本到结果文件
result_file_path = "数据集-三国演义-清洗结果.txt"
with open(result_file_path, "w", encoding="utf-8") as file:
    file.write(cleaned_text)

print("清洗结果已保存到文件：", result_file_path)