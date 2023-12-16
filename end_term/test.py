import pandas as pd


# 读取Excel文件
file_path = 'D:/a_zzw/a_code/github/大数据/code/end_term/dateSet/data.xls'  # 替换为你的文件路径
df = pd.read_excel(file_path)

# 随机排列数据
df_shuffled = df.sample(frac=1).reset_index(drop=True)

# 保存随机排列后的数据到新的Excel文件
output_path = 'D:/a_zzw/a_code/github/大数据/code/end_term/dateSet/sdata.xls'  # 替换为你的输出文件路径
df_shuffled.to_excel(output_path, index=False, engine='openpyxl')

print("数据已随机排列并保存到新的Excel文件。")
