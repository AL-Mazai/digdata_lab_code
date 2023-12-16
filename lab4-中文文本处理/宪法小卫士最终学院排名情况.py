import pandas as pd

# 读取Excel文件
data = pd.read_excel("宪法小卫士11.14统计结果.xlsx")

# 最大最小归一化方法处理数据
data['完成率归一化'] = (data['完成率'] - data['完成率'].min()) / (data['完成率'].max() - data['完成率'].min())
data['平均分归一化'] = (data['平均分'] - data['平均分'].min()) / (data['平均分'].max() - data['平均分'].min())
data['学院人数归一化'] = (data['注册数'] - data['注册数'].min()) / (data['注册数'].max() - data['注册数'].min())

# 计算综合得分
data['综合得分'] = data['完成率归一化'] * 0.5 + data['平均分归一化'] * 0.25 + data['学院人数归一化'] * 0.25

# 计算综合排名
data['综合排名'] = data['综合得分'].rank(ascending=False, method='min')

# 根据综合排名进行排序
data = data.sort_values(by='综合排名', ascending=True)

# 总排名
rank_result = data.head(25)

# 输出结果
print(rank_result[['校区', '注册数', '参与数', '平均分', '完成率', '综合得分', '综合排名']])

# 保存结果
rank_result.to_excel("排名结果.xlsx", index=False)