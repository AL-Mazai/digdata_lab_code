# 导入所需包
import requests
from bs4 import BeautifulSoup
from lxml import etree
import time
import csv

fp = open('内分泌.csv', 'a', newline='', encoding='utf_8_sig')
writer = csv.writer(fp)


headers = {
    'user-agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10.10; rv:70.0) Gecko/20100101 Firefox/70.0'
}

# 网页内容格式调整
# 内分泌科室在线问诊网址
url = 'https://www.120ask.com/list/nfmk/'
# 此处爬取2-101页，一共200页
for i in range(2, 10):
    # 构建完整的URL，例如 https://www.120ask.com/list/nfmk/2/
    r = url + str(i) + '/'
    # 发送GET请求获取网页内容
    html = requests.get(r, headers=headers)
    # 检查HTTP请求是否成功，如果不成功则抛出异常
    html.raise_for_status()
    # 设置网页的编码方式为自动推测的编码方式
    html.encoding = html.apparent_encoding
    # 使用BeautifulSoup解析网页内容
    soup = BeautifulSoup(html.text, 'html.parser')

    # 爬取所有一级链接，以进入详细页面
    for item in soup.find_all('p', 'h-pp1'):
        print(item, '\n')
        # 构建完整的问题链接，例如 https://www.120ask.com/question/12345678
        link = 'https:' + item.find('a', 'q-quename')['href']
        print(link)
        # 发送GET请求获取问题页面的HTML内容
        date_html = requests.get(link, headers=headers).text
        # 使用etree.HTML()方法解析HTML内容，生成XPath解析对象
        f = etree.HTML(date_html)

        # 提问者性别与年龄
        ques_gender_age = f.xpath('/html/body/div[1]/div[5]/div[2]/div[3]/div[1]/div/span[1]/text()')[0]

        '''问题描述'''
        # 使用XPath表达式提取问题描述信息，返回一个包含多个字符串的列表
        ques_des = f.xpath('/html/body/div[1]/div[5]/div[2]/div[3]/div[2]/p[1]/text()')
        # 对问题描述进行处理，去除空格并去除列表中的空字符串
        ques_des = [''.join(x.split()) for x in ques_des]
        while ques_des.count(''):
            ques_des.remove('')
        # 将问题描述信息保存在ques_des变量中
        ques_des = ques_des[0]
        # 打印问题描述信息
        # print('问题', ques_des)

        # 回复时间
        ans_time = f.xpath('normalize-space(/html/body/div[1]/div[5]/div[2]/div[7]/div[1]/div[2]/div[2]/span/text())')
        print('时间', ans_time)

        # 回复者职称
        anser_position = f.xpath('/html/body/div[1]/div[5]/div[2]/div[7]/div[1]/div[1]/div/span[1]/text()')
        anser_position = [''.join(x.split()) for x in anser_position]
        while anser_position.count(''):
            anser_position.remove('')
        # anser_position = anser_position[0]
        # print(anser_position)

        # 回复者擅长领域
        anser_good_at = f.xpath('/html/body/div[1]/div[5]/div[2]/div[7]/div[1]/div[1]/div/span[2]/text()')
        # print(anser_good_at)

        # 回复内容
        anser_content = f.xpath('/html/body/div[1]/div[5]/div[2]/div[7]/div[1]/div[2]/div[2]/div[1]/div[1]/p/text()')
        anser_content = [''.join(x.split()) for x in anser_content]
        # print(anser_content)

        # 提问时间
        release_time = f.xpath('//*[@id="body_main"]/div[5]/div[2]/div[3]/div[1]/div/span[2]/text()')[0]
        # print(release_time)

        # 回复者诊疗经验
        anser_help_amout = f.xpath('//*[@id="body_main"]/div[5]/div[2]/div[7]/div[1]/div[1]/div/span[3]/text()')
        anser_help_amout = [''.join(x.split()) for x in anser_help_amout]
        while anser_help_amout.count(''):
            anser_help_amout.remove('')
        # print(anser_help_amout)

        # 将提取的信息写入CSV文件
        writer.writerow(
            (release_time, ques_gender_age, ques_des, ans_time,
             anser_position, anser_good_at, anser_help_amout, anser_content)
        )

# 关闭文件
fp.close()

# 遍历页码范围，例如从第2页到第9页
for i in range(2, 10):
    # 构建完整的URL，例如 https://www.120ask.com/list/nfmk/2/
    r = url + str(i) + '/'
    # 发送GET请求获取网页内容
    html = requests.get(r, headers=headers)
    # 检查HTTP请求是否成功，如果不成功则抛出异常
    html.raise_for_status()
    # 设置网页的编码方式为自动推测的编码方式
    html.encoding = html.apparent_encoding
    # 使用BeautifulSoup解析网页内容
    soup = BeautifulSoup(html.text, 'html.parser')

    # 遍历每个问题项
    for item in soup.find_all('p', 'h-pp1'):
        # 提取问题标题
        ques_title = item.find('a', 'q-quename')['title']
        # 提取所在科室
        department = item.find('a').get_text()
        # 将提取的信息写入CSV文件
        writer.writerow((department, ques_title))
# 关闭文件
fp.close()

# 遍历页码范围，例如从第2页到第9页
for i in range(2, 10):
    # 构建完整的URL，例如 https://www.120ask.com/list/nfmk/2/
    r = url + str(i) + '/'
    # 发送GET请求获取网页内容
    html = requests.get(r, headers=headers)
    # 检查HTTP请求是否成功，如果不成功则抛出异常
    html.raise_for_status()
    # 设置网页的编码方式为自动推测的编码方式
    html.encoding = html.apparent_encoding
    # 使用BeautifulSoup解析网页内容
    soup = BeautifulSoup(html.text, 'html.parser')
    # 遍历每个问题项
    for item in soup.find_all('div', 'fr h-right-p'):
        # 提取回复数
        ans_amout1 = item.find_next()
        ans_amout = ans_amout1.get_text()
        # 提取回复状态
        ans_status = ans_amout1.find_next().get_text()
        # 将提取的信息写入CSV文件
        writer.writerow((ans_amout, ans_status))
# 关闭文件
fp.close()