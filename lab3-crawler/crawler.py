# 导入所需包
import requests
from bs4 import BeautifulSoup
from lxml import etree
import csv

# 打开CSV文件，指定编码和换行符处理方式
fp = open('抑郁症.csv', 'a', newline='', encoding='utf_8_sig')
# 创建CSV写入器对象
writer = csv.writer(fp)
# 定义表头信息
header = ['问题', '提问时间', '回答', '回答时间', '医生信息', '医院信息']
# 将表头写入CSV文件
writer.writerow(header)

# 定义请求头信息
headers = {
    'user-agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10.10; rv:70.0) Gecko/20100101 Firefox/70.0'
}

# 网页内容格式调整
url = 'http://so.120ask.com/?kw=抑郁症&page='
# 此处爬取2-101页
for i in range(2, 40):
    # 构建完整的URL，例如 https://www.120ask.com/list/nfmk/2/
    r = url + str(i) + '&isloc=1'
    # print(r)
    # 发送GET请求获取网页内容
    html = requests.get(r, headers=headers)
    # 检查HTTP请求是否成功，如果不成功则抛出异常
    html.raise_for_status()
    # 设置网页的编码方式为自动推测的编码方式
    html.encoding = html.apparent_encoding
    # 使用BeautifulSoup解析网页内容
    soup = BeautifulSoup(html.text, 'html.parser')

    # 爬取所有一级链接，以进入详细页面
    for item in soup.find_all('h3'):
        # print(item)
        # 构建完整的问题链接，例如 https://www.120ask.com/question/12345678
        link = 'https:' + item.find('a')['href']
        # print(link)
        # 发送GET请求获取问题页面的HTML内容
        date_html = requests.get(link, headers=headers).text
        # 使用etree.HTML()方法解析HTML内容，生成XPath解析对象
        f = etree.HTML(date_html)

        '''表格字段信息'''
        #问题
        question = f.xpath('normalize-space(/html/body/div[1]/div[5]/div[2]/div[3]/div[1]/h1/text())')
        # print('问题：', question)

        # 提问时间
        que_time = f.xpath('normalize-space(/html/body/div[1]/div[5]/div[2]/div[3]/div[1]/div/span[2]/text())')
        # print('提问时间：', que_time)

        # 回答
        answer = f.xpath('normalize-space(/html/body/div[1]/div[5]/div[2]/div[7]'
                         '/div[1]/div[2]/div[2]/div[1]/div[1]/p/text())').strip()
        # print(answer)

        # 回答时间
        answer_time = f.xpath('normalize-space(/html/body/div[1]/div[5]/div[2]/div[7]'
                         '/div[1]/div[2]/div[2]/span/text())').strip()
        # print(answer_time)

        # 医院信息
        hospital_doctor = f.xpath('/html/body/div[1]/div[5]/div[2]/div[7]'
                                '/div[1]/div[1]/div/span[1]/text()')
        hospital_info = hospital_doctor[1].split(' ')[1].strip()
        # print(hospital_info)
        # print(hospital_doctor[1].split(' '))

        # 医生信息
        doctor_name = f.xpath('normalize-space(/html/body/div[1]/div[5]/div[2]'
                              '/div[7]/div[1]/div[1]/div/span[1]/a)')
        doctor_position = hospital_doctor[1].split(' ')[2]
        doctor_major = f.xpath('normalize-space(/html/body/div[1]/div[5]/div[2]'
                               '/div[7]/div[1]/div[1]/div/span[2]/text())')
        #医生信息：姓名，职位，擅长
        doctor_info = doctor_name + ', ' + doctor_position + ', ' + doctor_major
        print(doctor_info)

        # 将提取的信息写入CSV文件
        writer.writerow(
            (question, que_time, answer, answer_time,
             doctor_info, hospital_info)
        )

# 关闭文件
fp.close()
