#-*- codeing = utf-8 -*-
import string
from urllib.parse import urlparse
import re
import os
import sys
import pandas as pd
import numpy as np

#URL长度
def get_length(url):
    length = len(url)
    return length

#不安全字符的数量
#不安全字符：'<','>','"','#','%','{','}','|','\\','^','~','[',']','`',' '
unsafeCharac=['<','>','"','#','%','{','}','|','\\','^','~','[',']','`',' ']
def get_unsafe_count(url):
    sum=0
    for i in range(len(unsafeCharac)):
        sum+=url.count(unsafeCharac[i],0,len(url))
    return sum

#str3.count("is",9,len(str3))
#保留字符的数量
#保留字符：';','/','?',':','@','=','&'
reversedCharac=[';', '/', '?', ':', '@', '=', '&']
def get_reversed_count(url):
    sum=0
    for i in range(len(reversedCharac)):
        sum+=url.count(reversedCharac[i], 0, len(url))
    return sum

#其他字符的数量
#其他字符：'$','-','_','.','+','!','*','/'','(',')',','
othersCharac=['$','-','_','.','+','!','*','\'','(',')',',']
def get_others_count(url):
    sum=0
    for i in range(len(othersCharac)):
        sum+=url.count(othersCharac[i],0,len(url))
    return sum

#数字的数量
def get_digits_count(url):
    count = 0
    for i in range(len(url)):
        if url[i] in string.digits:
            count += 1
    return count

#数字占总长度的比例
def get_digits_percent(url):
    return get_digits_count(url) / (len(url) * 1.0)

#大写字母的数量
def get_upcase_count(url):
    count = 0
    for i in range(len(url)):
        if url[i] in string.ascii_uppercase:
            count += 1
    return count

#大写字母占总长度的比例
def get_upcase_percent(url):
    return get_upcase_count(url) / (len(url) * 1.0)

#连续数字的最大长度
def get_digits_max_length(url):
    # 字符串转列表进行遍历
    str = list(url)
    count = 0
    length = 0
    for i in range(len(str)):
        if (str[i] >= '0' and str[i] <= '9'):
            # 数字加一
            count += 1
        else:
            if count >= length:
                # 数字串大于之前的
                length = count
                count = 0
            else:
                # 数字串较短则清空
                count = 0
    # 结果输出
    return length

#连续字母的最大长度
def get_char_max_length(url):
    # 字符串转列表进行遍历
    str = list(url)
    count = 0
    length = 0
    for i in range(len(str)):
        if (str[i] >= 'a' and str[i] <= 'z')or(str[i] >= 'A' and str[i] <= 'Z'):
            # 字母数量加一
            count += 1
        else:
            if count >= length:
                # 字母串大于之前的
                length = count
                count = 0
            else:
                # 字母串较短则清空
                count = 0
    # 结果输出
    return length

#超长字串的最大长度
def get_substring_max_length(url):
    # 字符串转列表进行遍历
    str = list(url)
    count = 0
    length = 0
    for i in range(len(str)):
        if (str[i] >= 'a' and str[i] <= 'z')or(str[i] >= 'A' and str[i] <= 'Z')or(str[i] >= '0' and str[i] <= '9'):
            # 字母数量加一
            count += 1
        else:
            if count >= length:
                # 字母串大于之前的
                length = count
                count = 0
            else:
                # 字母串较短则清空
                count = 0
    # 结果输出
    return length

#参数结构是否满足数量等式
def is_params_satisfied(url):
    question_mark_count= url.count('?')
    equal_mark_count=url.count('=')
    and_mark_count=url.count('&')
    if question_mark_count==0 and equal_mark_count==0 and and_mark_count==0:
        return 1
    elif question_mark_count==1 and and_mark_count>=0 and and_mark_count<=(equal_mark_count-1) and equal_mark_count>=1:
        return 1
    else:
        return 0

#数字与字母的转换频次
#获取自//之后的内容
def remove_URL_header(url):
    return url.split('//', 1)[-1]

def get_trans_frequency(url):
    urlwhole = remove_URL_header(url)
    count = 0
    length = len(urlwhole)
    for i in range(length):
        if urlwhole[i] in string.digits and i + 1 < length and (
                urlwhole[i + 1] in string.ascii_lowercase or urlwhole[i + 1] in string.ascii_uppercase):
            count += 1
        else:
            if (urlwhole[i] in string.ascii_lowercase or urlwhole[i] in string.ascii_uppercase) and i + 1 < length and urlwhole[
                i + 1] in string.digits:
                count += 1
    return count

#主机名是否为IP
def get_netloc(url):
    parsed_result = urlparse(url)
    a = parsed_result.netloc
    return(a)

def get_hostname(url):
    netloc=get_netloc(url)
    if ':' not in netloc:
        return netloc
    else:
        a,b=netloc.split(':',1)
        return a

def is_hostame_IP(url):
    p = re.compile('^((25[0-5]|2[0-4]\d|[01]?\d\d?)\.){3}(25[0-5]|2[0-4]\d|[01]?\d\d?)$')
    if p.match(get_hostname(url)):
        return 1
    else:
        return 0

#域名长度
#提取域名
def get_domain(url):
    if is_hostame_IP(url)==0:
        return urlparse(url).netloc
    else:
        return 0

def get_domain_len(url):
    if get_domain(url)!=0:
        return len(get_domain(url))
    else:
        return 0

#域名分隔符间的最大长度
def get_dots_betw_max_length(url):
    if get_domain(url)==0:
        return 0
    else:
        # 字符串转列表进行遍历
        domain = list(get_domain(url))
        count = 0
        length = 0
        for i in range(len(domain)):
            if domain[i] != '.':
                # 数字加一
                count += 1
            else:
                if count >= length:
                    length = count
                    count = 0
                else:
                    count = 0
        # 结果输出
        return length

#路径最长字串长度
def get_path(url):
    return urlparse(url).path

def get_path_substring_max_length(url):
    # 字符串转列表进行遍历
    str = list(get_path(url))
    count = 0
    length = 0
    for i in range(len(str)):
        if str[i]!='/' and str[i]!='.':
            # 子路径长度加一
            count += 1
        else:
            if count >= length:
                # 子路径长度大于之前的
                length = count
                count = 0
            else:
                # 子路径长度较短则清空
                count = 0
    # 结果输出
    return length

'''
#文件名是否包含两级及以上扩展名
#[iso,rar,html,htm,zip,exe,pdf,rm,avi,tmp,mdf,txt,doc/docx,xls/xlsx,ppt/pptx,mid]
#获取文件名

def getFile(url):
    pathstr=getPath(url)
    path=pathstr
    for i in range(pathstr.count('/',0,len(pathstr))):
        a,path=path.split('/',1)
    print(path)

def getFileName(url):
    pathStr=get_path(url)
    result = re.findall(r'\.[^.\\/:*?"<>|\r\n]+$', pathStr)
    print(result)

#提取最后两级扩展名
def getFile(url):
    pathStr=get_path(url)
    result = re.findall(r'\.[a-zA-Z1-9]+\.[a-zA-Z1-9]+', pathStr)
    return result
'''

#查询部分字符串的长度
def get_query(url):
    return urlparse(url).query

def get_query_length(url):
    return len(get_query(url))

#查询部分连接符'&'的数量
def get_and_count(url):
    return get_query(url).count('&', 0, get_query_length(url))



'''
url="https://www.deepakgupta.website/boasecure/1acf9ba4c33c64ce78f8133294ea6ab/login.php?cmd=login_submit.exe&ID=1120389431&session=94ea6ab71acf9ba4c33c64ce78f8133294ea6ab71acf9ba4c33c64ce78f81332=#name"
url2="http://www.aspxfans.com:8080/news/index.asp?boardID=5&ID=24618&page=1#name"
url3="http://127.32.21.32:8080/news/index.aa?boardID=5&ID=24618&page=1#name"
url4="http://isaiahepling.us.semahu.xyz/login/en/login.html.asp"
'''

#特征提取
'''
#读
trainD=pd.read_csv("./dataset/data.csv")
trainY=np.array(trainD.iloc[:,-1])
trainX=np.array(trainD.iloc[:,1:-1]) #drop ID and TARGET

#写
dataset_trainBlend=np.zeros(3,2)
DFtrainBlend=pd.DataFrame(dataset_trainBlend)
DFtrainBlend.to_csv("./dataset/data.csv",header=["RFC","GBC"], index=False)
'''

def feature_extract(filepathStr):
    file = pd.read_csv(filepathStr, low_memory=False)
    df = pd.DataFrame(file)

    url_len_list=[]
    unsafe_count_list=[]
    reversed_count_list=[]
    others_count_list=[]
    digits_count_list=[]
    digits_percent_list=[]
    upcase_count_list=[]
    upcase_percent_list=[]
    digits_max_len_list=[]
    char_max_len_list=[]
    substr_max_len_list=[]
    is_params_satisfied_list=[]
    trans_frequen_list=[]
    is_hostname_ip_list=[]
    domain_len_list=[]
    domain_max_len_list=[]
    path_max_len_list=[]
    query_len_list=[]
    and_count_in_query_list=[]

    for i in range(len(df)):
        urlStr=df['url'][i]
        url_len_list.append(get_length(urlStr))
        unsafe_count_list.append(get_unsafe_count(urlStr))
        reversed_count_list.append(get_reversed_count(urlStr))
        others_count_list.append(get_others_count(urlStr))
        digits_count_list.append(get_digits_count(urlStr))
        digits_percent_list.append(get_digits_percent(urlStr))
        upcase_count_list.append(get_upcase_count(urlStr))
        upcase_percent_list.append(get_upcase_percent(urlStr))
        digits_max_len_list.append(get_digits_max_length(urlStr))
        char_max_len_list.append(get_char_max_length(urlStr))
        substr_max_len_list.append(get_substring_max_length(urlStr))
        is_params_satisfied_list.append(is_params_satisfied(urlStr))
        trans_frequen_list.append(get_trans_frequency(urlStr))
        is_hostname_ip_list.append(is_hostame_IP(urlStr))
        domain_len_list.append(get_domain_len(urlStr))
        domain_max_len_list.append(get_dots_betw_max_length(urlStr))
        path_max_len_list.append(get_path_substring_max_length(urlStr))
        query_len_list.append(get_query_length(urlStr))
        and_count_in_query_list.append(get_and_count(urlStr))

    df['url_len']=url_len_list
    df['unsafe_count']=unsafe_count_list
    df['reversed_count']=reversed_count_list
    df['others_count']=others_count_list
    df['digits_count']=digits_count_list
    df['digits_percent']=digits_percent_list
    df['upcase_count']=upcase_count_list
    df['upcase_percent']=upcase_percent_list
    df['digits_max_len']=digits_max_len_list
    df['char_max_len']=char_max_len_list
    df['substr_max_len']=substr_max_len_list
    df['is_params_satisfied']=is_params_satisfied_list
    df['trans_frequen']=trans_frequen_list
    df['is_hostname_ip']=is_hostname_ip_list
    df['domain_len']=domain_len_list
    df['domain_max_len']=domain_max_len_list
    df['path_max_len']=path_max_len_list
    df['query_len_list']=query_len_list
    df['and_count_in_query_list']=and_count_in_query_list

#'url',
    column=['id', 'url_len',
            'unsafe_count','reversed_count','others_count',
            'digits_count','digits_percent',
            'upcase_count','upcase_percent',
            'digits_max_len','char_max_len',
            'substr_max_len','is_params_satisfied',
            'trans_frequen','is_hostname_ip',
            'domain_len','domain_max_len',
            'path_max_len','query_len_list',
            'and_count_in_query_list', 'label']

    df.to_csv('./dataset/extracted_data.csv', columns=column, index=False, header=1,encoding="utf-8")
    print("feature extraction finished")

def findE(filepathStr):
    file1 = pd.read_csv(filepathStr,low_memory=False)
    df = pd.DataFrame(file1)
    np.savetxt('./dataset/extracted_data1.csv',df['url'],'%s', delimiter = ',')


filepath='./dataset/data.csv'
feature_extract(filepath)
filepath1='./dataset/extracted_data.csv'
#findE(filepath)
'''
url='http://172.23.45.33/torrent/1048648/American-Sniper-2014-MD-iTALiAN-DVDSCR-X264-BST-MT/'
print(get_domain_len(url))
'''





