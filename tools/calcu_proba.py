#-*- codeing = utf-8 -*-
import matplotlib.pyplot as plt
import pandas as pd

def load_date(filepath):
    file = pd.read_csv(filepath, low_memory=False)
    df = pd.DataFrame(file)

    benign_len=[]
    malicious_len=[]

    lenArray_length=300
    begin=0
    middle=70000     #380691 #70000
    end=100000        #600306 #100000
    st1 = middle - begin
    st2 = end - middle

    for i in range(0,lenArray_length):
        benign_len.append(0)
        malicious_len.append(0)

    for i in range(begin,middle):
        url_len = df['url_len'][i]
        if url_len>=lenArray_length:
            continue
        benign_len[url_len]+=1

    for i in range(middle, end):
        url_len = df['url_len'][i]
        if url_len>=lenArray_length:
            continue
        malicious_len[url_len] += 1



    for i in range(0,lenArray_length):
        benign_len[i]=benign_len[i]/st1
   
    for i in range(0,lenArray_length):
        malicious_len[i]=malicious_len[i]/st2

    #,marker='*'
    plt.plot(benign_len, color='red',linewidth=1,label='benign')
    plt.plot(malicious_len, color='blue',linewidth=1,label='malicious',lineStyle='--')
    # 设置图形的标题，并给坐标轴加上标签

    plt.legend(('benign', 'malicious'), loc='upper right')
    plt.title("URL length", fontsize=24)
    plt.xlabel("length", fontsize=14)
    plt.ylabel("rate", fontsize=14)

    # 设置刻度表标记的大小
    plt.tick_params(axis="both", labelsize=14)
    plt.show()


load_date('./dataset/extracted_data.csv')