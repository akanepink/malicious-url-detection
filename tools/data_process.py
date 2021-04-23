#-*- codeing = utf-8 -*-
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve,auc,roc_auc_score
import matplotlib.pyplot as plt

def load_data():
    trainD = pd.read_csv("./dataset/extracted_data.csv", encoding='utf-8')
    trainY = np.array(trainD.iloc[:, -1])
    trainX = np.array(trainD.iloc[:, 1:-1])  # drop ID and TARGET

    #【利用train_test_split方法，将X,y随机划分为训练集（X_train），训练集标签（y_train），测试集（X_test），测试集标签（y_test），按训练集：测试集=7:3的概率划分，到此步骤，可以直接对数据进行处理】
    X_train, X_test, y_train, y_test = train_test_split(trainX, trainY, test_size=0.3, random_state=42)

    #【将训练集与数据集的数据分别保存为CSV文件】
    #np.column_stack将两个矩阵进行组合连接，numpy.savetxt 将txt文件保存为csv格式的文件
    '''
    train= np.column_stack((X_train,y_train))
    np.savetxt('./dataset/train_set.csv',train,'%s', delimiter = ',')

    test = np.column_stack((X_test, y_test))
    np.savetxt("./dataset/test_set.csv", test,'%s', delimiter = ',')
    '''
    return X_train, X_test, y_train, y_test


#计算指标
def calc_target(*data):
    X_train, X_test, y_train, y_test, predict_array ,fpr,tpr= data
    label_array = y_test

    tp = fp = tn = fn = 0

    for i in range(len(y_test)):
        if predict_array[i] == 1 and label_array[i] == 1:
            tp += 1
        elif predict_array[i] == 1 and label_array[i] == 0:
            fp += 1
        elif predict_array[i] == 0 and label_array[i] == 0:
            tn += 1
        else:
            fn += 1
    accuracy = (tp + tn) / (tp + fp + tn + fn)
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = (2 * tp) / (2 * tp + fp + fn)
    roc_auc=auc(fpr,tpr)

    print("准确率Accuracy:%0.4f" % accuracy)
    print("精确度Precision:%0.4f" % precision)
    print("召回率Recall:%0.4f" % recall)
    print("F1值:%0.4f" % f1)
    print("AUC值:%0.4f"%roc_auc)


#画ROC图
def draw_roc_auc(fpr,tpr):
    roc_auc=auc(fpr,tpr)

    plt.title('ROC Validation')
    plt.plot(fpr,tpr,'b',label='AUC=%0.2f'%roc_auc)
    plt.legend(loc='lower right')
    plt.plot([0,1],[0,1],'r--')
    plt.xlim([0,1])
    plt.ylim([0,1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()

