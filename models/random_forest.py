#-*- codeing = utf-8 -*-

import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score,GridSearchCV
from sklearn import ensemble
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
from tools.data_process import load_data,draw_roc_auc,calc_target


'''--------------------------------RandomForest-------------------------------------'''
#测试RandomForestClassifier的用法
def test_RandomForestClassifier_default(*data):
    X_train, X_test, y_train, y_test = data
    clf = ensemble.RandomForestClassifier()
    clf.fit(X_train, y_train)
    # .score——>平均准确度
    print("Training Score:%f" % clf.score(X_train, y_train))
    print("Testing Score:%f" % clf.score(X_test, y_test))
    print('---------------------------------------')

    # 画ROC曲线图
    predictions_validation = clf.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, predictions_validation)
    draw_roc_auc(fpr, tpr)

    # .predict(xtest)——>根据xtest,预测结果
    predict_array = clf.predict(X_test)
    # 计算指标并输出
    calc_target(X_train, X_test, y_train, y_test, predict_array, fpr, tpr)
    print('---------------------------------------')


''''''
def test_RandomForestClassifier(*data):
    X_train,X_test,y_train,y_test=data
    #clf=ensemble.RandomForestClassifier(n_estimators=350,max_depth=None,min_samples_split=2)
    clf = ensemble.RandomForestClassifier(n_estimators=300,max_depth=None,min_samples_split=2,max_features=10)
    '''
    clf = ensemble.RandomForestClassifier(n_estimators = 250,
                                          min_samples_split = 60,
                                          min_samples_leaf = 10,
                                          max_depth = 21,
                                          random_state = 10,
                                          class_weight=None,
                                          criterion='entropy')
    '''

    clf.fit(X_train,y_train)
    #.score——>平均准确度
    print("Training Score:%f"%clf.score(X_train,y_train))
    print("Testing Score:%f"%clf.score(X_test,y_test))
    print('---------------------------------------')

    #画ROC曲线图
    predictions_validation=clf.predict_proba(X_test)[:,1]
    fpr,tpr,_=roc_curve(y_test,predictions_validation)
    draw_roc_auc(fpr,tpr)

    # .predict(xtest)——>根据xtest,预测结果
    predict_array = clf.predict(X_test)
    #计算指标并输出
    calc_target(X_train,X_test,y_train,y_test,predict_array,fpr,tpr)
    print('---------------------------------------')

    '''
    # 交叉验证
    scores = cross_val_score(clf, X_train, y_train, cv=None)
    print("交叉验证准确率:%0.2f" % scores.mean())
    print('---------------------------------------')


    print("森林信息:%s"%clf.estimators_)
    print("准确率为:%s"%clf.score(X_test,y_test))
    print("AUC值为:%s"%roc_auc_score(y_test,clf.predict_proba(X_test)[:,1]))
    print("各特征重要性:%s"%clf.feature_importances_)
    '''

    #输出特征重要性排行榜与图
    trainD = pd.read_csv("./dataset/extracted_data.csv", encoding='utf-8')

    importances = clf.feature_importances_
    std = np.std([tree.feature_importances_ for tree in clf.estimators_], axis=0)
    indices = np.argsort(importances)[::-1]
    print("Feature ranking:")
    for f in range(min(20, X_train.shape[1])):
        print("%2d) %-*s %f" %(f+1,30,trainD.columns[indices[f]+1],importances[indices[f]]))
    plt.figure()
    plt.title("Feature importances")
    plt.bar(range(X_train.shape[1]),importances[indices],color="r",yerr=std[indices],align="center")
    plt.xticks(range(X_train.shape[1]),indices)
    plt.xlim([-1,X_train.shape[1]])
    plt.show()



#测试 RandomForestClassifier 的预测性能随 n_estimators 参数的影响
def test_RandomForestClassifier_num(*data):
    X_train,X_test,y_train,y_test=data
    nums=np.arange(2,50,3)
    fig=plt.figure()
    ax=fig.add_subplot(1,1,1)
    testing_scores=[]
    training_scores=[]
    for num in nums:
        # 决策树的个数
        clf=ensemble.RandomForestClassifier(n_estimators=num)
        clf.fit(X_train,y_train)
        training_scores.append(clf.score(X_train,y_train))
        testing_scores.append(clf.score(X_test,y_test))
    ax.plot(nums,training_scores,label="Training Score")
    ax.plot(nums,testing_scores,label="Testing Score")
    ax.set_xlabel("estimator num")
    ax.set_ylabel("score")
    ax.legend(loc="lower right")
    ax.set_ylim(0.9,1.05)
    plt.suptitle("RandomForestClassifier")
    # 设置 X 轴的网格线，风格为 点画线
    plt.grid(axis='x',linestyle='-.')
    plt.show()

#交叉验证
def test_cross_val_score(*data):
    X_train, X_test, y_train, y_test = data
    clf=ensemble.RandomForestClassifier(n_estimators=10,max_depth=None,min_samples_split=2,random_state=0,cv=None)
    scores=cross_val_score(clf,X_train,y_train)
    print(scores.mean())

#调整超参数
def test_gridSearchCV(*data):
    X_train, X_test, y_train, y_test = data

    param_test1={'n_estimators':range(25,500,25)}
    gsearch1 = GridSearchCV(estimator=ensemble.RandomForestClassifier(),
                            param_grid=param_test1,
                            scoring='roc_auc',
                            cv=5)
    '''
    gsearch1 = GridSearchCV(estimator=ensemble.RandomForestClassifier(min_samples_split=2,
                                                                      min_samples_leaf=20,
                                                                      max_depth=8,
                                                                      random_state=10),
                            param_grid=param_test1,
                            scoring='roc_auc',
                            cv=5)
    '''
    '''
    gsearch1=GridSearchCV(estimator=ensemble.RandomForestClassifier(min_samples_split=100,
                                                                    min_samples_leaf=20,
                                                                    max_depth=8,
                                                                    random_state=10),
                          param_grid=param_test1,
                          scoring='roc_auc',
                          cv=5)
    '''
    gsearch1.fit(X_train,y_train)
    print(gsearch1.best_params_,gsearch1.best_score_)

    best_estimator=gsearch1.best_params_['n_estimators']
    print("决策树棵树最优值：%d"%best_estimator)


    param_test2={'min_samples_split':range(60,200,20),'min_samples_leaf':range(10,110,10)}
    gsearch2=GridSearchCV(estimator=ensemble.RandomForestClassifier(n_estimators=best_estimator,
                                                                    max_depth=8,
                                                                    random_state=10),
                          param_grid=param_test2,
                          scoring='roc_auc',
                          cv=5)
    gsearch2.fit(X_train, y_train)
    print(gsearch2.best_params_,gsearch2.best_score_)

    best_split = gsearch2.best_params_['min_samples_split']
    best_leaf = gsearch2.best_params_['min_samples_leaf']

    print("决策树分割最小样本数量最优值：%d" % best_split)
    print("决策树叶子节点最小样本数量最优值：%d" % best_leaf)


    #最优最大深度
    param_test3={'max_depth':range(3,30,2)}
    gsearch3=GridSearchCV(estimator=ensemble.RandomForestClassifier(n_estimators=300,
                                                                    min_samples_split=60,
                                                                    min_samples_leaf=10,
                                                                    random_state=10),
                          param_grid=param_test3,
                          scoring='roc_auc',
                          cv=5)
    gsearch3.fit(X_train, y_train)
    print(gsearch3.best_params_,gsearch3.best_score_)

    best_depth = gsearch3.best_params_['max_depth']
    print("决策树最大深度最优值：%d" % best_depth)

    #系数；权重
    param_test4={'criterion':['gini','entropy'],'class_weight':[None,'balanced']}
    gsearch4=GridSearchCV(estimator=ensemble.RandomForestClassifier(n_estimators=300,
                                                                    min_samples_split=60,
                                                                    min_samples_leaf=10,
                                                                    max_depth=21,
                                                                    random_state=10),
                          param_grid=param_test4,
                          scoring='roc_auc',
                          cv=5)
    gsearch4.fit(X_train, y_train)
    print(gsearch4.best_params_,gsearch4.best_score_)


    best_criterion = gsearch4.best_params_['criterion']
    best_weight = gsearch4.best_params_['class_weight']

    print("随机森林系数最优值：%d" % best_criterion)
    print("随机森林权重最优值：%d" % best_weight)

'''---------------------------------------------------------------------------------'''




#专注随机森林
def rf_generate():
    X_train, X_test, y_train, y_test =load_data()
    test_RandomForestClassifier(X_train, X_test, y_train, y_test)

    # 寻找最优模型对应的决策树棵数
    #test_RandomForestClassifier_num(X_train, X_test, y_train, y_test)

    #test_gridSearchCV(X_train, X_test, y_train, y_test)


#rf_generate()
