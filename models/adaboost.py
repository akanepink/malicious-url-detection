#-*- codeing = utf-8 -*-


from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import roc_curve
from tools.data_process import draw_roc_auc,calc_target

def test_adaboostClassifier(*data):
    X_train,X_test,y_train,y_test=data
    bdt = AdaBoostClassifier()
    '''
    bdt = AdaBoostClassifier(ensemble.DecisionTreeClassifier(max_depth=2, min_samples_split=20, min_samples_leaf=5),
                             algorithm="SAMME",
                             n_estimators=300, learning_rate=0.5)
    '''
    bdt.fit(X_train, y_train)

    #画ROC曲线图
    predictions_validation=bdt.predict_proba(X_test)[:,1]
    fpr,tpr,_=roc_curve(y_test,predictions_validation)
    draw_roc_auc(fpr,tpr)
    # .predict(xtest)——>根据xtest,预测结果
    predict_array = bdt.predict(X_test)
    #计算指标并输出
    calc_target(X_train,X_test,y_train,y_test,predict_array,fpr,tpr)
    print('---------------------------------------')