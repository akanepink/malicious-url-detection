#-*- codeing = utf-8 -*-
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_curve
from tools.data_process import draw_roc_auc,calc_target

'''--------------------------------DecisionTree-------------------------------------'''
def test_DecisionTreeClassifier(*data):
    X_train,X_test,y_train,y_test=data
    clf = DecisionTreeClassifier()
    #clf=DecisionTreeClassifier(max_depth=None,min_samples_split=2)
    clf.fit(X_train,y_train)

    #.score——>平均准确度
    print("Training Score:%f"%clf.score(X_train,y_train))
    print("Testing Score:%f"%clf.score(X_test,y_test))
    print('---------------------------------------')

    # .predict(xtest)——>根据xtest,预测结果
    predict_array = clf.predict(X_test)

    # 画ROC图
    predictions_validation = clf.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, predictions_validation)
    draw_roc_auc(fpr, tpr)

    #计算指标并输出
    calc_target(X_train,X_test,y_train,y_test,predict_array,fpr,tpr)
    print('---------------------------------------')