#-*- codeing = utf-8 -*-

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_curve
from tools.data_process import draw_roc_auc,calc_target

def test_gbdtClassifier(*data):
    X_train,X_test,y_train,y_test=data
    gbm1 = GradientBoostingClassifier(learning_rate=0.1, n_estimators=60, max_depth=7, min_samples_leaf=60,
                                      min_samples_split=1200, max_features='sqrt', subsample=0.8, random_state=10)
    gbm1.fit(X_train, y_train)
    gbm1.fit(X_train, y_train)

    #画ROC曲线图
    predictions_validation=gbm1.predict_proba(X_test)[:,1]
    fpr,tpr,_=roc_curve(y_test,predictions_validation)
    draw_roc_auc(fpr,tpr)
    # .predict(xtest)——>根据xtest,预测结果
    predict_array = gbm1.predict(X_test)
    #计算指标并输出
    calc_target(X_train,X_test,y_train,y_test,predict_array,fpr,tpr)
    print('---------------------------------------')