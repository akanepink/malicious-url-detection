#-*- codeing = utf-8 -*-
from models.random_forest import test_gridSearchCV
from tools.data_process import load_data

def rf_generate():
    X_train, X_test, y_train, y_test =load_data()
    #test_RandomForestClassifier(X_train, X_test, y_train, y_test)

    test_gridSearchCV(X_train, X_test, y_train, y_test)

rf_generate()