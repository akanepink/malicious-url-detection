#-*- codeing = utf-8 -*-


from tools.data_process import load_data
from models.random_forest import test_RandomForestClassifier_default
from models.decision_tree import test_DecisionTreeClassifier
from models.adaboost import test_adaboostClassifier
from models.gbdt import test_gbdtClassifier


#多算法性能比较
def algorithm_compare():
    X_train, X_test, y_train, y_test = load_data()

    print('------RandomForest------')
    test_RandomForestClassifier_default(X_train, X_test, y_train, y_test)
    print('\n\n')

    print('------DecisionTree------')
    test_DecisionTreeClassifier(X_train, X_test, y_train, y_test)
    print('\n\n')

    print('--------AdaBoost--------')
    test_adaboostClassifier(X_train, X_test, y_train, y_test)
    print('\n\n')

    print('----------GBDT----------')
    test_gbdtClassifier(X_train, X_test, y_train, y_test)


algorithm_compare()
