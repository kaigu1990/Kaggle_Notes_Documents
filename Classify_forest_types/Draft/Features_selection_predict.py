# -*- coding: utf-8 -*-
"""
Created on Sun Oct 13 20:18:50 2019

@author: anlan
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.preprocessing import PowerTransformer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn import svm
from sklearn.model_selection import cross_val_score,cross_validate, train_test_split, GridSearchCV, StratifiedKFold
from sklearn.feature_selection import VarianceThreshold
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier, ExtraTreesClassifier
from mlxtend.classifier import StackingCVClassifier

np.random.seed(12345)

df_train = pd.read_csv("Desktop/learn-together/train.csv" , index_col=['Id'])
df_test = pd.read_csv("Desktop/learn-together/test.csv" , index_col=['Id'])

df_train.head()
df_train.shape
df_train.dtypes.value_counts()
df_train.columns
df_train.describe
df_train.iloc[:,10:].columns

df_train.iloc[:,10:] = df_train.iloc[:,10:].astype("category")

df_train.isna().sum()
df_train.isna().sum().sum()

df_train[df_train.duplicated()].shape

df_train.describe()
df_train.iloc[:,:10].describe()


X_train, X_test, y_train, y_test = train_test_split(df_train.iloc[:,:10], df_train.loc[:,'Cover_Type'], 
                                                    test_size=0.33, random_state=42)

'''
LinearSVC

基于liblinear库实现
有多种惩罚参数和损失函数可供选择
训练集实例数量大（大于1万）时也可以很好地进行归一化
既支持稠密输入矩阵也支持稀疏输入矩阵
多分类问题采用one-vs-rest方法实现
'''

linear_svm = LinearSVC(C=0.01, penalty="l2").fit(X_train, y_train)
# linear_svm.coef_
pred = linear_svm.predict(X_test)
print(classification_report(y_test, pred, labels=None))
slt = SelectFromModel(linear_svm, prefit=True)
X_train_s = slt.transform(X_train)
X_test_s = slt.transform(X_test)
num_features = X_train_s.shape[1]
#linear_svm = LinearSVC(C=0.01, penalty="l2").fit(X_train_s, y_train)
#pred = linear_svm.predict(X_test_s)
#print(classification_report(y_test, pred, labels=None))
print(X_train.columns[slt.get_support()])

'''
DecisionTreeClassifier
随机森林
'''
fr = DecisionTreeClassifier(random_state=0).fit(X_train, y_train)
print(fr.score(X_test, y_test))
importances = pd.DataFrame({'Features': X_test.columns, 
                                'Importances': fr.feature_importances_})
pred = fr.predict(X_test)
print(classification_report(y_test,pred, labels=None))
model = SelectFromModel(fr, prefit=True)
X_new = model.transform(X_train)
X_new.shape
X_new[:5,:5]
pd.DataFrame(X_new).describe()
print(X_train.columns[model.get_support()])



'''
https://www.cnblogs.com/NewBee-CHH/p/10880933.html
SGDClassifier是一个线性分类器（默认情况下,它是一个线性SVM），它使用SGD进行训练（即，使用SGD查找损失的最小值）
'''
sgdc = SGDClassifier()
sgdc.fit(X_train, y_train)
pred = sgdc.predict(X_test)
print(classification_report(y_test,pred, labels=None))


''' 
定性的指标
'''
X = df_train.select_dtypes("category").drop(columns=["Cover_Type"])
'''
LabelEncoder可以将标签分配一个0—n_classes-1之间的编码
d = defaultdict(LabelEncoder)
fit = X.apply(lambda x: d[x.name].fit_transform(x))
fit.columns
Y_train = df_train.loc[:,'Cover_Type']
X_train, X_test, y_train, y_test = train_test_split(fit, Y_train, test_size=0.33, random_state=42)
'''
X_train, X_test, y_train, y_test = train_test_split(df_train.iloc[:,10:54], df_train.loc[:,'Cover_Type'], 
                                                    test_size=0.33, random_state=123456)
clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)
pred = clf.predict(X_test)
accuracy = accuracy_score(pred, y_test)
print(clf)
print(classification_report(pred, y_test, labels=None))
feature_importances = pd.DataFrame(clf.feature_importances_,
                                   index = X_train.columns,
                                    columns=['importance']).sort_values('importance',ascending=False)
feature_importances

'''
PowerTransformer目前提供了两个这样的幂变换，Yeo-Johnson变换和Box-Cox变换，利用极大似然估计了稳定方差和最小偏度的最优参数。
并且，box-Cox要求输入数据严格为正数据，而Yeo-Johnson支持正负数据
'''
columns_t_analyze = df_train.iloc[:,:10]
columns_transformed =  PowerTransformer(method='yeo-johnson').fit_transform(columns_t_analyze)
columns_transformed = pd.DataFrame(columns_transformed)
columns_transformed.columns = columns_t_analyze.columns
columns_transformed = pd.concat([columns_transformed, df_train.loc[:,"Cover_Type"]], axis=1, join='inner')

'''
d = defaultdict(LabelEncoder)
fit = X.apply(lambda x: d[x.name].fit_transform(x))
fit.reset_index(drop=True, inplace=True)
columns_transformed.reset_index(drop=True, inplace=True)
features_preprocessing = pd.concat([fit, columns_transformed], axis=1, join='inner')
'''
features_preprocessing = pd.concat([columns_transformed, df_train.iloc[:,10:54]], axis=1, join="inner")

selected_columns=["Elevation", 'Horizontal_Distance_To_Roadways', 'Horizontal_Distance_To_Fire_Points', 'Wilderness_Area4', 'Soil_Type10', 
                  'Soil_Type38', 'Soil_Type39', 'Soil_Type40', 'Soil_Type4', 'Soil_Type3', 'Soil_Type17', 'Soil_Type2']

X_train, X_test, y_train, y_test = train_test_split(features_preprocessing.loc[:,selected_columns], 
                                                    features_preprocessing.loc[:,'Cover_Type'], 
                                                    test_size=0.33, random_state=42)

''' KNN '''
for i in range(3, 21, 3):
    neigh = KNeighborsClassifier(n_neighbors=i)
    neigh.fit(X_train, y_train)
    pred = neigh.predict(X_test)
    print("KNeighborsClassifier {}".format(i))
    print(classification_report(pred, y_test, labels=None))

''' Naive Bayes '''
gnb = GaussianNB()
gnb.fit(X_train, y_train)
pred = gnb.predict(X_test)
## accuracy
accuracy = accuracy_score(y_test,pred)
print("naive_bayes")
print(classification_report(y_test,pred, labels=None))



''' SVM '''
Sv=svm.SVC(gamma='scale',kernel='rbf')
Sv.fit(X_train, y_train)

pred = Sv.predict(X_test)
# accuracy
accuracy = accuracy_score(y_test,pred)
print(classification_report(y_test,pred, labels=None))


''' RF '''
clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)
pred = clf.predict(X_test)
accuracy = accuracy_score(pred, y_test)
print(clf)
print(classification_report(pred, y_test, labels=None))


''' xgboost '''
xgb = XGBClassifier(max_depth=10, subsample=0.8, colsample_bytree=0.7,missing=-999)

xgb.fit(X_train, y_train)
pred = xgb.predict(X_test)
accuracy = accuracy_score(pred, y_test)
print(xgb)
print(classification_report(pred, y_test, labels=None))


''' K fold approach xgboost'''
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import roc_auc_score
params = {
        'min_child_weight': [1, 5, 10, 13, 15],
        'gamma': [0.5, 1, 1.5, 2, 5],
        'subsample': [0.2, 0.4, 0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0],
        'max_depth': [3, 4, 5, 10, 20]
        }

xgb = XGBClassifier(silent=True, nthread=1)
folds = 3
param_comb = 5

skf = StratifiedKFold(n_splits=folds, shuffle = True, random_state = 1001)

random_search = RandomizedSearchCV(xgb, param_distributions=params, n_iter=param_comb, scoring='accuracy', n_jobs=4, cv=skf.split(X_train, y_train), verbose=3, random_state=1001 )

random_search.fit(X_train, y_train)

print('\n All results:')
print(random_search.cv_results_)
print('\n Best estimator:')
print(random_search.best_estimator_)
print('\n Best hyperparameters:')
print(random_search.best_params_)
results = pd.DataFrame(random_search.cv_results_)
results.to_csv('xgb-random-grid-search-results-01.csv', index=False)

xgb = XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
              colsample_bynode=1, colsample_bytree=0.8, gamma=5,
              learning_rate=0.1, max_delta_step=0, max_depth=10,
              min_child_weight=10, missing=None, n_estimators=100, n_jobs=1,
              nthread=1, objective='multi:softprob', random_state=0,
              reg_alpha=0, reg_lambda=1, scale_pos_weight=1, seed=None,
              silent=True, subsample=0.8, verbosity=1)

xgb.fit(X_train, y_train)
pred = xgb.predict(X_test)
print(classification_report(pred, y_test, labels=None))


'''
K fold approach svm
sklearn 的网格搜索（GridSearchCV）
[使用GridSearchCV（网格搜索），快速选择超参数](https://zhuanlan.zhihu.com/p/30103449)
'''

parameters = {'kernel':('linear', 'rbf'), 'C':[1, 10]}
svc = svm.SVC(gamma="scale")
clf = GridSearchCV(svc, parameters, cv=5)
clf.fit(X_train, y_train)
print('\n All results:')
print(clf.cv_results_)
print('\n Best estimator:')
print(clf.best_estimator_)
print('\n Best hyperparameters:')
print(clf.best_params_)

clf = svm.SVC(C=10, cache_size=200, class_weight=None, coef0=0.0,
    decision_function_shape='ovr', degree=3, gamma='scale', kernel='rbf',
    max_iter=-1, probability=False, random_state=None, shrinking=True,
    tol=0.001, verbose=False)
clf.fit(X_train, y_train)
pred = clf.predict(X_test)
print(classification_report(pred, y_test, labels=None))



'''
LGBMClassifier
'''
X_train, X_test, y_train, y_test = train_test_split(df_train.iloc[:,:10], df_train.loc[:,'Cover_Type'], 
                                                    test_size=0.33, random_state=42)


lgbc = LGBMClassifier(n_estimators=500, learning_rate= 0.1, 
               objective= 'multiclass', num_class=7, #class_weight=class_weight_lgbm, 
               random_state= 2019, n_jobs=-1)
lgbc.fit(X_train, y_train)
pred = lgbc.predict(X_test)
print(classification_report(pred, y_test, labels=None))
feature_importances = pd.DataFrame(lgbc.feature_importances_,
                                   index = X_train.columns,
                                    columns=['importance']).sort_values('importance',ascending=False)

X_train, X_test, y_train, y_test = train_test_split(df_train.iloc[:,10:54], df_train.loc[:,'Cover_Type'], 
                                                    test_size=0.33, random_state=42)


#==================== LAST TEST ========================#

X_train, X_test, y_train, y_test = train_test_split(df_train.iloc[:,:54], df_train.loc[:,'Cover_Type'], 
                                                    test_size=0.33, random_state=42)

lgbc = LGBMClassifier(n_estimators=500, learning_rate= 0.1, 
               objective= 'multiclass', num_class=7,
               random_state= 12345, n_jobs=-1)
lgbc.fit(X_train, y_train)
pred = lgbc.predict(X_test)
print(classification_report(pred, y_test, labels=None))
lgbc_feature_importances = pd.DataFrame(lgbc.feature_importances_,
                                   index = X_train.columns,
                                    columns=["importance"])
print(lgbc_feature_importances.sort_values("importance",ascending=False))
print(X_train.columns[lgbc_feature_importances["importance"] == 0])

def get_LGBC():
    return LGBMClassifier(n_estimators=500, learning_rate= 0.1, 
               objective= 'multiclass', num_class=7,
               random_state= 12345, n_jobs=-1)


for thre in [0,50,100,200,500]:
    print(np.mean(cross_val_score(get_LGBC(), 
                                  X_train.drop(X_train.columns[lgbc.feature_importances_<thre], axis=1), 
                                  y_train, cv=5)))

X_train.drop(X_train.columns[lgbc.feature_importances_ == 0], axis=1, inplace=True)
X_test.drop(X_test.columns[lgbc.feature_importances_ == 0], axis=1, inplace=True)

fr = DecisionTreeClassifier(random_state=12345).fit(X_train, y_train)
print(fr.score(X_test, y_test))
fr_feature_importances = pd.DataFrame(fr.feature_importances_, 
                            index = X_train.columns,
                             columns=['importance'])
print(fr_feature_importances.sort_values("importance",ascending=False))
print(X_train.columns[fr_feature_importances["importance"] == 0])

for thre in [0,0.0001,0.001,0.005,0.01]:
    print(np.mean(cross_val_score(DecisionTreeClassifier(random_state=12345), 
                                    X_train.drop(X_train.columns[fr.feature_importances_<thre], axis=1), 
                                    y_train, cv=5)))

X_train.drop(X_train.columns[fr.feature_importances_ == 0], axis=1, inplace=True)
X_test.drop(X_test.columns[fr.feature_importances_ == 0], axis=1, inplace=True)


ab_clf = AdaBoostClassifier(n_estimators=200,
                            base_estimator=DecisionTreeClassifier(
                                min_samples_leaf=2,
                                random_state=12345),
                            random_state=12345)
   
rf_clf = RandomForestClassifier(n_estimators=300,
                                random_state=12345,
                                n_jobs=1)

xgb_clf = XGBClassifier(n_estimators = 500, 
                        booster='gbtree', 
                        colsample_bylevel=1, 
                        colsample_bynode=1, 
                        colsample_bytree=0.8, 
                        gamma=5,
                        nthread=1, 
                        learning_rate=0.1,
                        max_delta_step=0, 
                        max_depth=10,
                        min_child_weight=10, 
                        missing=None, 
                        random_state= 12345,
                        n_jobs=1)                     

et_clf = ExtraTreesClassifier(n_estimators=300,
                              min_samples_leaf=1,
                              min_samples_split=2,
                              max_depth=50,
                              max_features=0.3,
                              bootstrap = False,
                              random_state=12345,
                              n_jobs=1)

lg_clf = LGBMClassifier(n_estimators=300,
                        num_leaves=128,
                        learning_rate= 0.1,
                        verbose=-1,
                        num_class=7,
                        random_state=12345,
                        n_jobs=1)

ensemble = [("AdaBoostClassifier", ab_clf),
            ("RandomForestClassifier", rf_clf),
            ("XGBClassifier", xgb_clf),
            ("ExtraTreesClassifier", et_clf),
            ("LGBMClassifier", lg_clf)]

print('> Cross-validating classifiers')
for label, clf in ensemble:
    score = cross_val_score(clf, X_train, y_train,
                            cv=5,
                            scoring='accuracy',
                            verbose=0,
                            n_jobs=-1)

    print('  -- {: <24} : {:.3f} : {}'.format(label, np.mean(score), np.around(score, 3)))
    

print('> Fitting stack')

stack = StackingCVClassifier(classifiers=[ab_clf, rf_clf, xgb_clf, et_clf, lg_clf],
                             meta_classifier=rf_clf,
                             cv=5,
                             stratify=True,
                             shuffle=True,
                             use_probas=True,
                             use_features_in_secondary=True,
                             verbose=1,
                             random_state=12345,
                             n_jobs=-1)

stack = stack.fit(X_train, y_train)

X_test = np.array(X_test)
print('> Making predictions')
pred = stack.predict(X_test)
print(classification_report(pred, y_test, labels=None))


#predictions = pd.Series(pred, index=X_test.index, dtype=y_train.dtype)


# ======================================================================== #
sel = VarianceThreshold(threshold=0)
df_train_new = sel.fit_transform(df_train)
#sel.get_support(df_train)
sel.get_support(indices=True)






