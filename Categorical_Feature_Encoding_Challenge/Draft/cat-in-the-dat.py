# -*- coding: utf-8 -*-
"""
Created on Mon Nov 25 23:52:40 2019

@author: anlan
"""

[https://www.kaggle.com/abhishek/entity-embeddings-to-handle-categories]
[https://www.kaggle.com/peterhurford/why-not-logistic-regression]
[https://www.kaggle.com/vikassingh1996/handling-categorical-variables-encoding-modeling]
[https://www.kaggle.com/shahules/an-overview-of-encoding-techniques/notebook]
[https://www.kaggle.com/ruchibahl18/categorical-data-encoding-techniques]
[https://www.kaggle.com/discdiver/category-encoders-examples]
 
 

import numpy as np
import pandas as pd


df_train = pd.read_csv("Desktop/cat-in-the-dat/train.csv" , index_col=['id'])
df_test = pd.read_csv("Desktop/cat-in-the-dat/test.csv" , index_col=['id'])

df_train.dtypes
df_train.head()
df_train.shape
df_train.dtypes.value_counts()
df_train.columns
df_train.describe

# Good show
df_train.info()

display(df_train.head())

X = df_train.drop("target", axis = 1)
y = df_train.loc[:,"target"]

X["bin_3"] = X["bin_3"].apply(lambda x: 1 if x == "T" else 0)
X["bin_4"] = X["bin_4"].apply(lambda x: 1 if x == "Y" else 0)

#pd.get_dummies(X[["nom_0", "nom_1", "nom_2", "nom_3", "nom_4"]])

X = X.drop(["nom_0", "nom_1", "nom_2", "nom_3", "nom_4"], axis=1) \
        .join(pd.get_dummies(X[["nom_0", "nom_1", "nom_2", "nom_3", "nom_4"]]))


from sklearn.feature_extraction import FeatureHasher

#h = FeatureHasher(input_type='string', n_features=1000)
#X[['nom_5', 'nom_6', 'nom_7', 'nom_8', 'nom_9']].values
#hash_X = h.fit_transform(X[['nom_5', 'nom_6', 'nom_7', 'nom_8', 'nom_9']].values)
#hash_X = pd.DataFrame(hash_X.toarray())

from category_encoders import LeaveOneOutEncoder
loo_encoder = LeaveOneOutEncoder(cols=["nom_5", "nom_6", "nom_7", "nom_8", "nom_9"])
loo_X = loo_encoder.fit_transform(X[["nom_5", "nom_6", "nom_7", "nom_8", "nom_9"]], y)
X = X.drop(["nom_5", "nom_6", "nom_7", "nom_8", "nom_9"], axis=1).join(loo_X)

X.ord_1.replace(to_replace = ['Novice', 'Contributor','Expert', 'Master', 'Grandmaster'],
                         value = [0, 1, 2, 3, 4], inplace = True)

X.ord_2.replace(to_replace = ['Freezing', 'Cold', 'Warm', 'Hot','Boiling Hot', 'Lava Hot'],
                         value = [0, 1, 2, 3, 4, 5], inplace = True)


from sklearn.preprocessing import LabelEncoder
for i in ["ord_3", "ord_4"]:
    le = LabelEncoder()
    X[[i]] = le.fit_transform(X[[i]])


from sklearn.preprocessing import OrdinalEncoder
oe = OrdinalEncoder(categories='auto')
X.ord_5 = oe.fit_transform(X.ord_5.values.reshape(-1,1))




def date_cyc_enc(df, col, max_vals):
    df[col + '_sin'] = np.sin(2 * np.pi * df[col]/max_vals)
    df[col + '_cos'] = np.cos(2 * np.pi * df[col]/max_vals)
    return df
X = date_cyc_enc(X, 'day', 7)
X = date_cyc_enc(X, 'month', 12)
X.drop(['day', 'month'], axis=1, inplace = True)


'''
* Tree based models does not depend on scaling
* Non-tree based models hugely depend on scaling
'''

'''For linear model'''

from sklearn.svm import LinearSVC
from sklearn.model_selection import GridSearchCV, KFold, StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

#min_max_scaler = MinMaxScaler()
#X_minmax = min_max_scaler.fit_transform(X)
scaler = StandardScaler()
X_new = scaler.fit_transform(X)


lr = LogisticRegression()
scores_lr = cross_val_score(lr, X_new, y, cv=5)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores_lr.mean(), scores_lr.std() * 2))

rc = RidgeClassifier()
scores_rc = cross_val_score(rc, X_new, y, cv=5)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores_rc.mean(), scores_rc.std() * 2))

lda = LinearDiscriminantAnalysis()
scores_lda = cross_val_score(lda, X_new, y, cv=5)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores_lda.mean(), scores_lda.std() * 2))

linear_svm = LinearSVC(C=0.01, penalty="l2")
scores_linear_svm = cross_val_score(linear_svm, X, y, cv=5)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores_linear_svm.mean(), scores_linear_svm.std() * 2))


'''For classifier model'''
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from xgboost import XGBClassifier

fr = DecisionTreeClassifier(random_state=0)
scores_dt = cross_val_score(fr, X, y, cv=5)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores_dt.mean(), scores_dt.std() * 2))


sgdc = SGDClassifier()
scores_sgdc = cross_val_score(sgdc, X, y, cv=5)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores_sgdc.mean(), scores_sgdc.std() * 2))

ab = AdaBoostClassifier()
scores_ab= cross_val_score(ab, X, y, cv=5)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores_ab.mean(), scores_ab.std() * 2))

gbm = GradientBoostingClassifier()
scores_gbm= cross_val_score(gbm, X, y, cv=5)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores_gbm.mean(), scores_gbm.std() * 2))

rf = RandomForestClassifier()
scores_rf= cross_val_score(rf, X, y, cv=5)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores_rf.mean(), scores_rf.std() * 2))

et = ExtraTreesClassifier()
scores_et= cross_val_score(et, X, y, cv=5)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores_et.mean(), scores_et.std() * 2))

xgb = XGBClassifier()
scores_xgb= cross_val_score(xgb, X, y, cv=5)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores_xgb.mean(), scores_xgb.std() * 2))


from sklearn.model_selection import RandomizedSearchCV

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

random_search = RandomizedSearchCV(xgb, param_distributions=params, n_iter=param_comb, 
                                   scoring='accuracy', n_jobs=1, cv=skf.split(X, y), 
                                   verbose=3, random_state=1001 )
random_search.fit(X, y)

print('\n All results:')
print(random_search.cv_results_)
print('\n Best estimator:')
print(random_search.best_estimator_)
print('\n Best hyperparameters:')
print(random_search.best_params_)
results = pd.DataFrame(random_search.cv_results_)

means = random_search.cv_results_['mean_test_score']
stds = random_search.cv_results_['std_test_score']
for mean, std, params in zip(means, stds, random_search.cv_results_['params']):
    print("%0.3f (+/-%0.03f) for %r"% (mean, std * 2, params))










fr = DecisionTreeClassifier(random_state=12345).fit(X, y)
#print(fr.score(X_test, y_test))
fr_feature_importances = pd.DataFrame(fr.feature_importances_, 
                            index = X.columns,
                             columns=['importance'])
print(fr_feature_importances.sort_values("importance",ascending=False))

X2 = X.drop(X.columns[fr.feature_importances_ > 0.01], axis=1)
fr = DecisionTreeClassifier(random_state=0)
scores_fr2 = cross_val_score(fr, X2, y, cv=5)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores_fr2.mean(), scores_fr2.std() * 2))






