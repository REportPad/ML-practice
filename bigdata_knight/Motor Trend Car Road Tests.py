#This code was executed in colab
!git clone https://github.com/AnalyticsKnight/yemoonsaBigdata
  
import pandas as pd
X_test = pd.read_csv("./yemoonsaBigdata/datasets/Part2/mpg_X_test.csv")
X_train = pd.read_csv("./yemoonsaBigdata/datasets/Part2/mpg_X_train.csv")
y_train = pd.read_csv("./yemoonsaBigdata/datasets/Part2/mpg_y_train.csv")

import numpy as np
print(X_train.info())

from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
X_train[['horsepower']] = imputer.fit_transform(X_train[['horsepower']])
X_test[['horsepower']] = imputer.fit_transform(X_test[['horsepower']])

COL_DEL = ['name']
COL_NUM = ['mpg','cylinders','displacement','horsepower','weight','acceleration','model_year']
COL_CAT = []
COL_Y = ['isUSA']
X_train = X_train.iloc[:,1:]
X_test = X_test.iloc[:,1:]

from sklearn.model_selection import train_test_split
X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, test_size=0.3)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X_tr[COL_NUM])
X_tr[COL_NUM] = scaler.transform(X_tr[COL_NUM])
X_val[COL_NUM] = scaler.transform(X_val[COL_NUM])
X_test[COL_NUM] = scaler.transform(X_test[COL_NUM])

from sklearn.neighbors import KNeighborsClassifier
modelKNN = KNeighborsClassifier(n_neighbors=5, metric='euclidean')
modelKNN.fit(X_tr, y_tr.values.ravel())

from sklearn.tree import DecisionTreeClassifier
modelDT = DecisionTreeClassifier(max_depth=10)
modelDT.fit(X_tr, y_tr)

y_val_pred = modelKNN.predict(X_val)
y_val_pred_probaKNN = modelKNN.predict_proba(X_val)
y_val_pred_probaDT = modelDT.predict_proba(X_val)

from sklearn.metrics import roc_auc_score
scoreKNN = roc_auc_score(y_val, y_val_pred_probaKNN[:,1])
scoreDT = roc_auc_score(y_val, y_val_pred_probaDT[:,1])
print(scoreKNN, scoreDT)

best_model = None
best_score = 0

for i in range(2,10):
  model = KNeighborsClassifier(n_neighbors=i, metric='euclidean')
  model.fit(X_tr, y_tr.values.ravel())
  y_val_pred_proba = model.predict_proba(X_val)
  score = roc_auc_score(y_val, y_val_pred_proba[:,1])
  print(i,"개의 이웃 확인", score)
  if best_score <= score:
    best_model = model
    
#print(best_model.predict_proba(X_test))
pred = best_model.predict_proba(X_test)[:,1]
pd.DataFrame({'isUSA':pred}).to_csv('./yemoonsaBigdata/res/003000000.csv',index=False)
