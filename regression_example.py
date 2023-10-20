#kaggle TPS-2021-AUG
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import lightgbm as lgb

train_data = pd.read_csv('TPS-Feb-2021/train.csv')
test_data = pd.read_csv('TPS-Feb-2021/test.csv')

y = train_data.iloc[:, -1] #target, target이 여러개면 숫자 변경
X = train_data.iloc[:, 1:-1] #data
X_train=X
y_train=y
X_test = test_data.iloc[:,1:]

cat_columns = X_train.columns[X_train.dtypes==object].tolist()
for c in cat_columns:
    X_train[c] = X_train[c].astype('category')

cat_columns = X_test.columns[X_test.dtypes==object].tolist()
for c in cat_columns:
    X_test[c] = X_test[c].astype('category')

lgb_reg = lgb.LGBMRegressor()
lgb_reg.fit(X_train, y_train)
lgb_pred = lgb_reg.predict(X_test)

submission = pd.read_csv('TPS-Feb-2021/sample_submission.csv')
submission['target'] = lgb_pred
submission.to_csv('submission_20231020.csv', index=False)
print("Your submission was successfully saved!")
