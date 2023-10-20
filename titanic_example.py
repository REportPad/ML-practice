import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import lightgbm as lgb

train_data = pd.read_csv('titanic/train.csv')
test_data = pd.read_csv('titanic/test.csv')

train_data.columns
train_data = train_data[['PassengerId', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp',
       'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked','Survived']]

y = train_data.iloc[:, -1] #target, target이 여러개면 숫자 변경
X = train_data.iloc[:, :-1] #data
X_train, X_valid, y_train, y_valid = train_test_split(X, y, random_state=42)

lgb_clf = lgb.LGBMClassifier() #분류일 경우
cat_columns = X_train.columns[X_train.dtypes==object].tolist()
for c in cat_columns:
    X_train[c] = X_train[c].astype('category')
    X_valid[c] = X_valid[c].astype('category')

cat_columns = test_data.columns[test_data.dtypes==object].tolist()
for c in cat_columns:
    test_data[c] = test_data[c].astype('category')

lgb_clf.fit(X_train, y_train)
lgb_pred = lgb_clf.predict(test_data)
output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': lgb_pred})
output.to_csv('submission2.csv', index=False)
print("Your submission was successfully saved!")
