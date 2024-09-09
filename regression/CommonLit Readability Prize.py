import pandas as pd
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

##train data FE
object_columns = train.select_dtypes(include='object')# object 타입인 열만 선택
missing_values_in_object = object_columns.columns[object_columns.isnull().any()]# object 타입 열에서 결측값 확인
print(missing_values_in_object)

object_columns = train.select_dtypes(include='int64')# int 타입인 열만 선택
missing_values_in_int64 = object_columns.columns[object_columns.isnull().any()]# int 타입 열에서 결측값 확인
print(missing_values_in_int64)

object_columns = train.select_dtypes(include='float64')# float 타입인 열만 선택
missing_values_in_float64 = object_columns.columns[object_columns.isnull().any()]# float 타입 열에서 결측값 확인
print(missing_values_in_float64)

for col in missing_values_in_float64:
  train[col].fillna(train[col].mean(), inplace=True)

##test data FE
object_columns = test.select_dtypes(include='object')# object 타입인 열만 선택
missing_values_in_object = object_columns.columns[object_columns.isnull().any()]# object 타입 열에서 결측값 확인
print(missing_values_in_object)

object_columns = test.select_dtypes(include='int64')# int 타입인 열만 선택
missing_values_in_int64 = object_columns.columns[object_columns.isnull().any()]# int 타입 열에서 결측값 확인
print(missing_values_in_int64)

object_columns = test.select_dtypes(include='float64')# float 타입인 열만 선택
missing_values_in_float64 = object_columns.columns[object_columns.isnull().any()]# float 타입 열에서 결측값 확인
print(missing_values_in_float64)

for col in missing_values_in_float64:
  test[col].fillna(test[col].mean(), inplace=True)

#labeling
from sklearn.preprocessing import LabelEncoder
categorical_columns = test.select_dtypes(include='object').columns
le = LabelEncoder()
for col in categorical_columns:
  test[col] = le.fit_transform(test[col])

categorical_columns = train.select_dtypes(include='object').columns
le = LabelEncoder()
for col in categorical_columns:
  train[col] = le.fit_transform(train[col])

##data split
from sklearn.model_selection import train_test_split
X = train.drop('price_doc', axis=1)  # Features
y = train['price_doc']               # Target variable
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

##model fitting
import xgboost as xgb
xgb_reg = xgb.XGBRegressor(objective='reg:squarederror', 
                          colsample_bytree=0.3, 
                          learning_rate=0.1,
                          max_depth=5, 
                          alpha=10, 
                          n_estimators=100)

# Fit the model on training data
xgb_reg.fit(X_train, y_train)

# Predict on test data
y_pred = xgb_reg.predict(X_test)

##check
from sklearn.metrics import mean_squared_error
import numpy as np
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
print(f"Mean Squared Error: {mse}")
print(f"Root Mean Squared Error: {rmse}")

##Submission
test_pred = xgb_reg.predict(test)

# 결과를 DataFrame으로 저장
submission = pd.DataFrame({
    'id': test['id'],
    'price_doc': test_pred
})
submission.to_csv('submission_0909.csv', index=False)
