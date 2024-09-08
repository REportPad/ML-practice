import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# data load
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

# info & describe
# print(train.info())
# print(train.describe())

#train FE
object_columns = train.select_dtypes(include='object')# object 타입인 열만 선택
missing_values_in_object = object_columns.columns[object_columns.isnull().any()]# object 타입 열에서 결측값 확인
print(missing_values_in_object)

object_columns = train.select_dtypes(include='int64')# int 타입인 열만 선택
missing_values_in_object = object_columns.columns[object_columns.isnull().any()]# int 타입 열에서 결측값 확인
print(missing_values_in_object)

object_columns = train.select_dtypes(include='float64')# float 타입인 열만 선택
missing_values_in_object = object_columns.columns[object_columns.isnull().any()]# float 타입 열에서 결측값 확인
print(missing_values_in_object)

# check None or NaN
has_missing_values = train.isnull().any().any()
print("None 또는 NaN이 있는지 여부:", has_missing_values)

#test FE
object_columns = test.select_dtypes(include='object')# object 타입인 열만 선택
missing_values_in_object = object_columns.columns[object_columns.isnull().any()]# object 타입 열에서 결측값 확인
print(missing_values_in_object)

object_columns = test.select_dtypes(include='int64')# int64 타입인 열만 선택
missing_values_in_object = object_columns.columns[object_columns.isnull().any()]# object 타입 열에서 결측값 확인
print(missing_values_in_object)

object_columns = test.select_dtypes(include='float64')# float64 타입인 열만 선택
missing_values_in_object = object_columns.columns[object_columns.isnull().any()]# object 타입 열에서 결측값 확인
print(missing_values_in_object)

# 데이터에 None 또는 NaN이 있는지 확인
has_missing_values = test.isnull().any().any()
print("None 또는 NaN이 있는지 여부:", has_missing_values)

# 레이블 인코딩 적용
categorical_columns = train.select_dtypes(include='object').columns
le = LabelEncoder()
for col in categorical_columns:
  train[col] = le.fit_transform(train[col])

categorical_columns = test.select_dtypes(include='object').columns
le = LabelEncoder()
for col in categorical_columns:
  test[col] = le.fit_transform(test[col])

X = train.drop('y', axis=1)  # Features
y = train['y']               # Target variable

# Splitting data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the XGBoost Regressor
xg_reg = xgb.XGBRegressor(objective='reg:squarederror', 
                          colsample_bytree=0.3, 
                          learning_rate=0.1,
                          max_depth=5, 
                          alpha=10, 
                          n_estimators=100)

# Fit the model on training data
xg_reg.fit(X_train, y_train)

# Predict on test data
y_pred = xg_reg.predict(X_test)

##check
import numpy as np
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
print(f"Mean Squared Error: {mse}")
print(f"Root Mean Squared Error: {rmse}")

#Submission
test_pred = xg_reg.predict(test)

# 결과를 DataFrame으로 저장
submission = pd.DataFrame({
    'ID': test['ID'],
    'y': test_pred
})

submission.to_csv('submission.csv', index=False)
