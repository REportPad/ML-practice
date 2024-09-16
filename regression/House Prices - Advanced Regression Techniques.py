import pandas as pd
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

#object type labeling
object_cols = train.select_dtypes(include=['object'])  # object 타입 열 선택
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
for col in object_cols:
  train[col] = label_encoder.fit_transform(train[col])

object_cols = test.select_dtypes(include=['object'])  # object 타입 열 선택
for col in object_cols:
  test[col] = label_encoder.fit_transform(test[col])

#data split
from sklearn.model_selection import train_test_split
X = train.drop(['SalePrice', 'Id'], axis=1)
y = train['SalePrice']
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)

import lightgbm as lgb
train_data = lgb.Dataset(X_train, label=y_train)
valid_data = lgb.Dataset(X_valid, label=y_valid)
lgb_regressor = lgb.LGBMRegressor(
    num_leaves=31,          # 트리의 리프 노드 수 (작을수록 과적합 방지)
    learning_rate=0.05,     # 학습률
    n_estimators=1000       # 부스팅 반복 횟수
)
lgb_regressor.fit(X_train, y_train, eval_set=[(X_valid, y_valid)])

from sklearn.metrics import mean_squared_error
y_pred = lgb_regressor.predict(X_valid)# 테스트 데이터에 대한 예측
mse = mean_squared_error(y_valid, y_pred)# 모델 성능 평가 (평균 제곱 오차)
print(f'Mean Squared Error: {mse}')

#test data prediction
X_test = test.drop('Id', axis=1)
test_pred = lgb_regressor.predict(X_test)

submission = pd.DataFrame({
    'Id': test['Id'],         # ID 열 추가
    'SalePrice': test_pred    # 예측 값 열 추가
})
submission.to_csv('submission_0916.csv', index=False)
