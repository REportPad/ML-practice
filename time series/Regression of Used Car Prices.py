import pandas as pd
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

#object type labeling
object_cols = test.select_dtypes(include=['object']).columns  # object 타입 열 선택
# print(object_cols)
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
for col in object_cols:
  train[col] = label_encoder.fit_transform(train[col])
  test[col] = label_encoder.fit_transform(test[col])

from sklearn.model_selection import train_test_split
X = train.drop(['price','id'], axis=1)
y = train['price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# LightGBM 모델 생성
import lightgbm as lgb
lgbm = lgb.LGBMRegressor()

# 하이퍼파라미터 공간 정의
import numpy as np
param_dist = {
    'num_leaves': np.arange(20, 150, 10),
    'learning_rate': [0.01, 0.05, 0.1, 0.15, 0.2],
    'n_estimators': np.arange(100, 1000, 100),
    'max_depth': np.arange(3, 15, 1)
}

# RandomizedSearchCV 설정
from sklearn.model_selection import RandomizedSearchCV
random_search = RandomizedSearchCV(lgbm, 
                                   param_distributions=param_dist, 
                                   n_iter=20, 
                                   scoring='neg_mean_squared_log_error', 
                                   cv=5, 
                                   verbose=1, 
                                   random_state=42, 
                                   n_jobs=-1)

# 학습
random_search.fit(X_train, y_train)

# 최적의 하이퍼파라미터 출력
print("Best Hyperparameters:", random_search.best_params_)

# 최적의 모델로 예측
best_model = random_search.best_estimator_

# 검증 데이터에 대한 예측
y_pred = best_model.predict(X_test)

# RMSLE 계산
from sklearn.metrics import mean_squared_log_error
rmsle = np.sqrt(mean_squared_log_error(y_test, y_pred))
print("Validation RMSLE:", rmsle)

#test data prediction
test_drop_id = test.drop('id', axis=1)
test_pred = best_model.predict(test_drop_id)
# test_pred_original = label_encoder_target.inverse_transform(test_pred)

submission = pd.DataFrame({
    'id': test['id'],         # ID 열 추가
    'price': test_pred    # 예측 값 열 추가
})
submission.to_csv('submission_1003_best_0.csv', index=False)
