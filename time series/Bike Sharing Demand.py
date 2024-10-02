import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import mean_squared_log_error
import lightgbm as lgb

# 데이터 로드
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

# 날짜 데이터를 datetime 형식으로 변환
train['datetime'] = pd.to_datetime(train['datetime'])
train['year'] = train['datetime'].dt.year
train['month'] = train['datetime'].dt.month
train['day'] = train['datetime'].dt.day
train['hour'] = train['datetime'].dt.hour
train['dayofweek'] = train['datetime'].dt.dayofweek

# 불필요한 열 제거 (datetime)
X = train.drop(['count', 'datetime', 'casual', 'registered'], axis=1)
y = train['count']

# 학습 데이터와 테스트 데이터 분리
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)

# LightGBM 모델 생성
lgbm = lgb.LGBMRegressor()

# 하이퍼파라미터 공간 정의
param_dist = {
    'num_leaves': np.arange(20, 150, 10),
    'learning_rate': [0.01, 0.05, 0.1, 0.15, 0.2],
    'n_estimators': np.arange(100, 1000, 100),
    'max_depth': np.arange(3, 15, 1)
}

# RandomizedSearchCV 설정
random_search = RandomizedSearchCV(lgbm, param_distributions=param_dist, 
                                   n_iter=20, scoring='neg_mean_squared_log_error', 
                                   cv=5, verbose=1, random_state=42, n_jobs=-1)

# 학습
random_search.fit(X_train, y_train)

# 최적의 하이퍼파라미터 출력
print("Best Hyperparameters:", random_search.best_params_)

# 최적의 모델로 예측
best_model = random_search.best_estimator_

# 검증 데이터에 대한 예측
y_pred = best_model.predict(X_valid)

# RMSLE 계산
rmsle = np.sqrt(mean_squared_log_error(y_valid, y_pred))
print("Validation RMSLE:", rmsle)

# 테스트 데이터 전처리 (학습 데이터와 동일하게)
test['datetime'] = pd.to_datetime(test['datetime'])
test['year'] = test['datetime'].dt.year
test['month'] = test['datetime'].dt.month
test['day'] = test['datetime'].dt.day
test['hour'] = test['datetime'].dt.hour
test['dayofweek'] = test['datetime'].dt.dayofweek

X_test = test.drop(['datetime'], axis=1)

# 예측
predictions = best_model.predict(X_test)

# 제출 파일 생성
submission = pd.read_csv('sampleSubmission.csv')
submission['count'] = predictions
submission.to_csv('submission.csv', index=False)
