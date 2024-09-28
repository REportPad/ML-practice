#1. EDA & Feature Engineering
import pandas as pd
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

#object type labeling
object_cols = test.select_dtypes(include=['object'])  # object 타입 열 선택
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
for col in object_cols:
  train[col] = label_encoder.fit_transform(train[col])
  test[col] = label_encoder.fit_transform(test[col])

#target labeling
label_encoder_target = LabelEncoder()
train['NObeyesdad'] = label_encoder_target.fit_transform(train['NObeyesdad'])

#train, test data split
from sklearn.model_selection import train_test_split
X = train.drop(['NObeyesdad','id'], axis=1)
y = train['NObeyesdad']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

import lightgbm as lgb
train_data = lgb.Dataset(X_train, label=y_train)
test_data = lgb.Dataset(X_test, label=y_test,reference=train_data)

# 모델 학습
import numpy as np
model = lgb.LGBMClassifier(
    objective='multiclass', 
    num_class=len(np.unique(y)),
    num_leaves=31,          # 트리의 리프 노드 수 (작을수록 과적합 방지)
    learning_rate=0.05,     # 학습률
    n_estimators=1000       # 부스팅 반복 횟수
    )
model.fit(X_train, y_train)

# 정확도 및 성능 평가
from sklearn.metrics import accuracy_score
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"Accuracy: {acc}")

#test data prediction
test_drop_id = test.drop('id', axis=1)
test_pred = model.predict(test_drop_id)
test_pred_original = label_encoder_target.inverse_transform(test_pred)

submission = pd.DataFrame({
    'id': test['id'],         # ID 열 추가
    'NObeyesdad': test_pred_original    # 예측 값 열 추가
})
submission.to_csv('submission_0921.csv', index=False)

##hypterparameter tuning
# LightGBM 모델 초기화
lgbm = lgb.LGBMClassifier(objective='multiclass', 
    num_class=len(np.unique(y)),
                          )

# 하이퍼파라미터 공간 설정
#최적 하이퍼파라미터: {'num_leaves': 20, 'n_estimators': 300, 'min_child_samples': 30, 'max_depth': 5, 'learning_rate': 0.05}
param_dist = {
    'num_leaves': [20, 31, 40],            # 리프 노드 수(31)
    'n_estimators': [100, 200, 300],     # 트리 개수 (100)
    'min_child_samples': [10,20,30],         # 최소 리프 노드 샘플 수 (20)
    'max_depth': [5, 10, 20],             # 트리 깊이 (-1)
    'learning_rate': [0.05, 0.1, 0.2],        # 학습률 (0.1)
}

# 4. RandomizedSearchCV 설정
from sklearn.model_selection import RandomizedSearchCV
random_search = RandomizedSearchCV(
    estimator=lgbm,
    param_distributions=param_dist,   # 파라미터 분포
    n_iter=10,                        # 시도할 파라미터 조합 수
    scoring='accuracy',               # 평가 지표
    cv=3,                             # 교차 검증 횟수
    verbose=1,                        # 출력 레벨
    random_state=42,                  # 랜덤 시드
    n_jobs=-1                         # 병렬 처리
)

# 5. RandomizedSearchCV 학습
random_search.fit(X_train, y_train)

# 6. 최적 하이퍼파라미터 출력
print(f"최적 하이퍼파라미터: {random_search.best_params_}")

# 7. 테스트 데이터로 최적 모델 평가
best_model = random_search.best_estimator_
y_pred = best_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"테스트 데이터 정확도: {accuracy}")


#test data prediction
test_drop_id = test.drop('id', axis=1)
test_pred = best_model.predict(test_drop_id)
test_pred_original = label_encoder_target.inverse_transform(test_pred)

submission = pd.DataFrame({
    'id': test['id'],         # ID 열 추가
    'NObeyesdad': test_pred_original    # 예측 값 열 추가
})
submission.to_csv('submission_best_0921.csv', index=False)

