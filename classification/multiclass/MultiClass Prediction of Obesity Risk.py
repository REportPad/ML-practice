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

# 모델 파라미터 설정
#[LightGBM] [Warning] No further splits with positive gain, best gain: -inf 발생할 경우,
#min_data_in_leaf 값을 낮추기
#num_leaves 값을 증가시키기
params = {
    'objective': 'multiclass',
    'num_class': len(np.unique(y)),  # 클래스 개수
    'metric': 'multi_logloss',
    'boosting_type': 'gbdt',
    'learning_rate': 0.05,
    'num_leaves': 31,
    'max_depth': -1,
    'random_state': 42
}

# 모델 학습
model = lgb.train(params, train_data, valid_sets=[test_data], num_boost_round=1000)

import numpy as np
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)

# 정확도 및 성능 평가
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, y_pred_classes)
print(f"Accuracy: {accuracy}")

from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred_classes))

#test data prediction
test_drop = test.drop('id', axis=1)
test_pred_prob = model.predict(test_drop)
test_pred = np.argmax(test_pred_prob, axis=1)
test_pred_original = label_encoder_target.inverse_transform(test_pred)

#create submission file
submission = pd.DataFrame({
    'id': test['id'],         # ID 열 추가
    'NObeyesdad': test_pred_original    # 예측 값 열 추가
})
submission.to_csv('submission_0921.csv', index=False)
