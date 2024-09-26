#1. EDA & Feature Engineering
import pandas as pd

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

import lightgbm as lgb
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import roc_auc_score

# Feature와 Target 설정
X = train.drop(['target','ID_code'], axis=1)
y = train['target']

# 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# LightGBM 데이터셋 생성
lgb_train = lgb.Dataset(X_train, y_train)
lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train)

# 하이퍼파라미터 설정
params = {
    'objective': 'binary',
    'metric': 'auc',
    'boosting_type': 'gbdt',
    'learning_rate': 0.05,#?
    'num_leaves': 31,#?
    'feature_fraction': 0.9,#?
    'bagging_fraction': 0.8,#?
    'bagging_freq': 5,#?
    'verbose': 0#?
}

# 모델 학습
gbm = lgb.train(params,
                lgb_train,
                num_boost_round=1000,
                )

# 예측 및 AUC 평가
y_pred = gbm.predict(X_test, num_iteration=gbm.best_iteration)
auc_score = roc_auc_score(y_test, y_pred)
print('AUC: {:.4f}'.format(auc_score))

##
test_ids = test['ID_code']
X_test = test.drop('ID_code', axis=1)# Feature 추출

test_pred = gbm.predict(X_test, num_iteration=gbm.best_iteration)
submission = pd.DataFrame({'ID_code': test_ids, 'target': test_pred})
submission.to_csv('submission.csv', index=False)
