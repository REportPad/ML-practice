#1. File read
import pandas as pd
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
# print(train.info())
# print(train.describe())

# 1) 결측값 확인
missing_values = train.isnull().sum()
missing_values_sorted = missing_values.sort_values(ascending=False)
print(missing_values_sorted)

missing_values = test.isnull().sum()
missing_values_sorted = missing_values.sort_values(ascending=False)
print(missing_values_sorted)

## 결측값 처리
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()# 라벨 인코더 객체 생성
train['Sex'] = label_encoder.fit_transform(train['Sex'])
train['Embarked'] = label_encoder.fit_transform(train['Embarked'])

# 불필요한 'Ticket', 'Name' 등 제거
train = train.drop(['Ticket', 'Name', 'Cabin'], axis=1)

# Feature와 Target 설정
from sklearn.model_selection import train_test_split
X = train.drop('Survived', axis=1)
y = train['Survived']
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)

#2. 모델 선택 및 학습
import xgboost as xgb
xgb_model = xgb.XGBClassifier(n_estimators=100, learning_rate=0.05, max_depth=4, random_state=42)# XGBoost 모델 학습
xgb_model.fit(X_train, y_train)
y_pred = xgb_model.predict(X_valid)

from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_valid, y_pred)
print(f'검증 데이터 정확도: {accuracy * 100:.2f}%')

# 3. 예측
X_test = test.drop('PassengerId', axis=1)
predictions = xgb_model.predict(X_test)

# 4. 제출 파일 생성
submission = pd.DataFrame({
    "PassengerId": test['PassengerId'],
    "Survived": predictions
})
submission.to_csv('submission.csv', index=False)
