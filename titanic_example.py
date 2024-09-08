#1. EDA & Feature Engineering
import pandas as pd

train = pd.read_csv('sample_data/train.csv')
test = pd.read_csv('sample_data/test.csv')

print(train.info())
print(test.info())
# print(train.describe())

# 1) 결측값 확인
print(train.isnull().sum())
print(test.isnull().sum())
# 결측값 채우기
# 결측치 처리 예시 (Age와 Embarked)
train_data['Age'].fillna(train_data['Age'].median(), inplace=True)
train_data['Embarked'].fillna(train_data['Embarked'].mode()[0], inplace=True)

# 'Sex'와 'Embarked' 범주형 변수 인코딩 (Label Encoding)
train_data['Sex'] = train_data['Sex'].map({'male': 0, 'female': 1}).astype(int)
train_data['Embarked'] = train_data['Embarked'].map({'S': 0, 'C': 1, 'Q': 2}).astype(int)

# 'Cabin'의 결측치는 "Unknown"으로 처리하거나 새로운 변수로 변환
train_data['Cabin'] = train_data['Cabin'].fillna('Unknown')
train_data['Has_Cabin'] = train_data['Cabin'].apply(lambda x: 0 if x == 'Unknown' else 1)

# 불필요한 'Ticket', 'Name' 등 제거
train_data = train_data.drop(['Ticket', 'Name', 'Cabin'], axis=1)

# Feature와 Target 설정
X = train_data.drop(['Survived', 'PassengerId'], axis=1)
y = train_data['Survived']

# 학습 데이터와 검증 데이터로 분리
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

#2. 모델 선택 및 학습
import xgboost as xgb
from sklearn.metrics import accuracy_score

# XGBoost 모델 생성
xgb_model = xgb.XGBClassifier(
    n_estimators=960,      # 트리 개수
    learning_rate=0.01,#0.05     # 학습률
    max_depth=3,            # 트리의 최대 깊이
    random_state=42
)

# 모델 학습
xgb_model.fit(X_train, y_train,
              eval_set=[(X_val, y_val)],    # 검증 데이터 설정
              verbose=False)

y_pred = xgb_model.predict(X_val) # 검증 데이터 예측

# 성능 평가
accuracy = accuracy_score(y_val, y_pred)
print(f"Validation Accuracy: {accuracy:.4f}")

# 3. 예측 및 제출 파일 생성
# 테스트 데이터 전처리 (학습 데이터와 동일한 방식으로 처리)
test_data['Age'].fillna(test_data['Age'].median(), inplace=True)
test_data['Fare'].fillna(test_data['Fare'].median(), inplace=True)
test_data['Sex'] = test_data['Sex'].map({'male': 0, 'female': 1}).astype(int)
test_data['Embarked'] = test_data['Embarked'].map({'S': 0, 'C': 1, 'Q': 2}).astype(int)
test_data['Cabin'] = test_data['Cabin'].fillna('Unknown')
test_data['Has_Cabin'] = test_data['Cabin'].apply(lambda x: 0 if x == 'Unknown' else 1)
test_data = test_data.drop(['Ticket', 'Name', 'Cabin'], axis=1)

passenger_id = test_data['PassengerId'] # PassengerId를 따로 저장 (최종 제출을 위해)
X_test = test_data.drop('PassengerId', axis=1)# Feature 추출
test_pred = xgb_model.predict(X_test)# 최종 예측

# 결과를 DataFrame으로 저장
submission = pd.DataFrame({
    'PassengerId': passenger_id,
    'Survived': test_pred
})

submission.to_csv('submission.csv', index=False)# 결과 저장
