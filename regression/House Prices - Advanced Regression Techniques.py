import pandas as pd
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

#결측값 내림차순 출력
missing_cols = train.isnull().sum()
missing_cols = missing_cols[missing_cols > 0].sort_values(ascending=False)
print(missing_cols)

#결측값 과반수 이상은 제거
train = train.drop(['PoolQC','MiscFeature','Alley','Fence','MasVnrType'], axis=1)
test = test.drop(['PoolQC','MiscFeature','Alley','Fence','MasVnrType'], axis=1)

#결측값 object type 처리
# object 타입의 열 출력
object_columns = train.select_dtypes(include=['object']).columns
print(object_columns)

# 결측값을 'Unknown'으로 대체
for col in object_columns:
    train[col].fillna('Unknown', inplace=True)
    test[col].fillna('Unknown', inplace=True)

# pandas의 get_dummies를 사용한 One-Hot Encoding
train = pd.get_dummies(train, columns=object_columns)
test = pd.get_dummies(test, columns=object_columns)

# train과 test의 열이 일치하도록 정렬 (필수 단계)
train, test = train.align(test, join='left', axis=1, fill_value=0)

#결측값 수치형 처리
# 수치형 변수의 결측값을 중앙값으로 대체
train.fillna(train.median(), inplace=True)
test.fillna(test.median(), inplace=True)
