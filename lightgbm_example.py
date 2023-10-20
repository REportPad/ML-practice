import pandas as pd
from sklearn.model_selection import train_test_split
import lightgbm as lgb

train_data = pd.read_csv('titanic/train.csv') #sep=';' 정렬된 데이터는 생략 가능
#train_data.head()
test_data = pd.read_csv('titanic/test.csv') #sep=';' 정렬된 데이터는 생략 가능
#train_data.head()


#열 순서를 변경하고 싶으면
#train_data.columns 입력하여 열 이름을 얻은 후
#train_data = train_data[[~]] 형태로 열 이름을 재배치함

y = train_data.iloc[:, -1] #target, target이 여러개면 숫자 변경
X = train_data.iloc[:, :-1] #data
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

lgb_reg = lgb.LGBMRegressor(random_state=42)
# lgb_clf = lgb.LGBMClassifier(n_estimators=400) #분류일 경우
cat_columns = X_train.columns[X_train.dtypes==object].tolist()

#object type 데이터 category로 변경
for c in cat_columns:
    X_train[c] = X_train[c].astype('category')
    X_test[c] = X_test[c].astype('category')

lgb_reg.fit(X_train, y_train)
lgb_pred = lgb_reg.predict(X_test)
submission = pd.read_csv('titanic/submission.csv')
submission['price'] = lgb_pred
