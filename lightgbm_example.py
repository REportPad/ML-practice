import pandas as pd
from sklearn.model_selection import train_test_split
import lightgbm as lgb

df = pd.read_csv('student-por.csv', sep=';') #sep=';' 정렬된 데이터는 생략 가능
df.head()

y = train.iloc[:, -1] #target, target이 여러개면 숫자 변경
X = train.iloc[:, :-1] #data
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
submission['price'] = lgb_pred
