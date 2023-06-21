import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

!git clone https://github.com/AnalyticsKnight/yemoonsaBigdata
X_test = pd.read_csv("./yemoonsaBigdata/datasets/Part2/penguin_X_test.csv")
X_train = pd.read_csv("./yemoonsaBigdata/datasets/Part2/penguin_X_train.csv")
y_train = pd.read_csv("./yemoonsaBigdata/datasets/Part2/penguin_y_train.csv")
print(X_train.info())

train = pd.concat([X_train, y_train], axis=1)
print(train.loc[(train.sex.isna()) | (train.bill_length_mm.isna()) | (train.bill_depth_mm.isna()) | (train.flipper_length_mm.isna()) |(train.body_mass_g.isna())])

train = train.dropna()
train.reset_index(drop=True, inplace=True)

X_train = train[['species','island', 'sex', 'bill_length_mm',  'bill_depth_mm',  'flipper_length_mm']]
y_train = train[['body_mass_g']]

print(X_train.describe())

COL_DEL = []
COL_NUM = ['bill_length_mm',  'bill_depth_mm',  'flipper_length_mm']
COL_CAT = ['species','island', 'sex']
COL_Y = ['body_mass_g']

X = pd.concat([X_train, X_test])

from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder(handle_unknown = 'ignore')
ohe.fit(X[COL_CAT])
X_train_res = ohe.transform(X_train[COL_CAT])
X_test_res = ohe.transform(X_test[COL_CAT])

X_train_ohe = pd.DataFrame(X_train_res.todense(), columns = ohe.get_feature_names_out())
X_test_ohe = pd.DataFrame(X_test_res.todense(), columns = ohe.get_feature_names_out())
print(X_train_ohe)

X_train_fin = pd.concat([X_train[COL_NUM], X_train_ohe], axis=1)
X_test_fin = pd.concat([X_test[COL_NUM], X_test_ohe], axis=1)

from sklearn.model_selection import train_test_split
X_tr, X_val, y_tr, y_val = train_test_split(X_train_fin, y_train, test_size=0.3)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(X_tr[COL_NUM])
X_tr[COL_NUM] = scaler.transform(X_tr[COL_NUM])
X_val[COL_NUM] = scaler.transform(X_val[COL_NUM])
X_test_fin[COL_NUM] = scaler.transform(X_test_fin[COL_NUM])

from sklearn.linear_model import LinearRegression
modelLR = LinearRegression()
modelLR.fit(X_tr, y_tr)

y_val_pred = modelLR.predict(X_val)
print(y_val_pred)

from sklearn.metrics import mean_squared_error, r2_score
mse = mean_squared_error(y_val, y_val_pred)
rmse = mean_squared_error(y_val, y_val_pred, squared=False)
print('MSE: {0:.3f}, RMSE:{1:.3F}'.format(mse,rmse))

y_pred = modelLR.predict(X_test_fin)

pd.DataFrame({'body_mass_g': y_pred[:,0]}).to_csv('./yemoonsaBigdata/res/04000000.csv',index=False)
