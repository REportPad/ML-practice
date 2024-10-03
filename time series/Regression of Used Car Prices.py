import pandas as pd
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

# print(len(train['fuel_type'].unique()))
# print(train['fuel_type'].unique())
# print(len(test['fuel_type'].unique()))
# print(test['fuel_type'].unique())

#Feature Engineering
#'fuel_type': '–'를 NaN으로 변경
train['fuel_type'].replace('–', np.nan, inplace=True)
test['fuel_type'].replace('–', np.nan, inplace=True)

#'model': Split the 'model' column in train and test datasets
train[['model_part_1', 'model_part_2', 'model_part_3']] = train['model'].str.split(' ', n=2, expand=True)
test[['model_part_1', 'model_part_2', 'model_part_3']] = test['model'].str.split(' ', n=2, expand=True)

#'engine':
import re
# Function to extract the engine features from the 'engine' column
def extract_engine_features(engine_str):
    # Initialize values for extraction
    horsepower = None
    engine_size = None
    cylinders = None
    
    # Extract horsepower (e.g., 320.0HP)
    hp_match = re.search(r'(\d+(\.\d+)?)HP', engine_str)
    if hp_match:
        horsepower = float(hp_match.group(1))
    
    # Extract engine size in liters (e.g., 5.3L)
    size_match = re.search(r'(\d+(\.\d+)?)L', engine_str)
    if size_match:
        engine_size = float(size_match.group(1))
    
    # Extract the number of cylinders (e.g., 8 Cylinder)
    cylinder_match = re.search(r'(\d+)\s*Cylinder', engine_str)
    if cylinder_match:
        cylinders = int(cylinder_match.group(1))
    
    return pd.Series([horsepower, engine_size, cylinders])

# Apply the function to the train and test datasets
train[['horsepower', 'engine_size', 'cylinders']] = train['engine'].apply(extract_engine_features)
test[['horsepower', 'engine_size', 'cylinders']] = test['engine'].apply(extract_engine_features)


#'transmission':
#Function to extract transmission-related features from the 'transmission' column
def extract_transmission_features(trans_str):
    # Initialize values for extraction
    trans_type = None
    num_gears = None
    special_feature = None
    
    # Extract transmission type (e.g., A/T, M/T)
    if "A/T" in trans_str:
        trans_type = "Automatic"
    elif "M/T" in trans_str:
        trans_type = "Manual"
    else:
        trans_type = "Other"
    
    # Extract number of gears (e.g., 7-Speed A/T -> 7)
    gears_match = re.search(r'(\d+)-Speed', trans_str)
    if gears_match:
        num_gears = int(gears_match.group(1))
    
    # Extract any special feature (e.g., "Dual Shift Mode")
    if "Dual Shift Mode" in trans_str:
        special_feature = "Dual Shift Mode"
    
    return pd.Series([trans_type, num_gears, special_feature])

# Apply the function to both train and test datasets
train[['trans_type', 'num_gears', 'special_feature']] = train['transmission'].apply(extract_transmission_features)
test[['trans_type', 'num_gears', 'special_feature']] = test['transmission'].apply(extract_transmission_features)

#removing columns
train = train.drop(['model','engine','transmission'], axis=1)
test = test.drop(['model','engine','transmission'], axis=1)

#object type labeling
object_cols = test.select_dtypes(include=['object']).columns  # object 타입 열 선택
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
for col in object_cols:
  train[col] = label_encoder.fit_transform(train[col])
  test[col] = label_encoder.fit_transform(test[col])

from sklearn.model_selection import train_test_split
X = train.drop(['price','id'], axis=1)
y = train['price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# LightGBM 모델 생성
import lightgbm as lgb
lgbm = lgb.LGBMRegressor()

# 하이퍼파라미터 공간 정의
import numpy as np
param_dist = {
    'num_leaves': [17, 31, 63, 127, 255],         # 리프 노드 수(31)
    'n_estimators': [100, 200, 400, 700, 1000],  # 트리 개수 (100)
    'min_child_samples': [10, 20, 40, 80, 100], # 최소 리프 노드 샘플 수 (20)
    'learning_rate': [0.03, 0.05, 0.1, 0.2, 0.3], # 학습률 (0.1, <= 0.3)
}

# RandomizedSearchCV 설정
from sklearn.model_selection import RandomizedSearchCV
random_search = RandomizedSearchCV(lgbm, 
                                   param_distributions=param_dist, 
                                   n_iter=10, 
                                   scoring='neg_mean_squared_log_error', 
                                   cv=3, 
                                   verbose=1, 
                                   random_state=42, 
                                   n_jobs=-1)
# 학습
random_search.fit(X_train, y_train)

# 최적의 하이퍼파라미터 출력
print("Best Hyperparameters:", random_search.best_params_)

# 최적의 모델로 예측
best_model = random_search.best_estimator_

# 검증 데이터에 대한 예측
y_pred = best_model.predict(X_test)

# RMSLE 계산
from sklearn.metrics import mean_squared_log_error
rmsle = np.sqrt(mean_squared_log_error(y_test, y_pred))
print("Validation RMSLE:", rmsle)

#test data prediction
test_drop_id = test.drop('id', axis=1)
test_pred = best_model.predict(test_drop_id)
# test_pred_original = label_encoder_target.inverse_transform(test_pred)

submission = pd.DataFrame({
    'id': test['id'],         # ID 열 추가
    'price': test_pred    # 예측 값 열 추가
})
submission.to_csv('submission_1003_best_0.csv', index=False)
