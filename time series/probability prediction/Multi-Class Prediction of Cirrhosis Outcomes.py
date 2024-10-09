# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

# Load the datasets
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')
sample_submission_df = pd.read_csv('sample_submission.csv')

# Preprocess the target variable (encode the target labels)
le = LabelEncoder()
train_df['Status_encoded'] = le.fit_transform(train_df['Status'])

# Separate features (X) and target (y) from the training set
X = train_df.drop(columns=['id', 'Status', 'Status_encoded'])
y = train_df['Status_encoded']

# Prepare the test set features
X_test = test_df.drop(columns=['id'])

# Identify categorical and numerical columns
categorical_cols = ['Drug', 'Sex', 'Ascites', 'Hepatomegaly', 'Spiders', 'Edema']
numerical_cols = list(set(X.columns) - set(categorical_cols))

# Define preprocessing pipelines for numerical and categorical data
numerical_transformer = SimpleImputer(strategy='median')
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Combine the transformers into a preprocessor
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])

# Create a pipeline that first preprocesses the data and then trains the model
model = Pipeline(steps=[('preprocessor', preprocessor),
                        ('classifier', RandomForestClassifier(random_state=42, class_weight='balanced'))])

# Train the model
model.fit(X, y)

# Predict probabilities for each class in the test set
y_proba = model.predict_proba(X_test)

# Retrieve class names from the LabelEncoder
class_names = le.inverse_transform(model.named_steps['classifier'].classes_)

# Prepare the submission file using the predicted probabilities
submission_df = sample_submission_df.copy()
for i, class_name in enumerate(class_names):
    submission_df[f'Status_{class_name}'] = y_proba[:, i]

# Save the final submission file
submission_path = 'submission_with_probabilities.csv'
submission_df.to_csv(submission_path, index=False)
