# machine-training
Testing


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

data = pd.read_csv('vgsales.csv')  # Updated the dataset filename

X = data.drop(columns=['Global_Sales'])  # Updated target column name
y = data['Global_Sales']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
numerical_features = X.select_dtypes(include=['int64', 'float64']).columns
numerical_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='mean')),('scaler', StandardScaler())])

categorical_features = X.select_dtypes(include=['object']).columns
categorical_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='most_frequent')),('onehot', OneHotEncoder(handle_unknown='ignore'))])


preprocessor = ColumnTransformer(
    transformers=[('num', numerical_transformer, numerical_features), ('cat', categorical_transformer, categorical_features) ])

model = Pipeline(steps=[('preprocessor', preprocessor),('regressor', RandomForestRegressor())])

model.fit(X_train, y_train)

train_score = model.score(X_train, y_train)
test_score = model.score(X_test, y_test)

print(f"Training score: {train_score}")
print(f"Testing score: {test_score}")

test_score = model.score(X_test, y_test) tran_scoe = model.score(X_train, y_train)
print(f"Testing score: {test_score}") print(f"Training score: {train_score}")
