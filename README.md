# machine-training
Testing

# Sistino na to kanoume me to pre-processing model me scikit - learn gia na petixoume to data pre-processing kathos kai to future engineering kathos to scikit - learn einai pio katallilo gia ayta ta tasks.
# Episis elega na to xwrisoume to project se dyo kommatia diladi to ena model tha einai gia to pre-processing kai to allo gia ta data pou ontos theloume opote fase 1 and fase 2 
# Episi mporoume na xrisimopoisoume to kaggle gia data testing?

# fase 1 model gia to data pre-processing and future engineering 
exo grapsei kapoia steps parakato an thes mporoume na ta akolouthisoume
Data Preprocessing:

1)Load the dataset and handle any missing values, outliers, or other data quality issues.
Encode categorical variables and normalize/standardize numerical features if needed.
Split the data into training and testing sets.
Feature Engineering:

2)Identify relevant features for your model. In your case, age, professional resume, and working experience are key features.
Extract information from the professional resume, such as skills or educational background, and convert it into numerical or categorical features.
Create new features that might provide additional insights (e.g., years of experience, education level).
Perform any necessary dimensionality reduction.
Model Training (First Model):

3)Choose a machine learning algorithm based on your problem (e.g., regression or classification).
Train the model on the preprocessed and engineered dataset using the training set.
Evaluate the model's performance on the testing set.
Make Predictions on the Original Dataset:

4)use the trained model to make predictions on the original dataset or any new data you want to process for the second model.
The predictions could represent, for example, a predicted level of expertise or suitability for a job.
Create a New Dataset for the Second Model:

Combine the predicted values with the original dataset or create a new dataset that includes the predicted values as additional features.
Include the original features (age, professional resume, working experience) along with the predicted values from the first model.




# fase 2 actuall model that we will use deep learning algorithms like pytorch or tensor flow and we will use the data 










import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt


data = pd.read_csv('vgsales.csv')

X = data.drop(columns=['Global_Sales'])
y = data['Global_Sales']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Preprocessing pipeline for numerical features
numerical_features = X.select_dtypes(include=['int64', 'float64']).columns
numerical_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='mean')),('scaler', StandardScaler())])


categorical_features = X.select_dtypes(include=['object']).columns
categorical_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='most_frequent')),('onehot', OneHotEncoder(handle_unknown='ignore'))])


preprocessor = ColumnTransformer(
    transformers=[('num', numerical_transformer, numerical_features),('cat', categorical_transformer, categorical_features)])

model = Pipeline(steps=[('preprocessor', preprocessor),('regressor', RandomForestRegressor())])

model.fit(X_train, y_train)
y_pred = model.predict(X_test)

plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, color='blue', alpha=0.5)
plt.title('Actual vs Predicted Global Sales')
plt.xlabel('Actual Global Sales')
plt.ylabel('Predicted Global Sales')
plt.show()

train_score = model.score(X_train, y_train)
test_score = model.score(X_test, y_test)

print(f"Training score: {train_score}")
print(f"Testing score: {test_score}")
