# house-price-prediction
prediction about house prices based on certain parameters

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor,
AdaBoostRegressor, ExtraTreesRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor

# Load the dataset
df = pd.read_csv("housepriceData(Sathvik).csv")

# Normalize the price

df['price'] = df['price'] / 100000
print(df.head())

# Plotting

plt.figure(figsize=(10, 6))
plt.scatter(df['bedrooms'], df['price'], c=df['bathrooms'], cmap='viridis')


plt.xlabel('Number of Bedrooms')
plt.ylabel('Price')
plt.title('Price vs. Bedrooms and Bathrooms')
plt.colorbar(label='Number of Bathrooms')
plt.show()
plt.figure(figsize=(12, 6))
df.boxplot(column='price', by='location', vert=False)
plt.xlabel('Price')
plt.ylabel('Location')
plt.title('Distribution of Apartment Prices by Location')
plt.suptitle('')
plt.show()
df['bedrooms'].plot(kind='hist', bins=20, title='Bedrooms')
plt.gca().spines[['top', 'right']].set_visible(False)
plt.show()
df['floors'].plot(kind='hist', bins=20, title='Floors')
plt.gca().spines[['top', 'right']].set_visible(False)
plt.show()
# Prepare features (X) and target (y)
X = df.drop(columns=['price'])
y = df['price']
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=2)

# Define the preprocessing step
step1 = ColumnTransformer(transformers=[
 ('col_tnf', OneHotEncoder(sparse=False, drop='first'), [0])
], remainder='passthrough')
# Define the model
step2 = LinearRegression()
# Create the pipeline
pipe = Pipeline([
 ('step1', step1),
 ('step2', step2)
])
# Fit the pipeline on the training data
pipe.fit(X_train, y_train)
# Predict on the test data
y_pred = pipe.predict(X_test)
# Print evaluation metrics
print('R2 score:', r2_score(y_test, y_pred))
print('MAE:', mean_absolute_error(y_test, y_pred))
# Define input variables for a new prediction
location = "Jubilee Hills"
floor = 1
bedrooms = 3
bathrooms = 2
