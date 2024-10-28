# Import Libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import LabelEncoder

# Load Data
data = pd.read_csv("dataset.csv")  # Replace with your actual file path
headers = ["symboling", "normalized-losses", "make", 
           "fuel-type", "aspiration","num-of-doors",
           "body-style","drive-wheels", "engine-location",
           "wheel-base","length", "width","height", "curb-weight",
           "engine-type","num-of-cylinders", "engine-size", 
           "fuel-system","bore","stroke", "compression-ratio",
           "horsepower", "peak-rpm","city-mpg","highway-mpg","price"]

data.columns=headers
# Data Preprocessing
# Replace "?" with NaN and drop rows with missing 'price'
data.replace("?", np.nan, inplace=True)
data.dropna(subset=['price'], inplace=True)

# Convert 'price' to numeric
data['price'] = pd.to_numeric(data['price'], errors='coerce')

# Drop rows with any missing values (for simplicity)
data.dropna(inplace=True)

# Selecting features for price prediction
features = [
    "symboling", "normalized-losses", "make", "fuel-type", "aspiration",
    "num-of-doors", "body-style", "drive-wheels", "engine-location",
    "wheel-base", "length", "width", "height", "curb-weight",
    "engine-type", "num-of-cylinders", "engine-size",
    "fuel-system", "bore", "stroke", "compression-ratio",
    "horsepower", "peak-rpm", "city-mpg", "highway-mpg"
]

# Handle categorical variables by filling missing values and encoding
categorical_features = [
    "make", "fuel-type", "aspiration", "num-of-doors",
    "body-style", "drive-wheels", "engine-location", "engine-type",
    "num-of-cylinders", "fuel-system"
]

# Fill missing categorical values
for feature in categorical_features:
    data[feature].fillna('unknown', inplace=True)

# Encode categorical variables
encoder = LabelEncoder()
for feature in categorical_features:
    data[feature] = encoder.fit_transform(data[feature])

# Splitting data into features (X) and target (y)
X = data[features]
y = data['price']

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model Building and Training
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Model Prediction
y_pred = model.predict(X_test)

# Model Evaluation
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Absolute Error: {mae}")
print(f"R-squared Score: {r2}")

# Feature Importance
feature_importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
print("Feature Importances:\n", feature_importances)

# Visualizing Feature Importances (Optional)
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(12, 6))
sns.barplot(x=feature_importances, y=feature_importances.index)
plt.title('Feature Importance in Predicting Car Price')
plt.xlabel('Importance Score')
plt.ylabel('Features')
plt.show()
