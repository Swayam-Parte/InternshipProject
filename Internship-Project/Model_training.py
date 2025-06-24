# Importing libraries
import pandas as pd
import seaborn as sns
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pickle

# Reading CSv file
df = pd.read_csv(r"D:\Work\garments_worker_productivity.csv")
print(df.head())

# Correlation Analysis through heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.savefig("correlation_heatmap.png")

# Descriptive analysis through pandas library features
print(df.describe())
print(df.info())  # To see datatypes and null counts
print(df['department'].value_counts())  # Category distribution

# Checking for Null Values
print(df.isnull().sum())

# Handling Date & Department Columns
df['department'] = df['department'].str.strip()
df = df.drop(['date'], axis=1)

# Handling Categorical Values
df = pd.get_dummies(df, drop_first=True)

# Drop rows with any NaN values
df.dropna(inplace=True)

# Splitting Data into Train and Test Sets
X = df.drop('actual_productivity', axis=1)
y = df['actual_productivity']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Save cleaned features and labels
X.to_csv("X_cleaned.csv", index=False)
y.to_csv("y_cleaned.csv", index=False)

