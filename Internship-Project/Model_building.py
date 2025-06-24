import pickle
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pandas as pd
X = pd.read_csv("X_cleaned.csv")
y = pd.read_csv("y_cleaned.csv")
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# 1. Linear Regression Model
model_lr = LinearRegression()
model_lr.fit(X_train, y_train)
pred_test = model_lr.predict(X_test)

mae_lr = mean_absolute_error(y_test, pred_test)
mse_lr = mean_squared_error(y_test, pred_test)
r2_lr = r2_score(y_test, pred_test)

print("\nLinear Regression:")
print("MAE:", mae_lr)
print("MSE:", mse_lr)
print("R² Score:", r2_lr)

# 2. Random Forest Model
model_rf = RandomForestRegressor(random_state=42)
model_rf.fit(X_train, y_train)
pred = model_rf.predict(X_test)

mae_rf = mean_absolute_error(y_test, pred)
mse_rf = mean_squared_error(y_test, pred)
r2_rf = r2_score(y_test, pred)

print("\nRandom Forest:")
print("MAE:", mae_rf)
print("MSE:", mse_rf)
print("R² Score:", r2_rf)

# 3. XGBoost Model
model_xgb = XGBRegressor(random_state=42, verbosity=0)
model_xgb.fit(X_train, y_train)
pred3 = model_xgb.predict(X_test)

mae_xgb = mean_absolute_error(y_test, pred3)
mse_xgb = mean_squared_error(y_test, pred3)
r2_xgb = r2_score(y_test, pred3)

print("\nXGBoost:")
print("MAE:", mae_xgb)
print("MSE:", mse_xgb)
print("R² Score:", r2_xgb)

# 4. Model Comparison
print("\n----- Model Comparison (Based on Test Set) -----")
print(f"{'Model':<20} {'MAE':<15} {'MSE':<15} {'R² Score':<15}")
print(f"{'Linear Regression':<20} {mae_lr:<15.4f} {mse_lr:<15.4f} {r2_lr:<15.4f}")
print(f"{'Random Forest':<20} {mae_rf:<15.4f} {mse_rf:<15.4f} {r2_rf:<15.4f}")
print(f"{'XGBoost':<20} {mae_xgb:<15.4f} {mse_xgb:<15.4f} {r2_xgb:<15.4f}")

# Saving the best model — Random Forest
with open("model.pkl", "wb") as f:
    pickle.dump(model_rf, f)

# Saving the feature order (important for prediction)
with open("feature_order.pkl", "wb") as f:
    pickle.dump(list(X.columns), f)

print("✅ Random Forest model and feature order saved successfully.")

# Convert actual and predicted productivity into binary categories
# Let's say: 1 if productivity >= 0.75, else 0

y_test_class = (y_test >= 0.75).astype(int)
y_pred_class = (pred >= 0.75).astype(int)
from sklearn.metrics import fbeta_score

f2 = fbeta_score(y_test_class, y_pred_class, beta=2)
print("F2 Score:", f2)
