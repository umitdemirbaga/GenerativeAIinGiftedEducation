# Calculation the MAE, MSE, R-squared, Training Time, and Testing Time for each ML algorithm

# importing all the necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score, precision_score, recall_score, f1_score
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense
from sklearn.preprocessing import StandardScaler
import time

# loading the preprocessed data
X = df.drop("moralSensitivity", axis=1)
y = df["moralSensitivity"]

# splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# defining the number of splits for cross-validation
num_splits = 5  

# defining a function to evaluate models with cross-validation
def evaluate_model_with_cv(model, model_name, X, y):
    kfold = KFold(n_splits=num_splits, shuffle=True, random_state=42)
    mae_scores = []
    mse_scores = []
    r2_scores = []

    training_start_time = time.time()  # Start measuring training time

    for train_index, test_index in kfold.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

        mae = mean_absolute_error(y_test, predictions)
        mse = mean_squared_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)

        mae_scores.append(mae)
        mse_scores.append(mse)
        r2_scores.append(r2)

    training_end_time = time.time()  # Stop measuring training time

    mae_mean = np.mean(mae_scores)
    mse_mean = np.mean(mse_scores)
    r2_mean = np.mean(r2_scores)

    testing_start_time = time.time()  # Start measuring testing time
    predictions = model.predict(X_test)
    testing_end_time = time.time()  # Stop measuring testing time

    print(f"{model_name} Metrics (Cross-Validation):")
    print(f"MAE: {mae_mean:.2f}")
    print(f"MSE: {mse_mean:.2f}")
    print(f"R-squared: {r2_mean:.2f}")

    print(f"Training Time: {training_end_time - training_start_time:.2f} seconds")
    print(f"Testing Time: {testing_end_time - testing_start_time:.2f} seconds")

    # discretising the target variable
    def discretize_moral_sensitivity(moral_sensitivity, threshold_low, threshold_high):
        if moral_sensitivity < threshold_low:
            return "low"
        elif threshold_low <= moral_sensitivity < threshold_high:
            return "medium"
        else:
            return "high"

    threshold_low = 3.0
    threshold_high = 4.0

    y_test_classes = y_test.apply(lambda x: discretize_moral_sensitivity(x, threshold_low, threshold_high))
    predictions_classes = pd.Series(predictions).apply(lambda x: discretize_moral_sensitivity(x, threshold_low, threshold_high))

    # calculation classification metrics
    accuracy = accuracy_score(y_test_classes, predictions_classes)
    precision = precision_score(y_test_classes, predictions_classes, average='weighted')
    recall = recall_score(y_test_classes, predictions_classes, average='weighted')
    f1 = f1_score(y_test_classes, predictions_classes, average='weighted')

    print()
    print(f"Accuracy: {accuracy:.2f}")
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F1-Score: {f1:.2f}")

    print()
    print(f"*********************************")

# Linear Regression
lr_model = LinearRegression()
evaluate_model_with_cv(lr_model, "Linear Regression", X, y)

# Decision Trees
dt_model = DecisionTreeRegressor()
evaluate_model_with_cv(dt_model, "Decision Trees", X, y)

# Random Forest
rf_model = RandomForestRegressor()
evaluate_model_with_cv(rf_model, "Random Forest", X, y)

# Support Vector Regression
svr_model = SVR()
evaluate_model_with_cv(svr_model, "Support Vector Regression", X, y)

# XGBoost
xgb_model = XGBRegressor()
evaluate_model_with_cv(xgb_model, "XGBoost", X, y)