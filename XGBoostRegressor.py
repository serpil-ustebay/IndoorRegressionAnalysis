import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import optuna
import joblib
from xgboost import XGBRegressor


def read_Data(rssi, coordinates):
    RSSI_train = pd.read_csv(rssi)
    x_y_train = pd.read_csv(coordinates)

    # Separate features and targets
    features = RSSI_train.values
    coordinates = x_y_train.iloc[:, :2].values

    return features, coordinates


# Load data from CSV file

file_train = "D:/indoor dataset/TUJI/RSS_training.csv"
coordinates_train = "D:/indoor dataset/TUJI/Coordinates_training.csv"

file_test = "D:/indoor dataset/TUJI/RSS_testing.csv"
coordinates_test = "D:/indoor dataset/TUJI/Coordinates_testing.csv"

x_train, y_train = read_Data(file_train, coordinates_train)

# Split the data into training and valid
x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=0.2, random_state=42)

x_test, y_test = read_Data(file_test, coordinates_test)


def objective(trial):
    # Define the hyperparameters to tune
    n_estimators = trial.suggest_int('n_estimators', 50, 300)
    max_depth = trial.suggest_int('max_depth', 2, 20)
    learning_rate = trial.suggest_float('learning_rate', 1e-3, 1e-1)
    gamma = trial.suggest_float('gamma', 1e-3, 1e1)
    min_child_weight = trial.suggest_int('min_child_weight', 1, 10)
    subsample = trial.suggest_float('subsample', 0.5, 1.0)
    colsample_bytree = trial.suggest_float('colsample_bytree', 0.5, 1.0)

    # Initialize the XGBoost Regressor with the hyperparameters
    model = XGBRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        gamma=gamma,
        min_child_weight=min_child_weight,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        random_state=42
    )

    # Train the model
    model.fit(x_train, y_train)

    # Make predictions on the validation set
    y_pred = model.predict(x_valid)

    # Calculate the overall Mean Absolute Error
    overall_mae = mean_absolute_error(y_valid, y_pred)

    return overall_mae


# Create a study object and optimize the objective function
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=500)

# Print the best hyperparameters
print(f"Best hyperparameters: {study.best_params}")

# Train the model with the best hyperparameters
best_params = study.best_params
best_model = XGBRegressor(
    n_estimators=best_params['n_estimators'],
    max_depth=best_params['max_depth'],
    learning_rate=best_params['learning_rate'],
    gamma=best_params['gamma'],
    min_child_weight=best_params['min_child_weight'],
    subsample=best_params['subsample'],
    colsample_bytree=best_params['colsample_bytree'],
    random_state=42
)
best_model.fit(x_train, y_train)

# Save the trained model to a file
model_filename = 'best_xgboost_model.pkl'
joblib.dump(best_model, model_filename)
print(f"Model saved to {model_filename}")

# Load the model from the file
loaded_model = joblib.load(model_filename)
print(f"Model loaded from {model_filename}")

# Make predictions on the test set using the loaded model
y_pred = loaded_model.predict(x_test)
y_pred = best_model.predict(x_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred, multioutput='raw_values')
print(f"Mean Squared Error for each output: {mse}")

# Calculate the overall Mean Squared Error
overall_mse = mean_squared_error(y_test, y_pred)
print(f"Overall Mean Squared Error: {overall_mse}")

print('####################################3')

# Calculate distances
from scipy.spatial.distance import euclidean

distances = np.array([euclidean(pred, true) for pred, true in zip(y_pred, y_test)])

# Calculate mean distance
mean_distance = np.mean(distances)
print(f"Mean distance: {mean_distance}")

# Calculate min distance
min_distance = np.min(distances)
print(f"Min distance: {min_distance}")

# Calculate max distance
max_distance = np.max(distances)
print(f"Max distance: {max_distance}")

# Calculate std distance
std_distance = np.std(distances)
print(f"std distance: {std_distance}")

# Calculate metrics for each feature
mae_feature_1 = mean_absolute_error(y_test[:, 0], y_pred[:, 0])
mae_feature_2 = mean_absolute_error(y_test[:, 1], y_pred[:, 1])

mse_feature_1 = mean_squared_error(y_test[:, 0], y_pred[:, 0])
mse_feature_2 = mean_squared_error(y_test[:, 1], y_pred[:, 1])

rmse_feature_1 = np.sqrt(mse_feature_1)
rmse_feature_2 = np.sqrt(mse_feature_2)

r2_feature_1 = r2_score(y_test[:, 0], y_pred[:, 0])
r2_feature_2 = r2_score(y_test[:, 1], y_pred[:, 1])

print(f"Feature 1 - MAE: {mae_feature_1}, MSE: {mse_feature_1}, RMSE: {rmse_feature_1}, R2: {r2_feature_1}")
print(f"Feature 2 - MAE: {mae_feature_2}, MSE: {mse_feature_2}, RMSE: {rmse_feature_2}, R2: {r2_feature_2}")

# Calculate combined metrics
mae_combined = mean_absolute_error(y_test, y_pred, multioutput='uniform_average')
mse_combined = mean_squared_error(y_test, y_pred, multioutput='uniform_average')
rmse_combined = np.sqrt(mse_combined)
r2_combined = r2_score(y_test, y_pred, multioutput='uniform_average')

print(f"Combined - MAE: {mae_combined}, MSE: {mse_combined}, RMSE: {rmse_combined}, R2: {r2_combined}")

# Plot optimization results
import optuna.visualization as vis

vis.plot_optimization_history(study).show()
vis.plot_param_importances(study).show()
