import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import optuna
import joblib


def read_Data(rssi, coordinates):
    RSSI_train = pd.read_csv(rssi)
    x_y_train = pd.read_csv(coordinates)

    # Separate features and targets
    features = RSSI_train.values
    coordinates = x_y_train.iloc[:, :2].values

    return features, coordinates




# Load data from CSV file
file_train = "/content/drive/MyDrive/Indoor-TUJI/Dataset/TUJI/RSS_training.csv"
coordinates_train = "/content/drive/MyDrive/Indoor-TUJI/Dataset/TUJI/Coordinates_training.csv"

file_test = "/content/drive/MyDrive/Indoor-TUJI/Dataset/TUJI/RSS_testing.csv"
coordinates_test = "/content/drive/MyDrive/Indoor-TUJI/Dataset/TUJI/Coordinates_testing.csv"

x_train, y_train = read_Data(file_train, coordinates_train)

# Split the data into training and valid
x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=0.2, random_state=42)


x_test, y_test = read_Data(file_test, coordinates_test)



def objective(trial):
    # Define the hyperparameters to tune
    n_estimators = trial.suggest_int('n_estimators', 50, 300)
    max_depth = trial.suggest_int('max_depth', 2, 20)
    min_samples_split = trial.suggest_int('min_samples_split', 2, 10)
    min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 10)
    max_features = trial.suggest_categorical('max_features', ['sqrt', 'log2', None])

    # Initialize the Random Forest Regressor with the hyperparameters
    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        random_state=42
    )

    # Train the model
    model.fit(x_train, y_train)

    # Make predictions on the test set
    y_pred = model.predict(x_valid)

    # Calculate the overall Mean absulate Error
    overall_mae = mean_absolute_error(y_valid,y_pred)

    return overall_mae

# Create a study object and optimize the objective function
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=500)



# Print the best hyperparameters
print(f"Best hyperparameters: {study.best_params}")

# Train the model with the best hyperparameters
best_params = study.best_params
best_model = RandomForestRegressor(
    n_estimators=best_params['n_estimators'],
    max_depth=best_params['max_depth'],
    min_samples_split=best_params['min_samples_split'],
    min_samples_leaf=best_params['min_samples_leaf'],
    max_features=best_params['max_features'],
    random_state=42
)
best_model.fit(x_train, y_train)

# Save the trained model to a file
model_filename = '/content/drive/MyDrive/Indoor-TUJI/best_random_forest_model.pkl'
joblib.dump(best_model, model_filename)
print(f"Model saved to {model_filename}")

# Load the model from the file
loaded_model = joblib.load(model_filename)
print(f"Model loaded from {model_filename}")

# Make predictions on the test set using the loaded model
y_pred = loaded_model.predict(x_test)

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

# Calculate mean distance
min_distance = np.min(distances)
print(f"Min distance: {min_distance}")

# Calculate mean distance
max_distance = np.max(distances)
print(f"Max distance: {max_distance}")

# Calculate std distance
max_distance = np.std(distances)
print(f"std distance: {max_distance}")





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