import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import pandas as pd
import numpy as np
from scipy.spatial.distance import euclidean
import optuna
from optuna.samplers import TPESampler
import optuna.visualization as vis
from tensorflow.keras.utils import plot_model

def plot_training_history(history, save_path=None):
    import matplotlib.pyplot as plt
    # Plot training & validation MAE values
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.plot(history.history['mae'])
    plt.plot(history.history['val_mae'])
    plt.title('Model MAE')
    plt.ylabel('MAE')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')

    # Plot training & validation loss values
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')

    if save_path:
        plt.savefig(save_path, dpi=1000)  # Set a high DPI value
    plt.show()
def read_Data(rssi, coordinates):
    RSSI_train = pd.read_csv(rssi)
    x_y_train = pd.read_csv(coordinates)

    # Separate features and targets
    features = RSSI_train.values
    coordinates = x_y_train.iloc[:, :2].values

    return features, coordinates

n_trials=30
epoch_tune=100
epoch_best=300

# Load data from CSV file
file_train = r"D:\indoor dataset\TUJI\RSS_training.csv"
coordinates_train = r"D:\indoor dataset\TUJI\Coordinates_training.csv"

file_test = r"D:\indoor dataset\TUJI\RSS_testing.csv"
coordinates_test = r"D:\indoor dataset\TUJI\Coordinates_testing.csv"

x_train, y_train = read_Data(file_train, coordinates_train)
x_test, y_test = read_Data(file_test, coordinates_test)

# Scale the data
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)


def objective(trial):
    # Define hyperparameters
    num_layers = trial.suggest_int('num_layers', 1, 10)
    units = [trial.suggest_int(f'units_l{i}', 8, 256) for i in range(num_layers)]
    dropout_rate = trial.suggest_float('dropout_rate', 0.1, 0.9)
    batch_size = trial.suggest_categorical('batch_size', [8, 16, 32, 64, 128])
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-1)

    # Suggest activation function
    activation_function = trial.suggest_categorical('activation_function',
                                                    ['relu', 'sigmoid', 'selu', 'softmax', 'exponential', 'linear'])

    # Suggest optimizer
    optimizer_name = trial.suggest_categorical('optimizer',
                                               ['adam', 'sgd', 'rmsprop', 'adagrad', 'adadelta', 'adamax', 'nadam',
                                                'ftrl'])
    # Select optimizer based on suggestion
    if optimizer_name == 'adam':
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    elif optimizer_name == 'sgd':
        optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate, momentum=0.9)
    elif optimizer_name == 'rmsprop':
        optimizer = tf.keras.optimizers.RMSprop(learning_rate=learning_rate, rho=0.9)
    elif optimizer_name == 'adagrad':
        optimizer = tf.keras.optimizers.Adagrad(learning_rate=learning_rate)
    elif optimizer_name == 'adadelta':
        optimizer = tf.keras.optimizers.Adadelta(learning_rate=learning_rate, rho=0.95)
    elif optimizer_name == 'adamax':
        optimizer = tf.keras.optimizers.Adamax(learning_rate=learning_rate)
    elif optimizer_name == 'nadam':
        optimizer = tf.keras.optimizers.Nadam(learning_rate=learning_rate)
    elif optimizer_name == 'ftrl':
        optimizer = tf.keras.optimizers.Ftrl(learning_rate=learning_rate)

    # Build the model
    model = Sequential()
    model.add(Dense(units[0], input_dim=x_train.shape[1], activation=activation_function))
    for u in units[1:]:
        model.add(Dense(u, activation=activation_function))
        model.add(tf.keras.layers.Dropout(dropout_rate))
    model.add(Dense(2))  # 2 output values (x and y)

    # Compile the model
    model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])

    # Train the model
    model.fit(
        x_train, y_train,
        epochs=epoch_tune,
        batch_size=batch_size,
        validation_split=0.2,
        verbose=0
    )

    # Evaluate the model
    loss, mae = model.evaluate(x_test, y_test, verbose=0)

    return mae  # Return MAE as the optimization objective


# Create an Optuna study with TPE sampler and optimize the objective function
study = optuna.create_study(sampler=optuna.samplers.TPESampler(), direction='minimize')
study.optimize(objective, n_trials=n_trials)

# Print the best hyperparameters
print("Best hyperparameters:")
print(study.best_params)

# Train the final model with the best hyperparameters
best_params = study.best_params

# Rebuild the model with the best hyperparameters
# Select optimizer based on suggestion
if best_params['optimizer'] == 'adam':
        optimizer = tf.keras.optimizers.Adam(learning_rate=best_params['learning_rate'])
elif best_params['optimizer'] == 'sgd':
        optimizer = tf.keras.optimizers.SGD(learning_rate=best_params['learning_rate'], momentum=0.9)
elif best_params['optimizer'] == 'rmsprop':
        optimizer = tf.keras.optimizers.RMSprop(learning_rate=best_params['learning_rate'], rho=0.9)
elif best_params['optimizer'] == 'adagrad':
        optimizer = tf.keras.optimizers.Adagrad(learning_rate=best_params['learning_rate'])
elif best_params['optimizer'] == 'adadelta':
        optimizer = tf.keras.optimizers.Adadelta(learning_rate=best_params['learning_rate'], rho=0.95)
elif best_params['optimizer'] == 'adamax':
        optimizer = tf.keras.optimizers.Adamax(learning_rate=best_params['learning_rate'])
elif best_params['optimizer'] == 'nadam':
        optimizer = tf.keras.optimizers.Nadam(learning_rate=best_params['learning_rate'])
elif best_params['optimizer'] == 'ftrl':
        optimizer = tf.keras.optimizers.Ftrl(learning_rate=best_params['learning_rate'])


model = Sequential()
model.add(Dense(best_params['units_l0'], input_dim=x_train.shape[1], activation=best_params['activation_function']))
for i in range(1, best_params['num_layers']):
    model.add(Dense(best_params[f'units_l{i}'], activation=best_params['activation_function']))
    model.add(tf.keras.layers.Dropout(best_params['dropout_rate']))
model.add(Dense(2))  # 2 output values (x and y)

# Compile the model
model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])

import os
import pydot
import tensorflow as tf
from tensorflow.keras.utils import plot_model
os.environ["PATH"] += os.pathsep + r"C:\Program Files\Graphviz\bin"
# Plot the model architecture
plot_model(model, to_file='plots/model_architecture.png', show_shapes=True, show_layer_names=True)


# Train the model with best hyperparameters
history = model.fit(
    x_train, y_train,
    epochs=epoch_best,
    batch_size=best_params['batch_size'],
    validation_split=0.2
)

##PLOT history
# Assuming 'history' is the variable holding the training history
plot_training_history(history, save_path='plots/training_history.png')

# Evaluate the model
loss, mae = model.evaluate(x_test, y_test)
print(f"Test Loss: {loss}, Test MAE: {mae}")

# Make predictions
predictions = model.predict(x_test)
df = pd.DataFrame(predictions)
df.to_csv("predictions.csv")

# Calculate distances
distances = np.array([euclidean(pred, true) for pred, true in zip(predictions, y_test)])

'''
# Print some example distances
print("Example distances between predicted and actual coordinates:")
for i in range(5):  # Print the first 5 distances
    print(f"Example {i + 1}: {distances[i]}")
'''
# Calculate mean distance
mean_distance = np.mean(distances)
print(f"Mean distance: {mean_distance}")

# Calculate mean distance
min_distance = np.min(distances)
print(f"Min distance: {min_distance}")

# Calculate mean distance
max_distance = np.max(distances)
print(f"Max distance: {max_distance}")

# Plot optimization results
#vis.plot_optimization_history(study).show()
#vis.plot_param_importances(study).show()
