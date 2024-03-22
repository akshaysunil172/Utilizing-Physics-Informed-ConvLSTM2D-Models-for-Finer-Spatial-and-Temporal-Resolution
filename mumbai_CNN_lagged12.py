import xarray as xr
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import ConvLSTM2D, BatchNormalization, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import ConvLSTM2D, BatchNormalization, Flatten, Dense, Dropout



def load_and_normalize_data(file_path, variable_name):
    ds = xr.open_dataset(file_path)
    data = ds[variable_name].values
    data_normalized = (data - np.min(data)) / (np.max(data) - np.min(data))
    return data_normalized

def create_sequences(features, targets, time_steps):
    X, y = [], []
    for i in range(len(features) - time_steps - 6):  # Adjusted to lag by 2 values
        X.append(features[i:(i + time_steps)])
        y.append(targets[i + time_steps + 6].reshape(-1))  # Adjusted to predict 2 hours ahead
    return np.array(X), np.array(y)

 

def build_model(input_shape, learning_rate=0.0001):  # Added a learning_rate parameter with a default value
    model = Sequential([
        ConvLSTM2D(filters=256, kernel_size=(2, 2), activation='relu', input_shape=input_shape, return_sequences=True, padding='same'),
        BatchNormalization(),
        ConvLSTM2D(filters=128, kernel_size=(2, 2), activation='relu', return_sequences=False, padding='same'),
        BatchNormalization(),
        Flatten(),
        Dense(256, activation='relu'),
        Dropout(0.4),
        Dense(4, activation='linear')  # Adjusted for 4 outputs corresponding to the 2x2 grid
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), loss='mse')  # Specified learning rate here
    return model


# Specific data loading based on your provided paths and variables
base_path = 'E:/CNN/CNN/Mumbai_CNN/2011_22/'
feature_files = ['mumbai_pv_2011_22_500hpa.nc', 'mumbai_pv_2011_22_850hpa.nc', 'mumbai_rh_2011_22_250hpa.nc', 'mumbai_rh_2011_22_500hpa.nc', 'mumbai_rh_2011_22_850hpa.nc', 'mumbai_hcc_2011_22.nc', 'mumbai_tcc_2011_22.nc','mumbai_t_2011_22_250hpa.nc', 'mumbai_t_2011_22_500hpa.nc','mumbai_t_2011_22_850hpa.nc', 'mumbai_sp_2011_22.nc']
target_file = 'mumbai_tp_2011_22.nc'
variable_names = [ 'pv', 'pv', 'r', 'r', 'r', 'hcc', 'tcc','t','t','t', 'sp']
#
#,'mumbai_sp_2011_22'

features = np.stack([load_and_normalize_data(base_path + file, var) for file, var in zip(feature_files, variable_names)], axis=-1)
target = load_and_normalize_data(base_path + target_file, 'tp')

# Continue with sequence creation, model training, and result plotting as described in previous responses
time_steps = 24  # Adjust as per your sequence length
X, y = create_sequences(features, target, time_steps)

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, shuffle =False)

# Reshape for ConvLSTM input
X_train = X_train.reshape(X_train.shape[0], time_steps, features.shape[1], features.shape[2], features.shape[3])
X_test = X_test.reshape(X_test.shape[0], time_steps, features.shape[1], features.shape[2], features.shape[3])

# Build the model
model = build_model(input_shape=X_train.shape[1:], learning_rate=0.0001)

# Model training configuration
early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=1)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=0.00001, verbose=1)

# Train the model
history = model.fit(
    X_train, y_train,
    validation_split=0.15,
    epochs=50
    ,  # Adjust the number of epochs as needed
    batch_size=32,  # Adjust the batch size as needed
    callbacks=[ reduce_lr] #no early stopping
)

# Predictions
y_train_pred = model.predict(X_train).reshape(-1, 2, 2)  # Reshape predictions to match the 2x2 grid format
# Reshape predictions to match the 2x2 grid format
y_test_pred = model.predict(X_test).reshape(-1, 2, 2)



def plot_results(y_true, y_pred, set_type):
    num_samples = y_true.shape[0]
    for i in range(num_samples):  # Loop over all samples
        fig, axs = plt.subplots(2, 4, figsize=(20, 10))  # 2 rows, 4 columns for actual and predicted grids
        for j in range(4):  # Loop over the 4 grids
            grid_index = j // 2 * 2 + j % 2  # Calculate the index for the grid
            axs[0, j].imshow(y_true[i, grid_index].reshape(2, 2), cmap='hot', interpolation='nearest')
            axs[0, j].set_title(f'Actual Grid {j + 1}')
            axs[1, j].imshow(y_pred[i, grid_index].reshape(2, 2), cmap='hot', interpolation='nearest')
            axs[1, j].set_title(f'Predicted Grid {j + 1}')
        plt.suptitle(f'{set_type} Set Predictions for Sample {i + 1}')
        plt.savefig(f'{set_type.lower()}_predictions_sample_{i + 1}.png')
        plt.close()


# Function to plot and save training and validation loss
def plot_and_save_training_loss(history):
    plt.figure(figsize=(8, 6))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('training_and_validation_loss.png')
    plt.close()


def plot_columns_combined(actuals_df, predictions_df, set_type):
    """
    Plot actual and predicted values of each column in the same subplot for direct comparison.

    Parameters:
    - actuals_df: DataFrame containing the actual values.
    - predictions_df: DataFrame containing the predicted values.
    - set_type: String indicating the type of data set (e.g., "Training" or "Test").
    """
    fig, axs = plt.subplots(4, 1, figsize=(10, 20), sharex=True)

    for i in range(4):
        # Plot actuals and predictions in the same subplot
        axs[i].plot(actuals_df.iloc[:, i], label='Actual', marker='o', linestyle='-', markersize=4)
        axs[i].plot(predictions_df.iloc[:, i], label='Predicted', marker='x', linestyle='--', markersize=4)
        axs[i].set_title(f'{set_type} Set - Column {i+1}')
        axs[i].legend()

    plt.suptitle(f'{set_type} Set Actual vs. Predicted - Combined Columns')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()


# Load the actual and predicted data from CSV files
train_actuals_df = pd.read_csv('E:/CNN/CNN/Mumbai_CNN/2011_22/train_actuals_lag.csv')
train_predictions_df = pd.read_csv('E:/CNN/CNN/Mumbai_CNN/2011_22/train_predictions_lag.csv')
test_actuals_df = pd.read_csv('E:/CNN/CNN/Mumbai_CNN/2011_22/test_actuals_lag.csv')
test_predictions_df = pd.read_csv('E:/CNN/CNN/Mumbai_CNN/2011_22/test_predictions_lag.csv')

# Plot combined actual and predicted columns for the training and test sets
plot_columns_combined(train_actuals_df, train_predictions_df, "Training")
plot_columns_combined(test_actuals_df, test_predictions_df, "Test")

# Save the model and predictions (optional)
model.save('ConvLSTM_model_lag.h5')
pd.DataFrame(y_train_pred.reshape(y_train_pred.shape[0], -1)).to_csv(base_path + 'train_predictions_lag.csv', index=False)  # Flatten the 2x2 grids for saving
pd.DataFrame(y_train.reshape(y_train.shape[0], -1)).to_csv(base_path + 'train_actuals_lag.csv', index=False)  # Flatten the 2x2 grids for saving
pd.DataFrame(y_test_pred.reshape(y_test_pred.shape[0], -1)).to_csv(base_path + 'test_predictions_lag.csv', index=False)  # Flatten the 2x2 grids for saving
pd.DataFrame(y_test.reshape(y_test.shape[0], -1)).to_csv(base_path + 'test_actuals_lag.csv', index=False)  # Flatten the 2x2 grids for saving


import matplotlib.pyplot as plt
import pandas as pd

def plot_columns_combined(actuals_df, predictions_df, set_type):
    """
    Plot actual and predicted values of each column in the same subplot for direct comparison.

    Parameters:
    - actuals_df: DataFrame containing the actual values.
    - predictions_df: DataFrame containing the predicted values.
    - set_type: String indicating the type of data set (e.g., "Training" or "Test").
    """
    fig, axs = plt.subplots(4, 1, figsize=(10, 20), sharex=True)

    for i in range(4):
        # Plot actuals and predictions in the same subplot
        axs[i].plot(actuals_df.iloc[:, i], label='Actual', marker='o', linestyle='-', markersize=4)
        axs[i].plot(predictions_df.iloc[:, i], label='Predicted', marker='x', linestyle='--', markersize=4)
        axs[i].set_title(f'{set_type} Set - Column {i+1}')
        axs[i].legend()

    plt.suptitle(f'{set_type} Set Actual vs. Predicted - Combined Columns')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(f'{base_path}{set_type.lower()}_predictions_sample_{i + 1}.png')
    plt.close()

    plt.show()

# Load the actual and predicted data from CSV files
train_actuals_df = pd.read_csv('E:/CNN/CNN/Mumbai_CNN/2011_22/train_actuals_lag.csv')
train_predictions_df = pd.read_csv('E:/CNN/CNN/Mumbai_CNN/2011_22/train_predictions_lag.csv')
test_actuals_df = pd.read_csv('E:/CNN/CNN/Mumbai_CNN/2011_22/test_actuals_lag.csv')
test_predictions_df = pd.read_csv('E:/CNN/CNN/Mumbai_CNN/2011_22/test_predictions_lag.csv')


# Plot combined actual and predicted columns for the training and test sets
plot_columns_combined(train_actuals_df, train_predictions_df, "Training")
plot_columns_combined(test_actuals_df, test_predictions_df, "Test")


import matplotlib.pyplot as plt

def plot_training_loss(history):
    """
    Plot the training loss over epochs and save the plot to a file.

    Parameters:
    - history: A Keras History object returned by the fit method of a model.
    """
    plt.figure(figsize=(10, 8))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')  # Plotting validation loss as well
    plt.title('Training and Validation Loss Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig('training_and_validation_loss_plot.png')  # Save the plot to a file
    plt.show()


# Assuming 'history' is your Keras History object from training the model
plot_training_loss(history)

#plt.plot(history.history['val_loss'], label='Validation Loss')
