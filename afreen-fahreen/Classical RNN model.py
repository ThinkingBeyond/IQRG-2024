#To upload the file from desktop
from google.colab import files 
uploaded = files.upload()

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN
import matplotlib.pyplot as plt

# Upload and read the data
data = pd.read_csv('co2_mm_mlo.csv')
print(data.head())

# Extract relevant columns and handle missing values
data_relevant = data[['year', 'month', 'average']]
data_relevant['average'] = data_relevant['average'].replace(-9.99, None).ffill()

# Normalize the data
scaler = MinMaxScaler()
data_relevant['average_scaled'] = scaler.fit_transform(data_relevant[['average']])

# Create sequences for RNN
def create_sequences(data, sequence_length):
    xs, ys = [], []
    for i in range(len(data) - sequence_length):
        x = data[i:i+sequence_length]
        y = data[i+sequence_length]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

sequence_length = 12
X, y = create_sequences(data_relevant['average_scaled'].values, sequence_length)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

# Define the RNN model
model = Sequential([
    SimpleRNN(50, activation='relu', input_shape=(sequence_length, 1)),
    Dense(1)
])
model.compile(optimizer='adam', loss='mse')

# Train the model
history = model.fit(X_train, y_train, epochs=50, validation_data=(X_test, y_test))

# Evaluate the model
loss = model.evaluate(X_test, y_test)
print(f'Test Loss: {loss}')

# Make predictions for the next 12 months
last_sequence = data_relevant['average_scaled'].values[-sequence_length:]
input_sequence = last_sequence.reshape((1, sequence_length, 1))

predictions = []
for _ in range(12):
    next_value = model.predict(input_sequence)[0][0]
    predictions.append(next_value)
    input_sequence = np.append(input_sequence[:, 1:, :], [[[next_value]]], axis=1)

predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))

# Plot predictions
plt.figure(figsize=(12, 6))
plt.plot(data_relevant['year'] + data_relevant['month']/12, data_relevant['average'], label='Actual')
plt.plot(np.arange(2023, 2024, 1/12), predictions, label='Predicted')
plt.xlabel('Year')
plt.ylabel('CO2 Levels')
plt.title('CO2 Level Predictions for 2023 using RNN')
plt.legend()
plt.show()

print('Predicted CO2 levels for 2023 using RNN:', predictions.flatten())

# Create input sequence from the last 12 months in the training data
last_sequence = data_relevant['average_scaled'].values[-sequence_length:]
input_sequence = last_sequence.reshape((1, sequence_length, 1))

# Predict the next 12 months
classical_predictions = []
for _ in range(12):
    next_value = model.predict(input_sequence)[0][0]
    classical_predictions.append(next_value)
    input_sequence = np.append(input_sequence[:, 1:, :], [[[next_value]]], axis=1)

# Convert predictions back to original scale
classical_predictions = scaler.inverse_transform(np.array(classical_predictions).reshape(-1, 1))

# Print predictions
print('Predicted CO2 levels for 2023:', classical_predictions.flatten())

# Import necessary library for plotting
import matplotlib.pyplot as plt

# Create a list of months for 2023
months = [f'2023-{i:02d}' for i in range(1, 13)]

# Plot the predictions
plt.figure(figsize=(10, 6))
plt.plot(months, classical_predictions, marker='o', linestyle='-', color='b')
plt.title('Predicted Monthly Average CO2 Levels for 2023 using RNN')
plt.xlabel('Month')
plt.ylabel('CO2 Levels (ppm)')
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
