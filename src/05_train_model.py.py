import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, BatchNormalization, GRU, Dense, Flatten, Dropout
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from tensorflow.keras.optimizers import Adam
from pykalman import KalmanFilter
import matplotlib.pyplot as plt
import time

# Load data
file_path = "Datasets/col_combine.pkl"
data = pd.read_pickle(file_path)

# Extract features and labels
X = data.drop(columns=["Weather", "IMU_Covariance_Combined"])
y = data["Weather"]

# Apply SMOTE for class balancing
smote = SMOTE()
X_resampled, y_resampled = smote.fit_resample(X, y)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# Reshape for CNN input
X_train = X_train.values.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test = X_test.values.reshape(X_test.shape[0], X_test.shape[1], 1)

# Learning rate schedule (Cyclical Learning Rate)
lr_schedule = ExponentialDecay(initial_learning_rate=0.001, decay_steps=1000, decay_rate=0.9)

# Define optimized CNN-GRU Model
model = Sequential([
    Conv1D(filters=64, kernel_size=3, activation='relu', padding='same', kernel_regularizer=l2(0.001), input_shape=(X_train.shape[1], 1)),
    BatchNormalization(),
    Conv1D(filters=128, kernel_size=3, activation='relu', padding='same', kernel_regularizer=l2(0.001)),
    BatchNormalization(),
    Dropout(0.4),

    Conv1D(filters=64, kernel_size=3, activation='relu', padding='same', kernel_regularizer=l2(0.001)),
    BatchNormalization(),

    GRU(128, return_sequences=True, kernel_regularizer=l2(0.001)),
    GRU(64, return_sequences=False, kernel_regularizer=l2(0.001)),
    Dropout(0.4),

    Dense(32, activation='relu', kernel_regularizer=l2(0.001)),
    Dense(len(np.unique(y)), activation='softmax')
])

# Compile the model with CLR optimizer
optimizer = Adam(learning_rate=lr_schedule)
model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

start_time = time.time()
# Train the model
history = model.fit(X_train, y_train, epochs=50, batch_size=64, validation_data=(X_test, y_test))
end_time = time.time()

training_time = end_time - start_time
val_loss = history.history['val_loss']
convergence_epoch = next((epoch for epoch in range(1, len(val_loss)) if abs(val_loss[epoch] - val_loss[epoch - 1]) < 0.01), len(val_loss))
final_train_accuracy = history.history['accuracy'][-1]
final_val_accuracy = history.history['val_accuracy'][-1]

print(f"Training Time: {training_time:.2f} seconds")
print(f"Convergence Epoch: {convergence_epoch}")
print(f"Value Loss: {val_loss}")
print(f"Final Training Accuracy: {final_train_accuracy * 100:.2f}%")
print(f"Final Validation Accuracy: {final_val_accuracy * 100:.2f}%")

# Save the model
model.save("Model/weather_cnn_gru_model_tuned4.h5")

# Adaptive Kalman Filter with smooth Q/R transition
kf = KalmanFilter(initial_state_mean=0, n_dim_obs=1)
Q_base, R_base = 1.0, 1.0

def adaptive_kf(motion, weather, prev_Q, prev_R):
    motion_factor = 1.2 if motion > 1 else 0.8
    weather_factor = 1.5 if weather in [1, 2] else 1.0
    
    Q_new = prev_Q * 0.9 + (Q_base * motion_factor * weather_factor) * 0.1  # Smooth update
    R_new = prev_R * 0.9 + (R_base / (motion_factor * weather_factor)) * 0.1
    
    kf.transition_covariance = np.array([[Q_new]])
    kf.observation_covariance = np.array([[R_new]])
    return Q_new, R_new

# Evaluate Adaptive Kalman Filter performance
weather_preds = model.predict(X_test)
weather_classes = np.argmax(weather_preds, axis=1)

adaptive_Q, adaptive_R = [], []
prev_Q, prev_R = Q_base, R_base

for motion, weather in zip(X_test[:, 0], weather_classes):
    prev_Q, prev_R = adaptive_kf(motion, weather, prev_Q, prev_R)
    adaptive_Q.append(prev_Q)
    adaptive_R.append(prev_R)

# Plot results
plt.figure(figsize=(12, 10))

# Training Performance
plt.subplot(2, 1, 1)
plt.plot(history.history['accuracy'], label="Train Accuracy", color="red")
plt.plot(history.history['val_accuracy'], label="Test Accuracy", color="orange")
plt.ylabel("Accuracy (%)")
plt.xlabel("Epochs")
plt.legend()
plt.title("(a) Training and Testing Accuracy")
plt.grid()

plt.twinx()
plt.plot(history.history['loss'], label="Train Loss", linestyle="--", color="blue")
plt.plot(history.history['val_loss'], label="Test Loss", linestyle="--", color="green")
plt.ylabel("Loss")
plt.legend(loc='upper right')

# Adaptive Q and R Values
plt.subplot(2, 1, 2)
plt.plot(adaptive_Q, label="Adaptive Q", linestyle="--", marker="o", color="blue")
plt.plot(adaptive_R, label="Adaptive R", linestyle="-.", marker="s", color="green")
plt.xlabel("Index")
plt.ylabel("Values")
plt.title("(b) Smoothed Adaptive Q and R")
plt.legend()
plt.grid()

plt.tight_layout()
plt.savefig("Model/adaptive_q_r_tuned4.png")
plt.show()
