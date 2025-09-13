import numpy as np
import pandas as pd
import tensorflow as tf
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
from tensorflow.keras.models import load_model
from sklearn.preprocessing import label_binarize
import seaborn as sns
from sklearn.model_selection import train_test_split

# Load the pretrained CNN-GRU model
df = pd.read_pickle('Datasets/col_combine.pkl')
print(df.columns)
model = load_model("Model/best_weather_cnn_gru_model_tuned4.h5")

X = df.drop(columns=["Weather","IMU_Covariance_Combined"])
y = df["Weather"]



# Define the Kalman filter class
class AdaptiveKalmanFilter:
    def __init__(self):
        self.Q = np.eye(4) * 0.1  # Process noise covariance
        self.R = np.eye(2) * 1.0  # Measurement noise covariance
        self.P = np.eye(4) * 1.0  # Estimate error covariance
        self.x = np.zeros((4, 1))  # State estimate

    def update_parameters(self, weather_label):
        if weather_label == 1:  # Fog
            self.R *= 1.5
            self.Q *= 1.2
        elif weather_label == 2:  # Rain
            self.R *= 2.0
            self.Q *= 1.5
        elif weather_label == 3:  # Snow
            self.R *= 2.5
            self.Q *= 1.8
        elif weather_label == 4:  # Night
            self.R *= 1.3
            self.Q *= 1.1
        else:  # Sunny (default)
            self.R = np.eye(2) * 1.0
            self.Q = np.eye(4) * 0.1

    def predict(self, u):
        self.x = self.x + u
        self.P = self.P + self.Q
        return self.x

    def update(self, z):
        y = z - self.x[:2]
        S = self.P[:2, :2] + self.R
        K = np.dot(self.P, np.linalg.inv(S))
        self.x += np.dot(K, y)
        self.P -= np.dot(K, self.P)
        return self.x

# Simulate test data (replace with actual test data)
X_train, X_test, y_train, y_true = train_test_split(X, y, test_size=0.9, random_state=18) # Example image size
X_test = X_test.values.reshape(X_test.shape[0], X_test.shape[1], 1)
#y_test = y_true.values.reshape(y_true.shape[0], y_true.shape[1], 1)
y_pred = model.predict(X_test)
y_pred_labels = np.argmax(y_pred, axis=1)

# Evaluation Metrics
accuracy = accuracy_score(y_true, y_pred_labels)
report = classification_report(y_true, y_pred_labels, target_names=["Sunny", "Fog", "Rain", "Snow", "Night"])
conf_matrix = confusion_matrix(y_true, y_pred_labels)

# Binarize the labels for ROC computation
y_true_bin = label_binarize(y_true, classes=[0, 1, 2, 3, 4])
y_pred_bin = y_pred

# Plot Confusion Matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=["Sunny", "Fog", "Rain", "Snow", "Night"], yticklabels=["Sunny", "Fog", "Rain", "Snow", "Night"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

# Plot ROC Curves
plt.figure(figsize=(10, 6))
for i in range(5):
    fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_pred_bin[:, i])
    plt.plot(fpr, tpr, label=f"Class {i} (AUC = {auc(fpr, tpr):.2f})")

plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curves")
plt.legend()
plt.show()

print(f"Accuracy: {accuracy:.4f}")
print("Classification Report:")
print(report)
