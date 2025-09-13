import shap
import matplotlib.pyplot as plt
import xgboost as xgb

# Train a simple XGBoost model for feature importance analysis
X_train = X.reshape(X.shape[0], -1)  # Flatten sequences for XGBoost
model = xgb.XGBClassifier()
model.fit(X_train, np.argmax(y, axis=1))  # Train on categorical labels

# Explain model predictions using SHAP
explainer = shap.Explainer(model)
shap_values = explainer(X_train[:100])  # Take a subset for efficiency

# Plot Feature Importance
shap.summary_plot(shap_values, X_train[:100])
