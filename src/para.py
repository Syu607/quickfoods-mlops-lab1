import joblib

MODEL_PATH = "models/delivery_time_model.pkl"

# 1. Load the model artifact using joblib (since your script used joblib.dump)
model = joblib.load(MODEL_PATH)

# 2. Count the parameters (coefficients + intercept)
num_coefficients = model.coef_.size
num_intercept = model.intercept_.size
total_params = num_coefficients + num_intercept

# 3. Print the results
print(f"Number of learned feature weights (coefficients): {num_coefficients}")
print(f"Number of learned bias values (intercept): {num_intercept}")
print(f"Total parameters learned by this model: {total_params}")

# Optional: View the actual learned parameters
print("\n--- Learned Values ---")
print(f"Weights for your 4 features: {model.coef_}")
print(f"Baseline intercept value: {model.intercept_:.4f}")