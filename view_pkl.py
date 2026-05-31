import joblib
import pickle
from sklearn.linear_model import LinearRegression
import numpy as np

# Load your saved model
model = joblib.load("models/delivery_time_model.pkl")

# 1. What type is it?
print(type(model))
# → <class 'sklearn.linear_model._base.LinearRegression'>

# 2. The learned weights (coefficients)
print("Weights:", model.coef_)
# → [2.85, 1.42, 5.31, 3.67]

# 3. The bias (intercept)
print("Intercept:", model.intercept_)
# → 8.24

# 4. Feature names (if trained with DataFrame)
print("Features:", model.feature_names_in_)
# → ['distance_km', 'items_count', 'is_peak_hour', 'traffic_level']

# 5. All internal attributes
print(dir(model))

# 6. Make a prediction manually (prove it's just math)
x = [5, 3, 1, 2]  # 5km, 3 items, peak hour, traffic=2
manual = sum(w * v for w, v in zip(model.coef_, x)) + model.intercept_
sklearn_pred = model.predict([x])[0]
print(f"Manual calc: {manual:.1f}")
print(f"sklearn pred: {sklearn_pred:.1f}")
# → Both: 40.4 — proving it's just multiplication + addition