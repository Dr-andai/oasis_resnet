import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error

# ---------------------------------------------------
# Load your results

df = pd.read_csv("../data/inference_results_2.csv")
true_ages = df['age'].values
predicted_ages = df['predicted_age'].values

# Brain-Age-Gap
bag = predicted_ages - true_ages

# Metrics
mae = mean_absolute_error(true_ages, predicted_ages)
rmse = np.sqrt(mean_squared_error(true_ages, predicted_ages))
corr = np.corrcoef(true_ages, predicted_ages)[0,1]

print(f"MAE: {mae:.3f}")
print(f"RMSE: {rmse:.3f}")
print(f"Correlation: {corr:.3f}")

# ---------------------------------------------------
# Plot: BAG Histogram
plt.figure()
plt.hist(bag, bins=10)
plt.title("Brain-Age-Gap Distribution")
plt.xlabel("Predicted − Actual Age")
plt.ylabel("Frequency")
plt.show()

# ---------------------------------------------------
# Plot: True vs Predicted Scatter
plt.figure()
plt.scatter(true_ages, predicted_ages)
plt.title("True Age vs Predicted Age")
plt.xlabel("True Age")
plt.ylabel("Predicted Age")
plt.plot([true_ages.min(), true_ages.max()],
         [true_ages.min(), true_ages.max()])
plt.show()

# ---------------------------------------------------
# Plot: MAE / RMSE vs True Age
# per-sample errors
abs_errors = np.abs(predicted_ages - true_ages)

plt.figure()
plt.scatter(true_ages, abs_errors)
plt.title("Absolute Error vs True Age")
plt.xlabel("True Age")
plt.ylabel("Absolute Error")
plt.show()
