import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from sklearn.linear_model import LinearRegression
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.model_selection import KFold, cross_val_score

# Define the features (X) and labels (y) as NumPy arrays
X = np.array([ 135, 135, 135, 135, 135, 130, 130, 130, 130, 130,
    130, 130, 130, 130, 125, 125, 123, 120, 120, 120,
    120, 120, 120, 120, 115, 110, 110, 110, 110, 110,
    110, 110, 110, 105, 105, 105, 105, 105, 105, 104,
    104, 104, 100, 100, 100, 100, 100, 100, 100, 100,
    100, 100, 100, 100, 100, 100, 100, 100, 100, 97,
    97, 95,95, 95, 95, 95, 95, 95, 95, 95, 95, 95, 95,
    90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90,
    90, 90, 90, 90, 90, 90, 90, 90, 90, 89, 87, 85,
    85, 85, 85, 85, 85, 80, 80]).reshape(-1, 1)
y = np.array([15750000, 13750000, 9500000, 8500000, 7250000, 18000000, 14600000, 12750000, 12750000, 12750000, 10200000, 8900000, 8000000, 3900000, 13000000, 8800000, 10500000, 18750000, 10500000, 8250000, 7990000, 6600000, 3950000, 3900000, 15400000, 23100000, 21950000, 12750000, 11500000, 11000000, 4000000, 3700000, 2900000, 14500000, 8000000, 3400000, 3380000, 3000000, 3000000, 4540000, 4540000, 4540000, 33000000, 19650000, 17000000, 10000000, 9850000, 7950000, 7500000, 6500000, 5500000, 4400000, 4300000, 4200000, 4000000, 3400000, 2800000, 2800000, 2750000, 27269000, 4550000, 5275000, 4640000, 4500000, 4450000, 3750000, 3650000, 3499000, 3400000, 2850000, 2785000, 2750000, 2700000, 20500000, 17800000, 14500000, 12000000, 12000000, 10000000, 8900000, 7950000, 7500000, 7000000, 6300000, 5555000, 5500000, 5500000, 5250000, 4750000, 3950000, 3450000, 3300000, 2785000, 2785000, 12000000, 6500000, 14500000, 12000000, 7000000, 6750000, 6500000, 1450000, 3900000, 3800000])





# Reshape X to a 2D array (required for sklearn)
X = X.reshape(-1, 1)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)



model=LinearRegression()
model.fit(X_train,y_train)
result=model.score(X_test,y_test)
print(result)

print(X.shape)

k_fold=KFold(10)
print(cross_val_score(model,X,y,cv=k_fold,n_jobs=1))

# Feature Scaling (Standardization)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize and train Gradient Boosting Regression model
gradient_boosting_reg = GradientBoostingRegressor()
gradient_boosting_reg.fit(X_train_scaled, y_train)

# Predict on the test set
y_pred = gradient_boosting_reg.predict(X_test_scaled)

# Calculate Mean Squared Error
mse = metrics.mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)

# Sort the data for plotting
sorted_indices = np.argsort(X_test[:, 0])
X_test_sorted = X_test[sorted_indices]
y_test_sorted = y_test[sorted_indices]
y_pred_sorted = y_pred[sorted_indices]

# Plot actual vs predicted
plt.scatter(y_test_sorted, y_pred_sorted)
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('Actual vs Predicted')
plt.show()

# Plot the regressor itself
plt.scatter(X_test_sorted, y_test_sorted, color='red', label='Actual Data')
plt.scatter(X_test_sorted, y_pred_sorted, color='blue', label='Predicted Data')
plt.plot(X_test_sorted, y_pred_sorted, color='green', label='Regressor')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Gradient Boosting Regression')
plt.legend()
plt.show()


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Check for outliers
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
sns.histplot(X, bins=20, kde=True, color='skyblue')
plt.title('Distribution of Features (X)')
plt.xlabel('X')
plt.ylabel('Frequency')

plt.subplot(1, 2, 2)
sns.histplot(y, bins=20, kde=True, color='salmon')
plt.title('Distribution of Labels (y)')
plt.xlabel('y')
plt.ylabel('Frequency')

plt.tight_layout()
plt.show()

# Check for missing values
missing_values = np.isnan(X).any() or np.isnan(y).any()
if missing_values:
    print("Missing values exist in the dataset.")
else:
    print("No missing values found in the dataset.")

# Check for duplicate entries
duplicate_entries = len(X) != len(set(tuple(row) for row in X)) or len(y) != len(set(y))
if duplicate_entries:
    print("Duplicate entries exist in the dataset.")
else:
    print("No duplicate entries found in the dataset.")

# Verify correctness of labels
# Since labels are provided directly, we assume they are correctly assigned to the corresponding features.

# Data quality checks
# Since the data provided is numerical and there are no missing values or duplicate entries,
# and the labels seem to be correctly assigned, we consider the data to be clean.

# Additional checks or specific domain knowledge validations could be performed depending on the context.

import matplotlib.pyplot as plt

# Plot the training data
plt.scatter(X_train, y_train, color='blue', label='Training Data')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Training Data')
plt.legend()
plt.show()

# Plot the testing data
plt.scatter(X_test, y_test, color='red', label='Testing Data')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Testing Data')
plt.legend()
plt.show()


import pandas as pd

# Create a DataFrame with X and y
df = pd.DataFrame({'X': X.flatten(), 'y': y})

# Calculate Pearson correlation coefficient
correlation_coefficient = df['X'].corr(df['y'])
print("Pearson correlation coefficient:", correlation_coefficient)


# Cross-validation
model_new = LinearRegression()
kfold_validation = KFold(15)
results = cross_val_score(model_new, X, y, cv=kfold_validation)
print(results)
print("Mean R^2 score:", np.mean(results))


a = X_train
b = y_train
c = X_test
d = y_pred

plt.scatter(X_train[:, 0], y_train, label='Training Data', color='blue')
plt.scatter(X_test[:, 0], y_pred, label='Predicted Data', color='red')
plt.xlabel('SquareMeter')
plt.ylabel('Price')
plt.title('Price vs. SquareMeter')
plt.legend()
plt.grid()
plt.show()

