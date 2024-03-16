import numpy as np
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_val_score

# Define the features (X) and labels (y) as NumPy arrays
Z = ["Tomtom", "Mueyyitzade", "Etiler", "Etiler", "Gayrettepe",
      "Haciahmet", "Üniversite", "Gayrettepe", "Hacimimi",
     "Üniversite","Tahtakale","Cihannüma","Tahtakale","Tahtakale","Firuzköy","Tahtakale",
     "Tahtakale","Tahtakale","Tahtakale","Denizkösler","Merkez"]

X = np.array([130, 70, 100, 100, 110, 120, 130, 155, 
             85, 105, 130, 130, 130, 130, 89, 72,
             75, 70, 71,120,80,130,120,55,
             85,123])
y = np.array([3300000, 4200000, 4250000, 10500000, 9950000, 6600000, 18000000, 4650000,
            14500000, 14500000, 12750000, 12750000, 12750000, 8900000, 12000000, 4250000,
            14000000, 11000000, 4300000,3900000,3900000,3900000,3950000,1450000,
            1450000,10500000])

# Reshape X to a 2D array (required for sklearn)
X = X.reshape(-1, 1)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature Scaling (Standardization)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Number of features
n_features = X_train_scaled.shape[1]



# Linear Regression Model
model = LinearRegression()


# Initialize parameters
theta = np.zeros((n_features, 1))

def compute_cost(X, y, theta):
    m = len(y)
    predictions = np.dot(X, theta)
    error = predictions - y
    cost = (1 / (2 * m)) * np.sum(error ** 2)
    return cost

def compute_gradient(X, y, theta):
    m = len(y)
    predictions = np.dot(X, theta)
    error = predictions - y
    gradient = (1 / m) * np.dot(X.T, error)
    return gradient

def update_parameters(theta, gradient, learning_rate):
    new_theta = theta - learning_rate * gradient
    return new_theta

# Training Loop
num_iterations = 10000
learning_rate = 0.05

for i in range(num_iterations):
    # Compute gradient
    gradient = compute_gradient(X_train_scaled, y_train, theta)
    
    # Update parameters
    theta = update_parameters(theta, gradient, learning_rate)
    
    # Compute cost (optional - for monitoring)
    cost = compute_cost(X_train_scaled, y_train, theta)
    print(f"Iteration {i+1}, Cost: {cost}")

# Trained parameters
print("Trained Parameters (theta):")
print(theta)

# Evaluate the model on the test set (compute MSE)
predictions_test = np.dot(X_test_scaled, theta)
y_test_reshaped = y_test.reshape(-1, 1)  # Reshape y_test to match the shape of predictions_test
mse_test = np.mean((predictions_test - y_test_reshaped) ** 2)
print("Mean Squared Error on Test Set:", mse_test)

# Calculate linear regression parameters (slope and intercept)
slope, intercept = np.polyfit(X.flatten(), y, 1)
print(f"Linear Regression: y = {slope} * x + {intercept}")


import numpy as np
import scipy.stats as stats



# Calculate linear regression parameters (slope and intercept)
slope, intercept = np.polyfit(X.flatten(), y, 1)

print(f"Linear Regression: y = {slope} * x + {intercept}")

# Calculate normal distribution parameters (mean and standard deviation) for X
mu_X, std_X = stats.norm.fit(X)
print(f"Normal Distribution for X: Mean = {mu_X}, Standard Deviation = {std_X}")

# Calculate normal distribution parameters (mean and standard deviation) for y
mu_y, std_y = stats.norm.fit(y)
print(f"Normal Distribution for y: Mean = {mu_y}, Standard Deviation = {std_y}")




import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm





import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

# Define the features (X) and labels (y) as NumPy arrays
X = np.array([130, 70, 100, 100, 110, 120, 130, 155, 235, 650, 
             685, 85, 105, 130, 130, 130, 130, 89, 72,
             75, 70, 71,120,80,130,120,55,85,302,
             302,192,123,310,200]).reshape(-1, 1)
y = np.array([3300000, 4200000, 4250000, 10500000, 9950000, 6600000, 18000000, 4650000, 34500000, 52000000,
            16500000, 14500000, 14500000, 12750000, 12750000, 12750000, 8900000, 12000000, 4250000,
            14000000, 11000000, 4300000,3900000,3900000,3900000,3950000,1450000,1450000,16700000,
            16250000,10700000,10500000,11000000,3890000])

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train the linear regression model
model = LinearRegression()
model.fit(X_train_scaled, y_train)
y_pred=model.predict(X_test_scaled)

#predicting
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

# Assuming y_test_reshaped and y_pred are defined earlier

mae = mean_absolute_error(y_test, y_pred)
print('Mean Absolute Error:', mae)

mse = mean_squared_error(y_test, y_pred)
print('Mean Squared Error:', mse)

rmse = np.sqrt(mse)
print('Root Mean Squared Error:', rmse)

print('R2 Score:',metrics.r2_score(y_test,y_pred))

# Get the coefficients (slope and intercept)
slope = model.coef_[0]
intercept = model.intercept_


# Plot the training and testing data points
plt.scatter(X_train_scaled, y_train, color='blue', label='Training Data')
plt.scatter(X_test_scaled, y_test, color='red', label='Testing Data')

# Plot the regression line
x_values = np.linspace(min(np.min(X_train_scaled), np.min(X_test_scaled)), 
                       max(np.max(X_train_scaled), np.max(X_test_scaled)), 100)
y_values = slope * x_values + intercept
plt.plot(x_values, y_values, color='green', label='Regression Line')

plt.xlabel('X (Standardized)')
plt.ylabel('y')
plt.title('Linear Regression')
plt.legend()
plt.show()

# Reshape y_pred
y_pred_reshaped = y_pred.reshape(-1, 1)

print("Shape of y_test:", y_test.shape)
print("Shape of y_pred:", y_pred.shape)

# Plot the predictions against the actual values
plt.scatter(y_test_reshaped, y_pred_reshaped)
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Actual vs Predicted Values')
plt.show()


plt.scatter(y_test,y_pred_reshaped)
plt.grid()
plt.show()

import pandas as pd
Compare=pd.DataFrame({'Actual':y_test.flatten(),'Predict':y_pred.flatten()})
print(Compare)



