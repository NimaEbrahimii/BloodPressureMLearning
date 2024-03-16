import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
#metrics traintestsplit
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
#warnings.simplefilter(action='ignore')
#plt.style.use('seaborn')

Data=pd.read_csv('data.csv')
#print(Data)

print(Data.shape)
#print(Data.isna().sum())

#Data['Pregnancy'].fillna(0, inplace=True)


# Check for missing values
missing_values = Data.isnull()

# Display the DataFrame with True/False values indicating missing data
#print(missing_values)


#Numerical=['Level_of_Hemoglobin', 'Genetic_Pedigree_Coefficient',	'Age'	,'BMI'	,'Sex'	,'Pregnancy'	,'Smoking',	'Physical_activity',	'salt_content_in_the_diet'	,'alcohol_consumption_per_day',	'Level_of_Stress','	Chronic_kidney_disease',	'Adrenal_and_thyroid_disorders']
#i=0
#while i<13:
   # fig=plt.figure(figsize=[15,5])
   # plt.subplot(1,2,1)
   # sns.boxplot(x=Numerical[i],data=Data)
   # i+=1
   # plt.subplot(1,2,2)
   # sns.boxplot(x=Numerical[i],data=Data)
   # i+=1
   # plt.show()

import pandas as pd

# Annahme: Ihr DataFrame heißt "Data"
features = ['Level_of_Hemoglobin', 'Genetic_Pedigree_Coefficient', 'Smoking', 'Physical_activity', 'salt_content_in_the_diet', 'alcohol_consumption_per_day','Adrenal_and_thyroid_disorders','BMI','Age','Chronic_kidney_disease','Level_of_Stress']
target = 'Blood_Pressure_Abnormality'

# Berechnung der Korrelationen
correlations = Data[features + [target]].corr()

# Zeigen Sie die Korrelationsmatrix an
print(correlations)


import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 8))
sns.heatmap(correlations, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Korrelationsmatrix')
plt.show()


import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load the dataset
Data = pd.read_csv('data.csv')

# Define features and target variable
features = ['Level_of_Hemoglobin', 'Genetic_Pedigree_Coefficient', 'Smoking', 'Physical_activity', 'salt_content_in_the_diet', 'alcohol_consumption_per_day','Adrenal_and_thyroid_disorders','BMI','Age','Chronic_kidney_disease','Level_of_Stress']
target = 'Blood_Pressure_Abnormality'

# Split the data into features (X) and target variable (y)
X = Data[features]
y = Data[target]
#replace missing values of y with mean imputer
import numpy as np

# Assuming 'y' is your NumPy array with missing values

# Calculate the mean of 'y' excluding NaN values
mean_y = np.nanmean(y)

# Find indices of NaN values in 'y'
nan_indices = np.isnan(y)

# Replace NaN values in 'y' with the mean
y[nan_indices] = mean_y
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Impute missing values in features
imputer = SimpleImputer(strategy='mean')
X_train_imputed = imputer.fit_transform(X_train)
X_test_imputed = imputer.transform(X_test)

# Convert continuous target variable to classes based on quartiles
#y_train_classes = pd.qcut(y_train, q=3, labels=['Low', 'Medium', 'High'])


# Train the SVM model on the classes
#svm_classifier = SVC(C=1.0, kernel='rbf')
#svm_classifier.fit(X_train_imputed, y_train_classes)

# Make predictions
#predictions = svm_classifier.predict(X_test_imputed)

# Calculate the accuracy of the model
#accuracy = accuracy_score(y_test, predictions)
#print("Accuracy:", accuracy)
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics

# Load the dataset
Data = pd.read_csv('data.csv')

# Define features and target variable
features = ['Level_of_Hemoglobin', 'Genetic_Pedigree_Coefficient', 'Smoking', 'Physical_activity', 'salt_content_in_the_diet', 'alcohol_consumption_per_day','Adrenal_and_thyroid_disorders','BMI','Age','Chronic_kidney_disease','Level_of_Stress']
target = 'Blood_Pressure_Abnormality'

# Replace missing values of the target variable 'Blood_Pressure_Abnormality' with the mean
mean_y = Data[target].mean()
# Replace missing values of the target variable 'Blood_Pressure_Abnormality' with the mean
Data[target] = Data[target].fillna(mean_y)

# Split the data into features (X) and target variable (y)
X = Data[features]
y = Data[target]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Convert continuous target variable to classes based on thresholds
# Define thresholds for categorizing the continuous target variable into classes
threshold_low = 50
threshold_high = 100

# Convert the continuous target variable 'y' into classes 'Low', 'Medium', 'High'
y_train_classes = pd.cut(y_train, bins=[-np.inf, threshold_low, threshold_high, np.inf], labels=['Low', 'Medium', 'High'])


# Train a Decision Tree Classifier
clf = DecisionTreeClassifier(criterion="entropy", random_state=100, max_depth=4, min_samples_leaf=5)
# Train the Decision Tree Classifier on the classes
clf.fit(X_train, y_train_classes)
# Make predictions
y_pred = clf.predict(X_test)

#train the dataset on a polynomial regression model and predict some values
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
# Create polynomial features
poly = PolynomialFeatures(degree=2)
X_train_poly = poly.fit_transform(X_train_imputed)
X_test_poly = poly.transform(X_test_imputed)
# Train a Linear Regression model on the polynomial features
poly_reg = LinearRegression()
poly_reg.fit(X_train_poly, y_train)
# Make predictions
y_pred_poly = poly_reg.predict(X_test_poly)

#sketch the polynomial regression on the dataset
# Plot the polynomial regression 
plt.figure(figsize=(10, 6)) 
# Use integer indexing to access the column for 'Level_of_Hemoglobin' 
plt.scatter(X_test_imputed[:, 2], y_test, color='green', label='Actual') 
plt.plot(X_test_imputed[:, 2], y_pred_poly, color='red', linewidth=2, label='Predicted') 
plt.xlabel('Level_of_Hemoglobin') 
plt.ylabel('Blood Pressure') 
plt.title('Polynomial Regression') 
plt.legend() 
plt.show()
