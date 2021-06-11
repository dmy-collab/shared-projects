import pandas as pd
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

dataset = pd.read_csv('Height.txt', delimiter='\t')
print(dataset.head())
print(dataset.describe().T)
sns.pairplot(dataset, hue='Gender')     # initial visualization seeking correlation
plt.show()

# Splitting the datasets for boys and girls:
boys = dataset[dataset['Gender'] == 'M']
boys.info()
girls = dataset[dataset['Gender'] == 'F']
girls.info()

# Selecting X:Features and y: Output:
X = boys[['Father', 'Mother']]
y = boys[['Height']]

# Importing methods form Scikit-Learn:
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics

# Train-Test Split:
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=110)

# Training Model:
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Prediction of Y-test:
predictions = regressor.predict(X_test)
plt.scatter(y_test, predictions, color='k')
plt.xlabel('Test Data')
plt.ylabel('Predicted Data')
plt.title('Test Data Vs. Predicted Data Boys')
plt.show()

# Intercept and slope:
print(regressor.intercept_)
print(regressor.coef_)

coeff_df = pd.DataFrame(regressor.coef_.T, X.columns, columns=['Coefficient'])
print(coeff_df)     # the importance of the feature

# Evaluation Matrices:
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, predictions))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, predictions))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))
print('R Squared:', metrics.r2_score(y_test, predictions))  # how many data points fall within the results of
                                                            # the line formed by the regression equation

# Custom Prediction:
father_height = 78
mother_height = 60
son_height = regressor.predict(np.array([[father_height, mother_height]]))

print(f"Predicted height boy: {son_height}")

# For Girls:
X = girls[['Father', 'Mother']]
y = girls[['Height']]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=110)
regressor = LinearRegression()
regressor.fit(X_train, y_train)
predictions = regressor.predict(X_test)
plt.scatter(y_test, predictions, color='k')
plt.xlabel('Test Data')
plt.ylabel('Predicted Data')
plt.title('Test Data Vs. Predicted Data Girls')
plt.show()

# To retrieve the intercept:
print(regressor.intercept_)
# For retrieving the slope:
print(regressor.coef_)

coeff_df = pd.DataFrame(regressor.coef_.T, X.columns, columns=['Coefficient'])
print(coeff_df)

# Custom Prediction:
father_height = 88
mother_height = 67
daughter_height = regressor.predict(np.array([[father_height, mother_height]]))
print(f"Predicted height girl: {daughter_height}")

print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, predictions))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, predictions))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))
print('R Squared:', metrics.r2_score(y_test, predictions))

