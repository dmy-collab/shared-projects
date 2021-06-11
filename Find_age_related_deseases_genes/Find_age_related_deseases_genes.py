import pandas as pd
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
pd.set_option('display.max_columns', None)

# importing the file:
dataset = pd.read_excel('Genes_dataset.xlsx')

# initial analysis:
print(dataset.head(1))
print(dataset.shape)
dataset['Target'].value_counts()

# Splitting the dataset on features and labels:
X = dataset.iloc[:, 2:-1].values    # features
y = dataset.iloc[:, -1].values      # labels

# data scaling:
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_norm = sc.fit_transform(X)

# PCA:
from sklearn.decomposition import PCA
sklearn_pca = PCA(n_components=2)
PCs = sklearn_pca.fit_transform(X_norm)

data_transform = pd.DataFrame(PCs, columns=['PC1', 'PC2'])
data_transform = pd.concat([data_transform, dataset.iloc[:, -1]], axis=1)

fig, axes = plt.subplots(figsize=(10, 8))
sns.set_style("whitegrid")
sns.scatterplot(x='PC1', y='PC2', data=data_transform, hue='Diabetic', cmap='grey')
plt.show()      # visualizing distribution of instances to find whether there is linearity in data distribution

# Splitting the dataset into the Training set and Test set:
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_norm, y, test_size=0.20, random_state=101)

# Training the Logistic Regression model and evaluation:
from sklearn.linear_model import LogisticRegression
Logit_classifier = LogisticRegression(random_state=0)
Logit_classifier.fit(X_train, y_train)
y_pred = Logit_classifier.predict(X_test)

from sklearn.metrics import confusion_matrix, classification_report
print('\n\tLogistic regression evaluation:')
print(confusion_matrix(y_test, y_pred))
print('\n')
print(classification_report(y_test, y_pred))

# Training the KNN model and evaluation:
# Choosing K Value:
from sklearn.neighbors import KNeighborsClassifier

errors = []

for i in range(1, 100):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train, y_train)
    pred_i = knn.predict(X_test)
    errors.append(np.mean(pred_i != y_test))

plt.figure(figsize=(10, 6))
plt.plot(range(1, 100), errors, color='black', linestyle='dashed', marker='o',
         markerfacecolor='black', markersize=10)
plt.title('Error Rate vs. K Value')
plt.xlabel('K Value')
plt.ylabel('Error Rate')
plt.show()      # plotting error rate to find optimal K-value. K=7 has lowes error rate according to the plot.

from sklearn.neighbors import KNeighborsClassifier
KNN_classifier = KNeighborsClassifier(n_neighbors=7)
KNN_classifier.fit(X_train, y_train)
y_pred = KNN_classifier.predict(X_test)

from sklearn.metrics import confusion_matrix, classification_report
print('\n\tKNN model evaluation:')
print(confusion_matrix(y_test, y_pred))
print('\n')
print(classification_report(y_test, y_pred))

# Training the Random Forest model and evaluation:
from sklearn.model_selection import GridSearchCV    # find optimal N of trees with grid search
from sklearn.ensemble import RandomForestClassifier
param_grid = {'n_estimators': [10, 100, 150, 200, 250, 300, 350, 400]}
RF_classifier = GridSearchCV(RandomForestClassifier(), param_grid, refit=True, verbose=0)
RF_classifier.fit(X_train, y_train)
y_pred = RF_classifier.predict(X_test)

from sklearn.metrics import confusion_matrix, classification_report
print('\n\tRandom Forest model evaluation:')
print(confusion_matrix(y_test, y_pred))
print('\n')
print(classification_report(y_test, y_pred))

# Training the SVM model and evaluation:
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
param_grid = {'C': [0.1, 1, 10, 100, 1000], 'gamma': [1, 0.1, 0.01, 0.001, 0.0001], 'kernel': ['rbf']}
SVM_classifier = GridSearchCV(SVC(), param_grid, refit=True, verbose=0)
SVM_classifier.fit(X_train, y_train)

y_pred = SVM_classifier.predict(X_test)

from sklearn.metrics import confusion_matrix, classification_report
print('\n\t SVM model and evaluation')
print(confusion_matrix(y_test, y_pred))
print('\n')
print(classification_report(y_test, y_pred))

# Training the ANN model and evaluation:
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.callbacks import EarlyStopping

model = Sequential()
model.add(Dense(units=60, activation='relu'))
model.add(Dropout(0.5))     # minimizing overfitting with dropout layer

model.add(Dense(units=15, activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(units=1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

early_stop = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=5)
model.fit(x=X_train,
          y=y_train,
          epochs=400,
          batch_size=64,
          validation_data=(X_test, y_test), verbose=1,
          callbacks=[early_stop]
          )

predictions = model.predict_classes(X_test)

from sklearn.metrics import classification_report, confusion_matrix
print('\n\tANN model evaluation:')
print(confusion_matrix(y_test, predictions))
print(classification_report(y_test, predictions))


# K-NN and random forest have shown the best performance more likely
# due to the fact that they can classify non-linearly separable dataset
# additional networks cell-location based features can further improve accuracy
