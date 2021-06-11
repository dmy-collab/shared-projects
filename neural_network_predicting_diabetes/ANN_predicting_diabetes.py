import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import classification_report, confusion_matrix
pd.set_option('display.max_columns', None)

# importing the file:
dataset = pd.read_csv('diabetes.csv')
X = dataset.iloc[:, 1:-1]     # features
y = dataset.iloc[:, -1]    # labels

# data scaling:
sc = StandardScaler()
X_norm = sc.fit_transform(X)
X = X_norm

# splitting dataset into the training and test sets:
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)


# Compiling the Model:
model = Sequential()
model.add(Dense(units=8, activation='relu', input_shape=(8,)))  # N - from number of features
model.add(Dropout(0.5))     # preventing overfitting
model.add(Dense(units=5, activation='relu'))    # (N of features + N of output)/2
model.add(Dropout(0.5))     # preventing overfitting
model.add(Dense(units=1, activation='sigmoid'))

# Choosing an optimizer and loss:
# For a binary classification problem:
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=["accuracy"])
print(f"\n\tModel summary: \n{model.summary()}")

# Early stopping as anti-overfitting:
early_stop = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=5)

# Training the model:
model.fit(x=X_train,
          y=y_train,
          epochs=400,
          validation_data=(X_test, y_test), verbose=1,
          callbacks=[early_stop]    # preventing overfitting
          )


# Evaluation of overfitting:
model_performance = pd.DataFrame(model.history.history)
ax = model_performance.plot()
ax.set_xlabel('Epoch')
plt.show()

# Model evaluation:
predictions = model.predict_classes(X_test)
print(f"\n\tConfusion  matrix: \n{confusion_matrix(y_test, predictions)}")
print(f"\n\tClassification report: \n{classification_report(y_test, predictions)}")

# Predicting new instance:
new_data = dataset.iloc[2, 1:-1].values
new_predicted = model.predict(sc.transform([new_data]))
print(f"\n\tPredicted New instance: \n{new_predicted}")
