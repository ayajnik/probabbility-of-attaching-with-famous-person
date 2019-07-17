# -*- coding: utf-8 -*-
"""RNN_2

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1g6XSxsWwzzuzADRI8qF1WRojUGIfX4E4
"""

#importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

print("Libraries imported.")

#importing the dataset and feature scaling
dataset_train = pd.read_excel("residuals.xlsx")
training_set = dataset_train.iloc[:,0:1].values
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0,1))
training_set_scaled = sc.fit_transform(training_set)
print ("\n")
print("Imported the libraries.")
print("\n")
print("Feature scaling also done.")

a = len(training_set_scaled)
print(a)

#creating a data structure with some timesteps and 1 output
x_train = []
y_train = []
for i in range(2,4):
  x_train.append(training_set_scaled[i-2:i,0])
  y_train.append(training_set_scaled[i,0])
x_train, y_train = np.array(x_train), np.array(y_train)

b = len(x_train)
print (b)
x_train

#reshaping
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

#importing keras modules
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
print("\n")
print("Keras modules uploaded.")

#initialising the RNN
regressor = Sequential()

#adding the first LSTM layer and some dropout
regressor.add(LSTM(units = 50, return_sequences = True, input_shape = (x_train.shape[1], 1)))
regressor.add(Dropout(0.2))

#adding the first LSTM layer and some dropout
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))

#adding the first LSTM layer and some dropout
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))

#adding the first LSTM layer and some dropout
regressor.add(LSTM(units = 50))
regressor.add(Dropout(0.2))

#adding the output layer
regressor.add(Dense(units = 1))
print("\n")
print("Output layer created")

#compiling the RNN
regressor.compile(optimizer='adam', loss = 'mean_squared_error')
print('Compilation done')

#fitting the rnn to the training set
regressor.fit(x_train, y_train, epochs = 100,batch_size = 32)

#making the prediction and visualizing the result
dataset_test = pd.read_excel("residuals.xlsx")

#getting the predicted stock price
inputs = (dataset_test).values
inputs.reshape(-1,1)
inputs = sc.transform(inputs)
x_test = []
for i in range(2,4):
  x_test.append(inputs[i-2:i,0])
x_test = np.array(x_test)
x_test = np.reshape(x_test,(x_test.shape[0], x_test.shape[1],1))
predicted_subscribers = regressor.predict(x_test)
predicted_subscribers = sc.inverse_transform(predicted_subscribers)

#visualizing the result
plt.plot(dataset_test,color = 'red',label = "Residuas from ARIMA model")
plt.plot(predicted_subscribers,color='blue',label = "Predicted subscribers from RNN")
plt.title("Youtube Subscribers of T-Series")
plt.ylabel("Subscribers")
plt.legend()
plt.show()

print(predicted_subscribers)

