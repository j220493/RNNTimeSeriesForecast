# Libraries
import tensorflow as tf
from tensorflow import keras
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

# Rolling window function


def create_dataset(data, window=1):
    dataX, dataY = [], []
    for i in range(len(data)-window-1):
        a = data[i:(i+window), 0]
        dataX.append(a)
        dataY.append(data[i + window, 0])
    return np.array(dataX), np.array(dataY)


# Data
df = pd.read_csv('AEP_hourly.csv')
df.head()
df.info()
df.describe()

# Creating temporal features
dataset = df.copy()
dataset['month'] = pd.to_datetime(df['Datetime']).dt.month
dataset['year'] = pd.to_datetime(df['Datetime']).dt.year
dataset['date'] = pd.to_datetime(df['Datetime']).dt.date
dataset['time'] = pd.to_datetime(df['Datetime']).dt.time
print(dataset.head())

# Group by day and take de mean energy consumption
dataset = dataset.groupby('date')['AEP_MW'].mean().reset_index()
print(dataset.info())

# Plotting data
sns.lineplot(data=dataset, x='date', y='AEP_MW')

# Scaling dataset
dataset = dataset['AEP_MW'].values
scaler = MinMaxScaler()
dataset = scaler.fit_transform(dataset.reshape(-1, 1))

# Train and test division
train_size = int(len(dataset) * 0.67)
test_size = len(dataset) - train_size
train, test = dataset[0:train_size, :], dataset[train_size:len(dataset), :]
print(len(train), len(test))

# Creating features into X=t and Y=t+1
look_back = 7
trainX, trainY = create_dataset(train, look_back)
testX, testY = create_dataset(test, look_back)
print(trainX.shape)

# Reshaping [batch, times, features]
trainX = np.reshape(trainX, (trainX.shape[0], look_back, 1))
testX = np.reshape(testX, (testX.shape[0], look_back, 1))
print(trainX.shape)

# Training model 1 (Vanilla RNN)
initializer = keras.initializers.GlorotNormal(seed=1234)
model1 = keras.models.Sequential()
model1.add(keras.layers.SimpleRNN(32, input_shape=(look_back, 1), kernel_initializer=initializer, return_sequences=True))
model1.add(keras.layers.SimpleRNN(64))
model1.add(keras.layers.Dense(1))
model1.summary()
model1.compile(loss='mean_squared_error', optimizer='adam')
history1 = model1.fit(trainX, trainY, epochs=20, batch_size=32)

# Training model 2 (LSTM)
initializer = keras.initializers.GlorotNormal(seed=1234)
model2 = keras.models.Sequential()
model2.add(keras.layers.LSTM(32, input_shape=(look_back, 1), kernel_initializer=initializer, return_sequences=True))
model2.add(keras.layers.LSTM(64))
model2.add(keras.layers.Dense(1))
model2.summary()
model2.compile(loss='mean_squared_error', optimizer='adam')
history2 = model2.fit(trainX, trainY, epochs=20, batch_size=32)

# Evaluating loss
plt.plot(history1.history['loss'], color='teal', label='loss RNN')
plt.plot(history2.history['loss'], color='orange', label='loss LSTM')
plt.legend(loc='upper left')


# Forecasting
yhatRNN = model1.predict(testX)
yhatLSTM = model2.predict(testX)

# Decoding predictions
testYDec = scaler.inverse_transform(testY.reshape(-1, 1))
yhatDecRNN = scaler.inverse_transform(yhatRNN)
yhatDecLSTM = scaler.inverse_transform(yhatLSTM)

# Plotting some predictions
plt.plot(testYDec, color='orange', label='test data')
plt.plot(yhatDecRNN, color='blue', label='predicted RNN')
plt.plot(yhatDecLSTM, color='green', label='predicted LSTM')
plt.legend(loc='upper left')

# Evaluating with metrics
mean_squared_error(testYDec, yhatDecRNN)
mean_squared_error(testYDec, yhatDecLSTM)
