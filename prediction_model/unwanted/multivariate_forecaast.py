import pandas as pd
import numpy as np
import matplotlib
from datetime import datetime as dt
matplotlib.use('TkAgg') 
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, Activation, Bidirectional
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import mean_squared_error 
from numpy import concatenate
from math import sqrt

def readCSV(state_name):
	df = pd.read_csv('prediction_model/state_fb1.csv')

	#get active cases and restrictions
	state_cases = df.loc[df['state_name'] == state_name]
	state_cases.fillna(method='ffill')
	state_cases.dropna(inplace = True)
	state_active = state_cases.iloc[:, [5, 6]]

	# place the date as index
	state_active.index = pd.to_datetime(state_cases['date'], format='%d/%m/%Y')
	print(state_active)
	plt.plot(state_active)
	# plt.show()
	return state_active

def normalizeData1(dataset):
	scaler = MinMaxScaler()
	scaler = MinMaxScaler(feature_range=(0, 1))
	dataset = scaler.fit_transform(dataset)
	return scaler, dataset

def splitDataset(dataset):
	#divide the data into train and test data
	train_size = int(len(dataset)*0.8)
	train = dataset[:train_size,:]
	test = dataset[train_size:, :]
	return train, test

def dataCleanup(train, test):
	#index the data into dependent and independent variables
	train_X, train_y = train[:, :], train[:, 0]
	test_X, test_y = test[:, :], test[:, 0]
	print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)

	#convert data into suitable dimension for using it as input in LSTM network
	train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
	test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))
	print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)
	return train_X, train_y, test_X, test_y

def createBiLSTM(units, X_train, y_train):
	#BiLSTM 
	model = Sequential()
	# Input layer
	model.add(Bidirectional(
				LSTM(units = units,activation="relu", return_sequences=True), 
				input_shape=(X_train.shape[1], X_train.shape[2])))
	# Hidden layer
	model.add(Bidirectional(LSTM(units=units, activation="relu")))
	model.add(Dense(1))
	#Compile model
	model.compile(optimizer="adam",loss="mse")
	return model 

def fitModel(model, X_train, y_train):
	early_stop = EarlyStopping(monitor = 'val_loss', patience = 10)
	history = model.fit(X_train, y_train, epochs = 250,  
						validation_data=(test_X, test_y),
						batch_size = 16, shuffle = False, 
						callbacks = [early_stop])
	return history


def plotLoss (history):
	plt.figure(figsize = (10, 6))
	plt.plot(history.history['loss'])
	plt.plot(history.history['val_loss'])
	plt.ylabel('Loss')
	plt.xlabel('epoch')
	plt.legend(['Train loss', 'Validation loss'], loc='upper right')
	# plt.show()

def prediction(model, test_X, test_y):

	test_predict = model.predict(test_X) 
	test_X = test_X.reshape((test_X.shape[0], test_X.shape[2]))
	inv_test_predict = concatenate((test_predict, test_X), axis=1)
	inv_test_predict = scaler.inverse_transform(inv_test_predict)
	inv_test_predict = inv_test_predict[:,0]

	test_y = test_y.reshape((len(test_y), 1))
	inv_test_y = concatenate((test_y, test_X), axis=1)
	inv_test_y = scaler.inverse_transform(inv_test_y)
	inv_test_y = inv_test_y[:,0]
	rmse_test = sqrt(mean_squared_error(inv_test_y, inv_test_predict))

	print(inv_test_predict, inv_test_y )
	print('Test RMSE: %.3f' % rmse_test)


if __name__ == "__main__":
	# Set random seed to get the same result after each time running the code
	tf.random.set_seed(1111)
	#prediction based on number of days and number of features
	SEQ_LENGTH = 5
	N_FEATURES = 1

	DAYS_TO_PREDICT= 7

	state = readCSV("selangor")
	scaler, dataset = normalizeData1(state)
	train, test = splitDataset(dataset)
	train_X, train_y, test_X, test_y = dataCleanup(train, test)
	model = createBiLSTM(60, train_X, train_y)
	history = fitModel(model, train_X, train_y)
	plotLoss(history)
	prediction_bilstm = prediction(model, test_X, test_y)


