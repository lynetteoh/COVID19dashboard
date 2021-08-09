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
  

def readCSV(state_name):
	df = pd.read_csv('prediction_model/state_fb1.csv')

	'''
	#get total infected

	# get first 3 columns 
	total_confirmed = df.iloc[:, :3]

	# get cases for a state
	state_cases = total_confirmed.loc[total_confirmed['state_name'] == state_name]

	#get only total infected
	state_confirmed = state_cases.iloc[:, [2]]

	# get daily infected
	# state_confirmed = state_confirmed.diff(axis=0).fillna(0)

	# place the date as index
	state_confirmed.index = pd.to_datetime(state_cases['date'], format='%d/%m/%Y')

	print(state_confirmed)
	plt.plot(state_confirmed)

	return state_confirmed
	'''

	#get active cases
	state_cases = df.loc[df['state_name'] == state_name].copy()
	state_cases.dropna(inplace = True)
	state_active = state_cases.iloc[:, [5]]

	# place the date as index
	state_active.index = pd.to_datetime(state_cases['date'], format='%d/%m/%Y')
	# state_confirmed.to_csv('prediction_model/cummulative.csv')
	print(state_active)
	plt.plot(state_active)
	# plt.show()
	return state_active
	
def splitDataset(data):
	train_size = int(len(data)*0.8)
	# train_size = len(data)-SEQ_LENGTH

	train = data[:train_size-1]
	test = data[train_size-1:]

	return train, test

def normalizeData(train, test):
	scaler = MinMaxScaler()
	scaler.fit(train)
	scaled_train = scaler.transform(train)
	scaled_test = scaler.transform(test)
	return scaled_train, scaled_test, scaler


def createSequence(scaled_train, scaled_test):
	generator = TimeseriesGenerator(scaled_train,scaled_train,length = SEQ_LENGTH,batch_size=1)
   
	# for i in range(len(generator)):
	#     x, y = generator[i]
	#     print('%s => %s' % (x, y))

	return generator

def createBiLSTM(units):
	#BiLSTM 
	model = Sequential()
	# Input layer
	model.add(Bidirectional(
				LSTM(units = units,activation="relu", return_sequences=True), 
				input_shape=(SEQ_LENGTH,N_FEATURES)))
	model.add(Bidirectional(LSTM(units=int(units), activation="relu")))
	# Hidden layer	
	model.add(Dense(1))
	#Compile model
	model.compile(optimizer="adam",loss="mse")
	return model 

def createLSTM(units):
	# LSTM model 
	model = Sequential()
	model.add(LSTM(units,activation="relu", input_shape=(SEQ_LENGTH, N_FEATURES)))
	model.add(Dense(units/2, activation='relu'))
	model.add(Dense(units=1))
	model.compile(optimizer="adam",loss="mse")
	return model


def fitModel(model, scaled_train, scaled_test, generator):
   
	validation_set = scaled_test
	# validation_set=validation_set.reshape(len(scaled_test),1)
	validation_gen = TimeseriesGenerator(validation_set,validation_set,length=SEQ_LENGTH,batch_size=1)
	print(len(validation_gen))
	for i in range(len(validation_gen)):
		x, y = validation_gen[i]
		print('%s => %s' % (x, y))
	early_stop = EarlyStopping(monitor='val_loss',patience=50, restore_best_weights=True)

	history = model.fit(generator,validation_data=validation_gen,epochs=1000,callbacks=[early_stop],steps_per_epoch=10)
	return history


def plotLoss(history, model_name):
	# pd.DataFrame(model.history.history).plot(title="loss vs epochs curve")
	plt.figure(figsize = (10, 6))
	plt.plot(history.history['loss'])
	plt.plot(history.history['val_loss'])
	plt.title('Model Train vs Validation Loss for ' + model_name)
	plt.ylabel('Loss')
	plt.xlabel('epoch')
	plt.legend(['Train loss', 'Validation loss'], loc='upper right')
	# plt.show()

def forecast(model, scaled_train, test, scaler):
	test_prediction = []
	first_eval_batch = scaled_train[-SEQ_LENGTH:]
	current_batch = first_eval_batch.reshape(1,SEQ_LENGTH, N_FEATURES)
	for i in range(len(test)+DAYS_TO_PREDICT):
		current_pred = model.predict(current_batch)[0]
		print(current_pred)
		test_prediction.append(current_pred)
		# use current predicted value to predict 
		current_batch = np.append(current_batch[:,1:,:1],[[current_pred]],axis=1)
	


	### inverse scaled data
	true_prediction = scaler.inverse_transform(test_prediction)
	time_series_array = test.index
	for k in range(0,DAYS_TO_PREDICT):
		time_series_array = time_series_array.append(time_series_array[-1:] + pd.DateOffset(1))

	# forecast confirmed
	# df_forecast = pd.DataFrame(columns=["confirmed","confirmed_predicted"],index=time_series_array)
	# df_forecast.loc[:,"confirmed_predicted"] = true_prediction[:,0]
	# df_forecast.loc[:,"confirmed"] = test["total_infected"]

	# forecast active
	df_forecast = pd.DataFrame(columns=["active","active_predicted"],index=time_series_array)
	df_forecast.loc[:,"active_predicted"] = true_prediction[:,0]
	df_forecast.loc[:,"active"] = test["active_cases"]

	print(df_forecast)

	df_forecast.plot(title="Predictions for next 7 days")
	# plt.show()

	# errors = np.array(df_forecast["confirmed_predicted"][:len(test)]) - np.array(df_forecast["confirmed"][:len(test)])
	errors = np.array(df_forecast["active_predicted"][:len(test)]) - np.array(df_forecast["active"][:len(test)])
	mse = np.mean(np.square(errors))
	rmse = np.sqrt(mse)
	mae = np.mean(np.abs(errors))
	print('Mean Square Error: {:.4f}'.format(mse))
	print('Mean Absolute Error: {:.4f}'.format(mae))
	print('Root Mean Square Error: {:.4f}'.format(rmse))
	print('')
	# MAPE = np.mean(np.abs(np.array(df_forecast["confirmed"][:len(test)]) - np.array(df_forecast["confirmed_predicted"][:len(test)]))/np.array(df_forecast["confirmed"][:len(test)]))
	MAPE = np.mean(np.abs(np.array(df_forecast["active"][:len(test)]) - np.array(df_forecast["active_predicted"][:len(test)]))/np.array(df_forecast["active"][:len(test)]))
	print("MAPE is " + str(MAPE*100) + " %")

if __name__ == "__main__":
	# Set random seed to get the same result after each time running the code
	tf.random.set_seed(1111)
	#prediction based on number of days 
	SEQ_LENGTH = 5
	N_FEATURES = 1

	DAYS_TO_PREDICT= 7

	state_confirmed = readCSV("selangor")
	train, test = splitDataset(state_confirmed)
	scaled_train, scaled_test, scaler = normalizeData(train, test)
	generator = createSequence(scaled_train, scaled_test)
	model = createBiLSTM(60)
	# model = createLSTM(120)
	history = fitModel(model, scaled_train, scaled_test, generator)
	plotLoss(history, "BiLSTM")
	forecast(model, scaled_train, test, scaler)
