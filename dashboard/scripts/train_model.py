import pandas as pd
import numpy as np
import matplotlib
from datetime import datetime as dt, date
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras import models
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, Activation, Bidirectional
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
from tensorflow.keras.callbacks import EarlyStopping
import os
from pathlib import Path
import csv

# Set random seed to get the same result after each time running the code
tf.random.set_seed(1111)
#prediction based on number of days 
SEQ_LENGTH = 5

N_FEATURES = 2

def read_country_confirmed(c_path):
	
	df = pd.read_csv(c_path)
	df.fillna(method='ffill', inplace=True)
	df.dropna(inplace = True)
	# to do: take restriction 
	confirmed = df.iloc[:, [2, 8]]
	print(confirmed)

	# place the date as index
	confirmed.index = pd.to_datetime(df['date'], format='%d/%m/%Y')
	return confirmed

def read_state_active(state_name, s_path):
	
	df = pd.read_csv(s_path)

	state_cases = df.loc[df['state_name'] == state_name].copy()
	state_cases.fillna(method='ffill', inplace=True)
	state_cases.dropna(inplace = True)
	state_active = state_cases.iloc[:, [5,6]]

	# place the date as index
	state_active.index = pd.to_datetime(state_cases['date'], format='%d/%m/%Y')
	return state_active

def splitDataset(data):
	train_size = len(data)-SEQ_LENGTH

	train = data[:train_size-1]
	test = data[train_size-1:]

	return train, test

def normalizeData(train, test):
	scaler = MinMaxScaler(feature_range=(0, 1))
	scaler.fit(train)
	scaled_train = scaler.transform(train)
	scaled_test = scaler.transform(test)
	return scaled_train, scaled_test, scaler


def createSequence(scaled_train):
	generator = TimeseriesGenerator(scaled_train,scaled_train,length = SEQ_LENGTH,batch_size=1)
	
	# uncomment if want to see the trainng data
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
	model.add(Dense(N_FEATURES))
	#Compile model
	model.compile(optimizer="adam",loss="mse")
	return model 

def fitModel(model,scaled_test, generator):
   
	validation_gen = TimeseriesGenerator(scaled_test,scaled_test,length=SEQ_LENGTH,batch_size=1)

	# uncomment to look at the validation data
	# for i in range(len(validation_gen)):
	# 	x, y = validation_gen[i]
	# 	print('%s => %s' % (x, y))
	
	early_stop = EarlyStopping(monitor='val_loss',patience=30, restore_best_weights=True)

	history = model.fit(generator,validation_data=validation_gen,epochs=1000, callbacks=[early_stop],steps_per_epoch=10, verbose=0)
	return history

def forecast_active(model, scaled_train, test, scaler, DAYS_TO_PREDICT):
	test_prediction = []
	first_eval_batch = scaled_train[-SEQ_LENGTH:]
	current_batch = first_eval_batch.reshape(1,SEQ_LENGTH, N_FEATURES)
	for i in range(len(test)+DAYS_TO_PREDICT):
		current_pred = model.predict(current_batch)[0]
		test_prediction.append(current_pred)
		# use current predicted value to predict 
		current_batch = np.append(current_batch[:,1:,:],[[current_pred]],axis=1)
	


	### inverse scaled data
	true_prediction = scaler.inverse_transform(test_prediction)
	time_series_array = test.index
	for k in range(0,DAYS_TO_PREDICT):
		time_series_array = time_series_array.append(time_series_array[-1:] + pd.DateOffset(1))

	# forecast active
	df_forecast = pd.DataFrame(columns=["active","active_predicted"],index=time_series_array)
	df_forecast.loc[:,"active_predicted"] = true_prediction[:,0]
	df_forecast.loc[:,"active"] = test["active_cases"]

	print(df_forecast)

	errors = np.array(df_forecast["active_predicted"][:len(test)]) - np.array(df_forecast["active"][:len(test)])
	mse = np.mean(np.square(errors))
	rmse = np.sqrt(mse)
	mae = np.mean(np.abs(errors))
	print('Mean Square Error: {:.4f}'.format(mse))
	print('Mean Absolute Error: {:.4f}'.format(mae))
	print('Root Mean Square Error: {:.4f}'.format(rmse))
	print('')
	MAPE = np.mean(np.abs(np.array(df_forecast["active"][:len(test)]) - np.array(df_forecast["active_predicted"][:len(test)]))/np.array(df_forecast["active"][:len(test)]))
	print("MAPE is " + str(MAPE*100) + " %")

	return df_forecast[df_forecast['active'].isnull()]


def forecast_confirmed(model, scaled_train, test, scaler, DAYS_TO_PREDICT):
	test_prediction = []
	first_eval_batch = scaled_train[-SEQ_LENGTH:]
	current_batch = first_eval_batch.reshape(1,SEQ_LENGTH, N_FEATURES)
	for i in range(len(test)+DAYS_TO_PREDICT):
		current_pred = model.predict(current_batch)[0]
		test_prediction.append(current_pred)
		# use current predicted value to predict 
		current_batch = np.append(current_batch[:,1:,:],[[current_pred]],axis=1)
	


	### inverse scaled data
	true_prediction = scaler.inverse_transform(test_prediction)
	time_series_array = test.index
	for k in range(0,DAYS_TO_PREDICT):
		time_series_array = time_series_array.append(time_series_array[-1:] + pd.DateOffset(1))

	# forecast confirmed
	df_forecast = pd.DataFrame(columns=["confirmed","confirmed_predicted"],index=time_series_array)
	df_forecast.loc[:,"confirmed_predicted"] = true_prediction[:,0]
	df_forecast.loc[:,"confirmed"] = test["total_infected"]

	print(df_forecast)

	errors = np.array(df_forecast["confirmed_predicted"][:len(test)]) - np.array(df_forecast["confirmed"][:len(test)])
	mse = np.mean(np.square(errors))
	rmse = np.sqrt(mse)
	mae = np.mean(np.abs(errors))
	print('Mean Square Error: {:.4f}'.format(mse))
	print('Mean Absolute Error: {:.4f}'.format(mae))
	print('Root Mean Square Error: {:.4f}'.format(rmse))
	print('')
	MAPE = np.mean(np.abs(np.array(df_forecast["confirmed"][:len(test)]) - np.array(df_forecast["confirmed_predicted"][:len(test)]))/np.array(df_forecast["confirmed"][:len(test)]))
	print("MAPE is " + str(MAPE*100) + " %")

	return df_forecast[df_forecast['confirmed'].isnull()]


def run():
	DAYS_TO_PREDICT = 7
	BASE_DIR = Path(__file__).resolve(strict=True).parent.parent.parent
	models_path = os.path.join(BASE_DIR, 'models')
	model_c = train_country(BASE_DIR, models_path, DAYS_TO_PREDICT)
	train_state(BASE_DIR, models_path, DAYS_TO_PREDICT)
	forecast_confirmed(model_c, scaled_train, test, scaler, DAYS_TO_PREDICT)

def train_country(BASE_DIR, models_path, DAYS_TO_PREDICT):

	# train country
	c_path = os.path.join(BASE_DIR, 'data/malaysia_cases.csv')
	model_name = 'model_malaysia.h5'
	path = os.path.join(models_path, model_name )
	state_active = read_country_confirmed(c_path)
	train, test = splitDataset(state_active)
	scaled_train, scaled_test, scaler = normalizeData(train, test)
	generator = createSequence(scaled_train)
	model = createBiLSTM(60)
	history = fitModel(model, scaled_test, generator)
	model.save(path)
	


def train_state(BASE_DIR, models_path, DAYS_TO_PREDICT):
	name_path = os.path.join( BASE_DIR, 'data/state_name.csv')
	s_path = os.path.join( BASE_DIR, 'data/state_fb.csv')

	with open(name_path, 'r') as states_file:
		csv_r = csv.reader(states_file)
		next(csv_r)
		for r in csv_r: 
			state_name = r[0]
			print(state_name)
			model = train_state_individual(state_name, BASE_DIR, models_path, s_path, DAYS_TO_PREDICT)
			forecast_active(model, scaled_train, test, scaler, DAYS_TO_PREDICT)

			
		
def train_state_individual(state_name, BASE_DIR, models_path, s_path, DAYS_TO_PREDICT):
	model_name = 'model_' + state_name + '.h5'
	path = os.path.join(models_path, model_name )
	state_active = read_state_active(state_name, s_path)
	train, test = splitDataset(state_active)
	scaled_train, scaled_test, scaler = normalizeData(train, test)
	generator = createSequence(scaled_train)
	model = createBiLSTM(60)
	history = fitModel(model, scaled_test, generator)
	model.save(path)
	return model 
	
