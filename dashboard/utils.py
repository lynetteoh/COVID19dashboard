import pandas as pd
import numpy as np
import matplotlib
from datetime import datetime as dt
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras import models
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, Activation, Bidirectional
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
from tensorflow.keras.callbacks import EarlyStopping
import os
import random as rn 
from .scripts.train_model import *
from datetime import datetime, timedelta
from dashboard.models import Case, Zone


def forecast_country_confirmed(BASE_DIR, models_path, c_path, latest_c_date, DAYS_TO_PREDICT=7):
	# Case.objects.filter(Ref_Key=1, Is_actual=False).delete()

	# get forecast up to date
	last_pred_day =  latest_c_date + timedelta(DAYS_TO_PREDICT)
	last_forecast = Case.objects.filter(Reference_ID="MYS", Is_actual=False)
	
	if not last_forecast or last_forecast[0].Date < last_pred_day:
		if last_forecast:
			last_forecast_date = last_forecast[0].Date
		else:
			# no forecast data available in database
			last_forecast_date = latest_c_date

		model_name = 'model_malaysia.h5'
		path = os.path.join(models_path, model_name )
		found = False
		
		for f in os.listdir(models_path):
			if f == model_name:
				found = True
				break

		if not found: 
			train_country(BASE_DIR, models_path, DAYS_TO_PREDICT)

		model = models.load_model(path)
		state_active = read_country_confirmed(c_path)
		train, test = splitDataset(state_active)
		scaled_train, scaled_test, scaler = normalizeData(train, test)
		forecast = forecast_confirmed(model, scaled_train, test, scaler, DAYS_TO_PREDICT)
		
		# only create those not in database
		forecast =  forecast.loc[forecast.index > datetime(last_forecast_date.year, last_forecast_date.month, last_forecast_date.day)].copy()
		# print(forecast)

		# get previous total infected
		prev_total_infected = Case.objects.get(Reference_ID="MYS", Date=last_forecast_date).Total_infected
		
		for case in forecast.itertuples():
			#get date
			date = case[0].date()
			daily_infected = int(case.confirmed_predicted) - prev_total_infected
			print(date, daily_infected)
			c, created = Case.objects.get_or_create(Reference_ID="MYS", Date=date, defaults = { 'Ref_Key': 1, 'Total_infected': int(case.confirmed_predicted), 'Daily_infected' : daily_infected, 'Is_actual':False})
			prev_total_infected = int(case.confirmed_predicted)

def forecast_state_active(BASE_DIR, states, models_path, s_path, latest_s_date, DAYS_TO_PREDICT=7):
	# Case.objects.filter(Ref_Key=2, Is_actual=False).delete()
	# Zone.objects.filter(Ref_Key=1, Is_actual=False).delete()

	last_pred_day =  latest_s_date + timedelta(DAYS_TO_PREDICT)
	
	for state in states:
		last_forecast = Case.objects.filter(Reference_ID=state.State_ID, Is_actual=False)
	
		if not last_forecast or last_forecast[0].Date < last_pred_day:
			if last_forecast:
				last_forecast_date = last_forecast[0].Date
			else:
				# no forecast data available in database
				last_forecast_date = latest_s_date

			state_name = state.State_Name
			print(state_name)
			model_name = 'model_' + state_name + '.h5'
			path = os.path.join(models_path, model_name )
			found = False
			
			for f in os.listdir(models_path):
				if f == model_name:
					found = True
					break
			
			if not found: 
				print("training model")
				train_state_individual(state_name, BASE_DIR, models_path, s_path, DAYS_TO_PREDICT)

			model = models.load_model(path)
			state_active = read_state_active(state_name, s_path)
			train, test = splitDataset(state_active)
			scaled_train, scaled_test, scaler = normalizeData(train, test)
			forecast = forecast_active(model, scaled_train, test, scaler, DAYS_TO_PREDICT)
			# only create those not in database
			forecast =  forecast.loc[forecast.index > datetime(last_forecast_date.year, last_forecast_date.month, last_forecast_date.day)].copy()
			
			for case in forecast.itertuples():
				#get date
				date = case[0].date()
				current_cases = Case(Reference_ID=state.State_ID, Date=date, Ref_Key=2, Active_cases=int(case.active_predicted), Is_actual=False)
				current_cases.save()

				if current_cases.Active_cases == 0:
					zone_colour = 1
				elif current_cases.Active_cases > 40:
					zone_colour = 3
				else:
					zone_colour = 2

				zone = Zone(Ref_Key=1, Reference_ID=state.State_ID, Date=date, Zone_colour=zone_colour, Is_actual=False)
				zone.save()

