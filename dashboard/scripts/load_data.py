import csv
import pandas as pd
import numpy as np
from datetime import datetime as dt
from django.db.models import Sum
import os
from pathlib import Path

from dashboard.models import Case, Country, District, State, Zone

def run():
	BASE_DIR = Path(__file__).resolve(strict=True).parent.parent.parent
	name_path = os.path.join( BASE_DIR, 'data/state_name.csv')
	s_path = os.path.join( BASE_DIR, 'data/state_fb.csv')
	d_path = os.path.join( BASE_DIR, 'data/district_name.csv')
	c_path = os.path.join(BASE_DIR, 'data/malaysia_cases.csv')
	daily_path =  os.path.join( BASE_DIR, 'data/daily')

	# clean database
	Case.objects.all().delete()
	District.objects.all().delete()
	Country.objects.all().delete()
	State.objects.all().delete()
	Zone.objects.all().delete()

	# # create country
	country, created = Country.objects.get_or_create(Country_ID = 'MYS',  defaults = { 'Country_Name' : 'Malaysia'})
	

	print("Adding state information to database")
	# save state name to database
	with open(name_path, 'r') as states_file:
		csv_r = csv.reader(states_file)
		next(csv_r)
		for r in csv_r: 
			state, created = State.objects.get_or_create(State_ID = r[1], defaults = { 'State_Name' : r[0], 'Country_ID' : country})

	print("Inserting district data to database")
	# save district name to database
	with open(d_path,'r') as districts_file:
		csv_r = csv.reader(districts_file)
		next(csv_r)
		for r in csv_r:
			state = State.objects.get(State_ID = r[2])
			district, created = District.objects.get_or_create(District_ID = r[1], defaults = {'District_Name' : r[0], 'State_ID' : state})

	print("Inserting country cases to database")
	malaysia_case = pd.read_csv(c_path)
	malaysia_case['date'] = pd.to_datetime(malaysia_case['date'], format='%d/%m/%Y').dt.strftime('%Y-%m-%d')
	# malaysia_case = malaysia_case.replace({np.nan: None}, inplace=True)
	malaysia_case = malaysia_case.where(pd.notnull(malaysia_case), None)
	
	# only required if want to save updated infomation to the file
	# malaysia_case.to_csv('data/malaysia_cases.csv', index=False)

	prev = None
	for case in malaysia_case.itertuples():
		 
		# to get daily cases use current cases - yesterday case
		if prev:
			current_cases = Case(Ref_Key=1, Reference_ID=country.Country_ID, Date=case.date, Total_tests=case.total_tests, Total_infected=case.total_infected, Total_recoveries=case.total_recovered, Total_deaths=case.deaths, Active_cases=case.active_cases, Respiratory_aid=case.respiratory_aid, No_of_patient_in_ICU=case.icu, Daily_infected=case.total_infected - prev.Total_infected, Daily_deaths=case.deaths - prev.Daily_deaths, Daily_recoveries=case.total_recovered - prev.Total_recoveries )
		else:  
			current_cases = Case(Ref_Key=1, Reference_ID=country.Country_ID, Date=case.date, Total_tests=case.total_tests, Total_infected=case.total_infected, Total_recoveries=case.total_recovered, Total_deaths=case.deaths, Active_cases=case.active_cases, Respiratory_aid=case.respiratory_aid, No_of_patient_in_ICU=case.icu, Daily_infected=case.total_infected, Daily_deaths=case.deaths, Daily_recoveries=case.total_recovered )

		current_cases.save()
		prev = current_cases

	print("adding district cases to database")
	# load district cases to database
	states = State.objects.all()
	for state in states:
		if state.State_Name == "perlis" or state.State_Name == "labuan" or state.State_Name == "putrajaya":
			continue
		district_case_path = state.State_Name + ".csv"
		district_case_path = os.path.join( daily_path, district_case_path)
		district_case = pd.read_csv(district_case_path)
		district_case['date'] = pd.to_datetime(district_case['date'], format='%d/%m/%Y').dt.strftime('%Y-%m-%d')
		district_case = district_case.where(pd.notnull(district_case), None)
		columns = list(district_case)
		
		print("reading districts cases in " + state.State_Name)
		for case in district_case.itertuples(index=False):
			for i in range(1,len(columns)):
				district = District.objects.get(District_Name=columns[i])
				current_cases = Case(Ref_Key=3, Reference_ID=district.District_ID, Date=case.date, Active_cases=case[i])
				if current_cases.Active_cases != None:
					if current_cases.Active_cases == 0:
						zone_colour = 1
					elif current_cases.Active_cases > 40:
						zone_colour = 3
					else:
						zone_colour = 2

					zone = Zone(Ref_Key=2, Reference_ID=district.District_ID, Date=case.date, Zone_colour=zone_colour)
					zone.save()
				else:
					zone = Zone(Ref_Key=2, Reference_ID=district.District_ID, Date=case.date, Zone_colour=0)
					zone.save()
				current_cases.save()

	print("Inserting state cases to database")
	case_stats = pd.read_csv(s_path)
	case_stats['date'] = pd.to_datetime(case_stats['date'], format='%d/%m/%Y').dt.strftime('%Y-%m-%d')
	# malaysia_case = malaysia_case.replace({np.nan: None}, inplace=True)
	case_stats = case_stats.where(pd.notnull(case_stats), None)

	prev = []
	prev_available = False
	for case in case_stats.itertuples():
		state = State.objects.get(State_Name=case.state_name)

		if prev_available:
			prev_case = prev.pop(0)
			# print(prev_case )
			# print(state.State_Name)
			current_cases = Case(Ref_Key=2, Reference_ID=state.State_ID, Date=case.date, Total_infected=case.total_infected, Daily_infected=case.total_infected - prev_case.Total_infected if case.total_infected != None and prev_case.Total_infected != None and case.total_infected - prev_case.Total_infected > 0 else 0, Total_recoveries=case.total_recovered, Total_deaths=case.deaths, Daily_deaths=case.deaths - prev_case.Total_deaths if case.deaths != None and prev_case.Total_deaths != None and case.deaths - prev_case.Total_deaths > 0 else 0, Active_cases=case.active_cases)
		else:  
			current_cases = Case(Ref_Key=2, Reference_ID=state.State_ID, Date=case.date, Total_infected=case.total_infected, Daily_infected=case.total_infected, Total_recoveries = case.total_recovered, Total_deaths=case.deaths, Daily_deaths=case.deaths, Active_cases=case.active_cases )
		
		if current_cases.Active_cases != None:
			if current_cases.Active_cases == 0:
				zone_colour = 1
			elif current_cases.Active_cases > 40:
				zone_colour = 3
			else:
				zone_colour = 2
			
			zone = Zone(Ref_Key=1, Reference_ID=state.State_ID, Date=case.date, Zone_colour=zone_colour)
			zone.save()

		else:
			zone = Zone(Ref_Key=1, Reference_ID=state.State_ID, Date=case.date, Zone_colour=0)
			zone.save()
		

		current_cases.save()
		prev.append(current_cases)

		if len(prev)== 16:
			prev_available = True

