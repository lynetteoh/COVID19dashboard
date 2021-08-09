from django.shortcuts import render, HttpResponse
from dashboard.models import Case, State, Zone, District
import json
from django.core import serializers
from datetime import timedelta, datetime
from django.template.loader import get_template 
from .utils import *

from pathlib import Path
import os

# Create your views here.

# helper functions
def getStatesCases(states_cases):
	states_list = {}
	for state_case in states_cases:
		s_name = State.objects.get(State_ID=state_case.Reference_ID)
		s_name = s_name.State_Name.replace("-", " ").capitalize()
		states_list[s_name] = state_case
	return states_list

# render functions

def index(request): 
	# get all cases from country level to draw chart
	all_cases = Case.objects.filter(Reference_ID="MYS", Is_actual=True).order_by('Date')
	all_cases = serializers.serialize('json', all_cases)

	# get malaysia today case
	mys_cases = Case.objects.filter(Reference_ID="MYS", Is_actual=True)[0]
	print(mys_cases)

	#get yesterday case
	yesterday_cases = Case.objects.filter(Reference_ID="MYS", Is_actual=True)[1]

	states_cases = Case.objects.filter(Ref_Key=2, Is_actual=True)[0:16]
	states_list = getStatesCases(states_cases)

	# print(states_list)

	zones = Zone.objects.filter(Ref_Key=1, Is_actual=True, Date=states_cases[0].Date).order_by('-Zone_colour')
	map_zones = serializers.serialize('json', zones)
	zones_list = {}
	for zone in zones:
		s_name = State.objects.get(State_ID=zone.Reference_ID)
		s_name = s_name.State_Name.replace("-", " ").capitalize()
		zones_list[s_name] = zone
	# print(zones)

	context ={
	   	"total_infected" : mys_cases.Total_infected,
	   	"daily_infected" : mys_cases.Daily_infected,
	  	"total_recovery" : mys_cases.Total_recoveries,
	   	"daily_recovery" : mys_cases.Daily_recoveries,
	   	"total_death" : mys_cases.Total_deaths,
	   	"daily_death":mys_cases.Daily_deaths,
	   	"test" : mys_cases.Total_tests if mys_cases.Total_tests != None else "-",
		"daily_test": mys_cases.Total_tests - yesterday_cases.Total_tests if mys_cases.Total_tests != None and yesterday_cases.Total_tests  != None else "-",
	   	"ventilator" : mys_cases.Respiratory_aid,
		"daily_ventilator": mys_cases.Respiratory_aid - yesterday_cases.Respiratory_aid,
	   	"ICU" : mys_cases.No_of_patient_in_ICU,
	   	"daily_ICU" : mys_cases.No_of_patient_in_ICU - yesterday_cases.No_of_patient_in_ICU,
		"date":mys_cases.Date,
		"treatment" : mys_cases.Active_cases, 
		"daily_treatment": mys_cases.Active_cases - yesterday_cases.Active_cases,
		"states":states_list,
		"all_cases":json.dumps(all_cases),
		"map_zones":json.dumps(map_zones),
		"zones":zones_list,
		"state_date": states_cases[0].Date,
	}

	return render(request, "dashboard.html", context)


def state(request, sName):
	# name received is capitalized and without hyphen
	sName = sName.lower().replace(" ", "-")
	
	state = State.objects.get(State_Name=sName)
	case = Case.objects.filter(Reference_ID=state.State_ID, Is_actual=True)[0]

	districts = District.objects.filter(State_ID=state.State_ID)

	districts_list = []
	for district in districts:
		district_Name = district.District_Name.capitalize()
		district_case = Case.objects.filter(Ref_Key=3, Reference_ID=district.District_ID, Is_actual=True)[0]
		zone_colour = Zone.objects.filter(Ref_Key=2, Reference_ID=district.District_ID, Is_actual=True)[0]
		# print(zone_colour)
		districts_list.append([district_Name, district_case, zone_colour])
	
	districts_list.sort(key=lambda x:x[2].Zone_colour, reverse=True)
	# print(districts_list)

	context={
		"state":state.State_Name.replace("-"," ").capitalize(),
		"confirmed": case.Total_infected if case.Total_infected != None else "-" ,
		"recovered":case.Total_recoveries if case.Total_recoveries != None else "-",
		"deaths": case.Total_deaths if case.Total_deaths != None else "-",
		"treatment": case.Active_cases if case.Active_cases != None else "-",
		"daily_infected":case.Daily_infected if case.Daily_infected != None else "-",
		"daily_deaths":case.Daily_deaths if case.Daily_deaths != None else "-",
		"date": case.Date,
		"districts": districts_list,
		"district_date": districts_list.pop()[1].Date if len(districts_list) > 0 else None
	}

	return render(request, "state.html", context)  	

def report(request):

	if request.method == 'POST':
		start_date = '2020-03-13'
		d = request.POST.get('endDate')
		end_date = datetime.strptime(d, '%m/%d/%Y').date()
		# end_date = end_date.strftime('%Y-%m-%d')

		latest_available_date = Case.objects.filter(Ref_Key=1, Reference_ID="MYS", Is_actual=True)[0].Date

		if end_date > latest_available_date:
			end_date = latest_available_date
		
		#country cases as of date
		mys_case = Case.objects.get(Ref_Key=1, Reference_ID="MYS", Date=end_date, Is_actual=True)

		#state cases as of date
		states_cases = Case.objects.filter(Ref_Key=2, Date=end_date, Is_actual=True)
		states_list = getStatesCases(states_cases)


		states = State.objects.all()
		districts_list = []
		
		# district cases as of date
		for state in states:
			s_name = state.State_Name.replace("-", " ").capitalize()
			districts = District.objects.filter(State_ID=state.State_ID)
			for district in districts:
				district_Name = district.District_Name.capitalize()
				district_case = Case.objects.get(Ref_Key=3, Reference_ID=district.District_ID, Date=end_date, Is_actual=True)
				districts_list.append([s_name, district_Name, district_case])
		
		# print(districts_list)

		# country daily cases
		all_cases = Case.objects.filter(Reference_ID="MYS", Date__range = (start_date, end_date), Is_actual=True).order_by('Date')
		cases = serializers.serialize('json', all_cases)

		# state daily cases
		s_cases = {}
		for state in states:
			s_case = Case.objects.filter(Reference_ID=state.State_ID, Date__range = (start_date, end_date), Is_actual=True).order_by('Date')
			s_name = state.State_Name.replace("-", " ").capitalize()
			s_cases[s_name] = s_case

		zones = Zone.objects.filter(Ref_Key=1, Date = end_date, Is_actual=True)
		map_zones = serializers.serialize('json', zones)

		context={
			"total_infected" : mys_case.Total_infected,
			"daily_infected" : mys_case.Daily_infected,
			"total_recovery" : mys_case.Total_recoveries,
			"daily_recovery" : mys_case.Daily_recoveries,
			"total_death" : mys_case.Total_deaths,
			"daily_death":mys_case.Daily_deaths,
			"treatment" :mys_case.Active_cases,
			"daily_treatment" :mys_case.Active_cases,
			"test" :  mys_case.Total_tests if mys_case.Total_tests else "-",
			"ventilator" : mys_case.Respiratory_aid,
			"ICU" : mys_case.No_of_patient_in_ICU,
			"date":mys_case.Date, 
			"all_cases": all_cases,
			"cases" : json.dumps(cases),
			"states_cases" : states_list,
			"districts" : districts_list, 
			"daily_s_cases": s_cases, 
			"map_zones" : json.dumps(map_zones)
		}

		# yesterday = datetime.strptime(end_date, "%Y-%m-%d") - timedelta(1)
		yesterday = end_date - timedelta(1)
		yesterday_case =  Case.objects.get(Ref_Key=1, Reference_ID="MYS", Date=yesterday, Is_actual=True)

		if yesterday_case != None: 
			context["daily_test"] =   mys_case.Total_tests - yesterday_case.Total_tests if mys_case.Total_tests != None and yesterday_case.Total_tests != None else "-"
			context["daily_ventilator"] = mys_case.Respiratory_aid - yesterday_case.Respiratory_aid
			context["daily_ICU"] = mys_case.No_of_patient_in_ICU - yesterday_case.No_of_patient_in_ICU
			context["daily_treatment"] = mys_case.Active_cases - yesterday_case.Active_cases
		
		return render(request, "report.html", context)

def forecast(request):
	# days to predict
	days_predict = 7

	# get file directory
	BASE_DIR = Path(__file__).resolve(strict=True).parent.parent
	s_path = os.path.join( BASE_DIR, 'data/state_fb.csv')
	c_path = os.path.join(BASE_DIR, 'data/malaysia_cases.csv')
	models_path = os.path.join(BASE_DIR, 'models')

	latest_c_date = mys_cases = Case.objects.filter(Reference_ID="MYS", Is_actual=True)[0].Date
	latest_s_date = Case.objects.filter(Ref_Key=2, Is_actual=True)[0].Date

	states = State.objects.all()
	forecast_country_confirmed(BASE_DIR, models_path, c_path, latest_c_date, days_predict)
	forecast_state_active(BASE_DIR, states, models_path, s_path, latest_s_date, days_predict)
	
	#get Malaysia case
	all_cases = Case.objects.filter(Reference_ID="MYS", Is_actual=True).order_by('Date')
	all_cases = serializers.serialize('json', all_cases) 
	forecast_my = Case.objects.filter(Reference_ID="MYS", Is_actual=False)
	forecast_my_cases = forecast_my.order_by('Date')
	forecast_my_cases = serializers.serialize('json', forecast_my_cases)

	state_zones = {}
	
	for state in states:
		actual_case = Zone.objects.filter(Reference_ID=state.State_ID, Date=latest_s_date, Is_actual=True)
		s_case = Zone.objects.filter(Reference_ID=state.State_ID, Date__gt=latest_s_date, Is_actual=False)
		s_name = state.State_Name.replace("-", " ").capitalize()
		state_zones[s_name] = s_case.union(actual_case).order_by('Date')

	

	print(state_zones)
	context= {
		"all_cases":json.dumps(all_cases), 
		"forecast_my": json.dumps(forecast_my_cases),
		"actual_date" : latest_s_date, 
		"state_zones" : state_zones, 
		"last_forecast_date" : forecast_my[0].Date

	}
	return render(request, "forecast.html", context)

	
