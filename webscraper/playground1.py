import requests, json, csv
import pandas as pd
from urllib.request import urlopen
from covid19dh import covid19
from datetime import datetime, date, timedelta
from os import path


def getCountryCasesFromAPI():
	url = "https://api.apify.com/v2/datasets/7Fdb90FMDLZir2ROo/items?format=json&clean=1"
	req = requests.get(url)
	with open( 'malaysia_cases.csv', 'w' ) as out_file:
		csv_w = csv.writer( out_file )
		# Write CSV Header, If you dont need that, remove this line
		csv_w.writerow(["date", "total_tests", "total_infected",  "total_recovered", "deaths", "active_cases","respiratory_aid", "inICU"])
		json_text = json.loads(req.text)
		for r in json_text:
			d = datetime.strptime(r.get("lastUpdatedAtApify"), '%Y-%m-%dT%H:%M:%S.%fZ')
			csv_w.writerow([d.strftime('%d/%m/%Y'), r.get("testedTotal", ""), r.get("testedPositive"), r.get("recovered"), r.get("deceased"), r.get("activeCases", ""), r.get("respiratoryAid", ""), r.get("inICU", "")])

	# clean up data 
	data = pd.read_csv("malaysia_cases_api.csv")
	prev = []
	for row in data.itertuples():
		#remove row with the same date
		if prev:
			cur_date = row[1]
			prev_date = prev[1]
			if cur_date == prev_date:
				data.drop(prev[0], axis=0, inplace=True) 
		prev = row
	# only need data from column 2 to 8
	data[list(data)[1:7]] = data[list(data)[1:7]].astype(str)
	data.to_csv('malaysia_cases_api.csv', index=False)   

def getCountryCaseFromDataHub():
	# get from other source
	# url: https://github.com/covid19datahub/COVID19
	x, src = covid19('MY')
	df = pd.DataFrame(x)
	df.rename(columns = {"tests":"total_tests", "confirmed":"total_infected", "recovered":"total_recovered", "hosp":"active_cases", "vent":"respiratory_aid"}, inplace = True)
	df[list(df)[2:8]] = df[list(df)[2:8]].astype(str)
	df.to_csv('malaysia_cases_data.csv', index=False, columns=list(df)[1:8])

def mergeData():
	all_data = pd.read_csv("malaysia_cases_api.csv")
	missing_data = pd.read_csv("malaysia_cases_data.csv")
	all_data.fillna(missing_data)
	all_data.to_csv("merged_malaysia_cases.csv", index=False)

def getLastestCountryCases():
	url = "https://api.apify.com/v2/key-value-stores/6t65lJVfs3d8s6aKc/records/LATEST?disableRedirect=true"
	req = requests.get(url)
	with open( '../data/malaysia_cases.csv', 'a+', newline='' ) as out_file:
		csv_w = csv.writer( out_file )
		json_text = json.loads(req.text)
		# print(json_text['testedPositive'])
		d = datetime.strptime(json_text["lastUpdatedAtApify"], '%Y-%m-%dT%H:%M:%S.%fZ')
		csv_w.writerow([d.strftime('%d/%m/%Y'), "", json_text["testedPositive"], json_text["recovered"], json_text["deceased"], json_text["activeCases"], json_text["respiratoryAid"], json_text["inICU"]])


if __name__ == "__main__":
	# if(path.exists("malaysia_cases_api.csv"))
	# today = date.today()
	# print(today)
	# yesterday = today - timedelta(days=10)
	# print(yesterday)
	# x, src = covid19("MY", start = yesterday)
	# print(x)

	# getLastestCountryCases()

	df = pd.read_csv("../data/malaysia_cases.csv")
	latest_date = df.tail(1).date.item()
	print(latest_date)
	latest_date = datetime.strptime(latest_date,"%Y-%m-%d")
	x, src = covid19("MY", start = latest_date + timedelta(1))
	df = pd.DataFrame(x)
	df.rename(columns = {"tests":"total_tests", "confirmed":"total_infected", "recovered":"total_recovered", "hosp":"active_cases", "vent":"respiratory_aid"}, inplace = True)
	df[list(df)[2:8]] = df[list(df)[2:8]].astype(str)
	# df.to_csv('malaysia_cases_data.csv', index=False, columns=list(df)[1:8])
	with open("../data/malaysia_cases.csv", "a+") as malaysia_cases:
		csv_w = csv.writer(malaysia_cases)
		for row in df.itertuples(index=False):
			# print(row)
			#date, total_tests,total_infected,total_recovered,deaths,active_cases,respiratory_aid,icu
			csv_w.writerow([row.date, row.total_tests, row.total_infected, row.total_recovered, row.deaths, row.active_cases, row.respiratory_aid, row.icu])
