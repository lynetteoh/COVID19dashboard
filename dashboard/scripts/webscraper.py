# script for daily update
from bs4 import BeautifulSoup
# from urllib.request import urlopen
import requests
import csv 
import time
from datetime import datetime, timedelta
import os
from pathlib import Path

from dashboard.models import Case, Country, District, State, Zone

def run():
	scrapeStateStats()

def scrapeCountryStats():
	# scrape country level
	country_url = "http://covid-19.livephotos123.com/en"
	# page = urlopen(url)
	# html = page.read().decode("utf-8")
	# soup = BeautifulSoup(html, "html.parser")

	country_page = requests.get(country_url)
	country_soup = BeautifulSoup(country_page.content, "html.parser")
	# print(soup.prettify())

	# subtitle = soup.find_all('span', class_='sub-title')
	# print(soup.find_all('div', class_='col-xs-12 col-md-6 text-center'))


	# get country level covid information
	section = country_soup.find('div', class_='col-xs-12 col-md-6 text-center')
	types = section.find_all('div')
	# print(types)
	all_info = {}
	for t in types:
		info = t.find_all('span')
		# print(info)

		if info:
			# get type
			key = info[0].get_text()
			cases_num = []
			for i in range(1, len(info)):
				#get changes
				cases_num.append(info[i].get_text())

			# todo : use pandas and save to csv file
			all_info[key] = cases_num

	print(all_info)

	

def scrapeStateStats():
	BASE_DIR = Path(__file__).resolve(strict=True).parent.parent.parent
	name_path = os.path.join( BASE_DIR, 'data/state_name.csv')
	s_path = os.path.join( BASE_DIR, 'data/state_fb.csv')
	c_path = os.path.join(BASE_DIR, 'data/state_stats_scrapped.csv')

	updated = False
	with open(s_path, "r") as f:
		reader = list(csv.reader(f))
		today = time.strftime('%d/%m/%Y')
		if reader[len(reader)-1][0] == today:
			print("Updated state_fb.csv as of " + reader[len(reader)-1][0])
			updated = True
	
	state_stats_updated = False

	with open(c_path, "r") as fs:
			reader_fs = list(csv.reader(fs))
			today = time.strftime('%d/%m/%Y')
			if reader_fs[len(reader_fs)-1][0] == today:
				print("Updated state_stats_scrapped.csv as of " + reader_fs[len(reader_fs)-1][0])
				state_stats_updated = True

	if not updated:
		# get state info
		with open(name_path, "r") as state_name:
			reader = csv.reader(state_name)
			next(reader) # skip first row
			for row in reader:
				print(row)
				# get state cases
				case_type = ["Active_14", "In treatment", "Deaths"]
				state_url = "https://newslab.malaysiakini.com/covid-19/en/state/" + row[0]
				state_page = requests.get(state_url)
				state_soup = BeautifulSoup(state_page.content, "html.parser")
				date_soup = state_soup.find('div', class_="jsx-3636536621 uk-text-large")
				d = date_soup.get_text().strip("As of ").strip(" PM").strip(" AM")
				latest_date = datetime.strptime(d, '%b %d, %Y %H:%M')
				date_updated = latest_date.strftime('%d/%m/%Y')
				state_info = {}
				infected = state_soup.find("div", class_="jsx-3636536621 uk-text-center uk-text-large")
				state_info['Infected'] = int(infected.find("strong", class_= "jsx-3636536621" ).get_text().replace(",", ""))
				stats = state_soup.find('ul', class_="jsx-3636536621 uk-grid uk-grid-divider uk-grid-small uk-flex-center uk-child-width-1-2 uk-child-width-1-3@s uk-child-width-1-4@m uk-child-width-1-5@l uk-margin")
				# get info
				stats = stats.find_all('span', class_= "jsx-3636536621 uk-heading-medium")
				for i in range(0, len(case_type)):
					state_info[case_type[i]] = int(stats[i].get_text().replace(",", ""))

				state_info["Recovered"] = state_info.get("Infected") - state_info.get("In treatment") - state_info.get("Deaths")
				print(state_info)

				
				with open(s_path, "a+") as state_stats:
					csv_w = csv.writer(state_stats)
					csv_w = csv_w.writerow([date_updated, row[0], state_info.get("Infected"), state_info.get("Recovered"), state_info.get("Deaths"), state_info.get("In treatment")])
				
				state = State.objects.get(State_Name=row[0])
				date = latest_date.strftime('%Y-%m-%d')
				yesterday = latest_date - timedelta(1)
				try:
					case = Case.objects.get(Ref_Key=2, Reference_ID = state.State_ID, Date=yesterday)
				except Case.DoesNotExist:
					case = None
				if not case:
					state_case, created = Case.objects.get_or_create(Reference_ID = state.State_ID, Date=date,  Is_actual=True, defaults = { 'Ref_Key': 2, 'Total_infected' : state_info.get("Infected"), 'Total_deaths': state_info.get("Deaths"),'Total_recoveries': state_info.get("Recovered"), 'Active_cases': state_info.get("In treatment")})
				else:
					state_case, created = Case.objects.get_or_create(Reference_ID = state.State_ID, Date=date, Is_actual=True, defaults = { 'Ref_Key': 2, 'Total_infected' : state_info.get("Infected"), 'Total_deaths': state_info.get("Deaths"),'Total_recoveries': state_info.get("Recovered"), 'Active_cases': state_info.get("In treatment"), 'Daily_infected': state_info.get("Infected") - case.Total_infected if int(state_info.get("Infected")) - case.Total_infected > 0 else 0, 'Daily_deaths' : state_info.get("Deaths") - case.Total_deaths if state_info.get("Deaths") - case.Total_deaths > 0 else 0})
				
				if not created:
					state_case.Total_infected = state_info.get("Infected")
					state_case.Total_deaths = state_info.get("Deaths")
					state_case.Total_recoveries = state_info.get("Recovered")
					state_case.Active_cases = state_info.get("In treatment")
					if case:
						state_case.Daily_deaths = state_info.get("Deaths") - case.Total_deaths if state_info.get("Deaths") - case.Total_deaths > 0 else 0
						state_case.Daily_infected = state_info.get("Infected") - case.Total_infected if state_info.get("Infected") - case.Total_infected > 0 else 0
					state_case.save()
				
				if state_case.Active_cases == 0:
					zone_colour = "1"
				elif state_case.Active_cases > 40:
					zone_colour = "3"
				else:
					zone_colour = "2"

				zone = Zone(Ref_Key=1, Reference_ID=state.State_ID, Date=state_case.Date, Zone_colour=zone_colour)
				zone.save()

				if not state_stats_updated:
					date = latest_date.strftime('%d/%m/%Y')
					with open(c_path, "a+") as state_stats:
						csv_w = csv.writer(state_stats)
						csv_w = csv_w.writerow([date, row[0], state_info.get("Infected"), state_info.get("Recovered"), state_info.get("Deaths"), state_info.get("In treatment")])

				
			

