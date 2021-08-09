# script for daily update
from bs4 import BeautifulSoup
# from urllib.request import urlopen
import requests
import csv 

#scrape country level
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

# get state info

#scrape state and district level 
with open("state_name.csv", "r") as state_name:
	reader =csv.reader(state_name)
	next(reader) # skip first row
	for row in reader:
		print(row)
	case_type = ["Infected", "In treatment", "Deaths"]
	state_url = "https://newslab.malaysiakini.com/covid-19/en/state/" + "kedah"
	state_page = requests.get(state_url)
	state_soup = BeautifulSoup(state_page.content, "html.parser")
	# get the list for total state info
	stats = state_soup.find('ul', class_="jsx-3636536621 uk-grid uk-grid-divider uk-grid-small uk-flex-center uk-child-width-1-2 uk-child-width-1-3@s uk-child-width-1-4@m uk-child-width-1-5@l uk-margin")
	stats = stats.find_all('span', class_= "jsx-3636536621 uk-heading-medium")
	state_info = {}
	for i in range(0, len(case_type)):
		state_info[case_type[i]] = int(stats[i].get_text())

	state_info["Recovered"] = state_info.get("Infected") - state_info.get("In treatment") - state_info.get("Deaths")
	print(state_info)
