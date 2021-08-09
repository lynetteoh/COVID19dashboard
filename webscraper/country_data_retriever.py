import requests, json, csv
from urllib.request import urlopen
from datetime import datetime
from os import path


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
	getLastestCountryCases()
