#get from twitter
# import GetOldTweets3 as got
# import pandas as pd

# # Parameters: (list of twitter usernames), (max number of most recent tweets to pull from)
# def username_tweets_to_csv(username, count):
# 	# Creation of query object
# 	# tweetCriteria = got.manager.TweetCriteria().setUsername(username).setSince("2020-04-28")
# 	tweetCriteria = got.manager.TweetCriteria().setUsername(username).setMaxTweets(10)
# 	# Creation of list that contains all tweets
# 	tweets = got.manager.TweetManager.getTweets(tweetCriteria)

# 	# Creating list of chosen tweet data
# 	user_tweets = [[tweet.date, tweet.text] for tweet in tweets if "Status terkini COVID-19 setakat" in tweet.text]
# 	print(user_tweets[0][1].split("\n"))
# 	print(user_tweets)

# 	# Creation of dataframe from tweets list
# 	tweets_df = pd.DataFrame(user_tweets, columns = ['Datetime', 'Text'])

# 	# Converting dataframe to CSV
# 	tweets_df.to_csv('{}-{}k-tweets.csv'.format(username, int(count/1000)), sep=',')

# # Input username(s) to scrape tweets and name csv file
# # Max recent tweets pulls x amount of most recent tweets from that user
# username = 'DGHisham'
# count = 1000

# # Calling function to turn username's past x amount of tweets into a CSV file
# username_tweets_to_csv(username, count)


#get from website
import requests, json, csv
import pandas as pd
from urllib.request import urlopen
from covid19dh import covid19
from datetime import datetime
import numpy as np

# import csv

# url = "https://api.apify.com/v2/key-value-stores/6t65lJVfs3d8s6aKc/records/LATEST?disableRedirect=true" 
url = "https://api.apify.com/v2/datasets/7Fdb90FMDLZir2ROo/items?format=json&clean=1"
# page = urlopen(url)
# json = page.read().decode("utf-8")
# print(json)


req = requests.get(url)
with open( 'malaysia_cases.csv', 'w' ) as out_file:
	csv_w = csv.writer( out_file )
	# Write CSV Header, If you dont need that, remove this line
	csv_w.writerow(["date", "total_tests", "a",  "total_recovered", "deaths", "active_cases","respiratory_aid", "inICU"])
	x = json.loads(req.text)
	for r in x:
		print(r)
		d = datetime.strptime(r.get("lastUpdatedAtApify"), '%Y-%m-%dT%H:%M:%S.%fZ')
		csv_w.writerow([d.strftime('%d/%m/%Y'), r.get("testedTotal", ""), r.get("testedPositive"), r.get("recovered"), r.get("deceased"), r.get("activeCases", ""), r.get("respiratoryAid", ""), r.get("inICU", "")])

# clean up data 
data = pd.read_csv("malaysia_cases.csv")
prev = []
for row in data.itertuples():
	#remove row with the same date
	if prev:
		cur_date = row[1]
		prev_date = prev[1]
		if cur_date == prev_date:
			data.drop(prev[0], axis=0, inplace=True) 
	prev = row
data[list(data)[1:7]] = data[list(data)[1:7]].astype(str)
data.fillna(np.nan)
data.to_csv('malaysia_cases_cleaned.csv', index=False)   


# get from other source
# url: https://github.com/covid19datahub/COVID19
x, src = covid19('MY')
df = pd.DataFrame(x)
# df.drop('id', axis = 1, inplace = True)
# df.drop(list(df)[8:], axis = 1, inplace = True) 
df.rename(columns = {"tests":"total_tests", "confirmed":"total_infected", "recovered":"total_recovered", "hosp":"active_cases", "vent":"respiratory_aid"}, inplace = True)
df[list(df)[2:8]] = df[list(df)[2:8]].astype(str)
# df.replace("nan", np.nan, inplace=True)
df.to_csv('malaysia_cases_data.csv', index=False, columns=list(df)[1:8])

all_data = pd.read_csv("malaysia_cases_cleaned.csv")
# print(all_data.dtypes)
update = pd.read_csv("malaysia_cases_data.csv")
# update.dropna(subset = ["total_tests"], inplace=True)

#print(update.dtypes)
# merged = pd.merge(all_data, update, on='date', how='outer')
# merged = all_data.combine_first(update)
# all_data.fillna(update)
# all_data.to_csv("merged_malaysia_cases.csv", index=False)
# fill_dict = all_data.set_index('date')['total_tests'].to_dict()
# all_data['total_tests'] = all_data['total_tests'].replace('nan', all_data['date'].map(fill_dict))
update.replace("nan", "", inplace=True)
update.to_csv("merged_malaysia_cases.csv", index=False)
