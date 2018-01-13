import requests
import pandas as pd
import sys
import json

###################################

# Requesting Player ID (or PERSON_ID) from stats.nba.com API
# so that we can easily find data regarding a specific player.
# We can only pull data based on a player's unique ID#, not their name

PARAMS = {'LeagueID': '00', 
          'Season': '2013-18',
          'IsOnlyCurrentSeason': '0',
         }

# set headers, otherwise the API might not work
HEADERS = {'user-agent': ('Mozilla/5.0 (Macintosh; Intel Mac OS X 10_10_5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/45.0.2454.101 Safari/537.36'),
           'referer': 'http://stats.nba.com/scores/'
          }


# NOTE: This API is not documented well and is poorly maintained. 
# Let's hope they don't change the endpoints and/or required parameters
r = requests.get('http://stats.nba.com/stats/commonallplayers', 
                 params=PARAMS, headers=HEADERS)


# And let's organize this data into a nice Pandas data frame

playerInfo_json_headers = r.json()['resultSets'][0]['headers']
playerInfo_json_content = r.json()['resultSets'][0]['rowSet']

playerInfo_df = pd.DataFrame(playerInfo_json_content, columns=playerInfo_json_headers)

###################################

# player_id=0 : not specifying PLAYER shooting
# team_id=0 : not specifying TEAM of shooter
# opp_team_id=0 : not specifying OPPOSING TEAM

player_name = ''
if player_name in playerInfo_df.DISPLAY_FIRST_LAST:
    player_id = int(playerInfo_df[playerInfo_df.DISPLAY_FIRST_LAST == player_name].PERSON_ID)
else:
    player_id = 0
team_id = 0
opp_team_id = 0
# season = 2016

# game_id='' : not specifying GAME ID
# season_type : either 'Regular Season', 'Playoffs', 'Pre Season', or 'All Star'
game_id = ''
season_type = 'Regular Season'


#---------------------------------
for season in range(2003, 1990, -1):
	season_string = str(season) + '-' + str(season+1)[2:]
	print(season_string, ' START')

	PARAMS = {'Period': 0, 
	          'VsConference': '', 
	          'LeagueID': '00', 
	          'LastNGames': '0', 
	          'TeamID': str(team_id), 
	          'Position': '', 
	          'Location': '',
	          'Outcome': '',
	          'ContextMeasure': 'FGA',
	          'DateFrom': '',
	          'StartPeriod': '',
	          'DateTo': '',
	          'OpponentTeamID': str(opp_team_id),
	          'ContextFilter': '',
	          'RangeType': '',
	          'Season': season_string,
	          'AheadBehind': '',
	          'PlayerID': str(player_id),
	          'EndRange': '',
	          'VsDivision': '',
	          'PointDiff': '',
	          'RookieYear': '',
	          'GameSegment': '',
	          'Month': '0',
	          'ClutchTime': '',
	          'StartRange': '',
	          'EndPeriod': '',
	          'SeasonType': season_type,
	          'SeasonSegment': '',
	          'GameID': str(game_id),
	          'PlayerPosition': ''
	         }

	# set headers, otherwise the API might not work
	HEADERS = {'user-agent': ('Mozilla/5.0 (Macintosh; Intel Mac OS X 10_10_5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/45.0.2454.101 Safari/537.36'),
	           'referer': 'http://stats.nba.com/scores/'
	          }

	# Requesting data from stats.nba.com with the endpoint /stats/shotchartdetail
	# This gives us information on the matchup, game situtation, and xy-location
	# of every shot that we requested (falls under our filters)


	# NOTE: This API is not documented well and is poorly maintained. 
	# Let's hope they don't change the endpoints and/or required parameters
	r = requests.get('http://stats.nba.com/stats/shotchartdetail', 
	                 params=PARAMS, headers=HEADERS)

	print(season_string, ' DONE')