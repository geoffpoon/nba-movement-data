(# =============================================================================
# IMPORTS
# =============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy
import seaborn as sns

import plot_court
import sklearn.model_selection
from sklearn.decomposition import NMF

import subprocess
import os
import simplejson as json
#import ujson as json
from pprint import pprint

# =============================================================================
# 
# =============================================================================

data_path = os.path.abspath(os.path.join(os.getcwd(), os.pardir)) + '/data'

gameId = '0021500663'
data_path = os.path.abspath(os.path.join(os.getcwd(), os.pardir)) + '/data'
json_path = data_path + '/%s.json'%gameId
with open(json_path) as json_dat_file:
    json_dat = json.load(json_dat_file)
    json_dat_file.close()
    
#%%
    
data_path = os.path.abspath(os.path.join(os.getcwd(), os.pardir)) + '/data/shots/shots.csv'
shots_dat = pd.read_csv(data_path)


    
#%%

    
def scrape_player_XYHist_fromJSON(gameId, playersName):
    data_path = os.path.abspath(os.path.join(os.getcwd(), os.pardir)) + '/data'
    json_path = data_path + '/%s.json'%gameId
    with open(json_path) as json_dat_file:
        json_dat = json.load(json_dat)
        json_dat_file.close()
    
    movement_headers = ["team_id", "player_id", 
                        "x_loc", "y_loc", 
                        "radius", 
                        "game_clock", "shot_clock", "quarter", 
                        "game_id", "event_id"]
    
    events = json_dat['events']
    moments = dict()
    
    for event in events:
        event_id = event['eventId']
        movement_dat = event['moments']
        for moment in movement_dat:
#            moment[0] = quarter (1, 2, 3, or 4 ...)
#            moment[1] = time of event in miliseconds (includes date)
#            moment[2] = number is the number of seconds left in the quarter
#            moment[3] = number of seconds left on the shot clock
#            moment[4] = UNKNOWN
#            moment[5] = list of positions of each player
#                       (colums = teamID, playerID, x, y, z; 
#                       where 0th entry is the ball (teamId = playerID = -1)
#                       where z=0 except for the ball)
            for teamID, playerID, x, y, z in moment[5][1:]:
                if playersName

