import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib import  offsetbox as osb
from matplotlib.cbook import get_sample_data
from matplotlib._png import read_png
from io import BytesIO, StringIO
from PIL import Image

import scipy
import numpy as np
import seaborn as sns
import pandas as pd

import plot_court
import lgcp_func as lgcp
import nmf_func as nmf
import sklearn.model_selection as sklearnMS
import pickle


import requests

import sys
import subprocess
import os
import simplejson as json
from pprint import pprint

from cache_func import cached

# =============================================================================
# 
# =============================================================================

#shots_path = os.path.abspath(os.path.join(os.getcwd(), os.pardir)) + '/data/shots/shots.csv'
#shots_dat = pd.read_csv(shots_path)
#
#headers_2015 = ['GRID_TYPE', 'GAME_ID', 'GAME_EVENT_ID', 
#                   'PLAYER_ID', 'PLAYER_NAME', 'TEAM_ID', 'TEAM_NAME', 
#                   'PERIOD', 'MINUTES_REMAINING', 'SECONDS_REMAINING', 
#                   'EVENT_TYPE', 'ACTION_TYPE', 'SHOT_TYPE',
#                   'SHOT_ZONE_BASIC', 'SHOT_ZONE_AREA', 
#                   'SHOT_ZONE_RANGE', 'SHOT_DISTANCE',
#                   'LOC_X', 'LOC_Y', 
#                   'SHOT_ATTEMPTED_FLAG', 'SHOT_MADE_FLAG', 
#                   'GAME_DATE', 'HTM', 'VTM']       # HTM = Home team, VTM = Visiting team
#
#df = pd.DataFrame(shots_dat, 
#                  columns = ['PLAYER_ID', 'PLAYER_NAME', 
#                             'TEAM_ID', 'TEAM_NAME', 'ACTION_TYPE',
#                             'SHOT_DISTANCE', 'SHOT_TYPE', 
#                             'LOC_X', 'LOC_Y', 
#                             'SHOT_MADE_FLAG'])

# use df.COLUMN.unique() to get list of unique value for column

#%%
# =============================================================================
# FUNCTIONS: Extract/clean data
# =============================================================================


#@cached('train_test_split_shotInfo_cache.pickle')
def train_test_split_player_shotInfo(season_string, randSeed = 546682):
    shots_path = os.path.abspath(os.path.join(os.getcwd(), os.pardir)) + '/data/shots/shots_data_%s.json'%season_string
    with open(shots_path, 'r') as json_data:
        d = json.load(json_data)
        json_data.close()
        
    shots_dat = pd.DataFrame(d['resultSets'][0]['rowSet'], 
                             columns=d['resultSets'][0]['headers'])
    
#    headers_2015 = ['GRID_TYPE', 'GAME_ID', 'GAME_EVENT_ID', 
#                       'PLAYER_ID', 'PLAYER_NAME', 'TEAM_ID', 'TEAM_NAME', 
#                       'PERIOD', 'MINUTES_REMAINING', 'SECONDS_REMAINING', 
#                       'EVENT_TYPE', 'ACTION_TYPE', 'SHOT_TYPE',
#                       'SHOT_ZONE_BASIC', 'SHOT_ZONE_AREA', 
#                       'SHOT_ZONE_RANGE', 'SHOT_DISTANCE',
#                       'LOC_X', 'LOC_Y', 
#                       'SHOT_ATTEMPTED_FLAG', 'SHOT_MADE_FLAG', 
#                       'GAME_DATE', 'HTM', 'VTM']       # HTM = Home team, VTM = Visiting team
    
    df = pd.DataFrame(shots_dat, 
                      columns = ['PLAYER_ID', 'PLAYER_NAME', 
                                 'TEAM_ID', 'TEAM_NAME', 'ACTION_TYPE',
                                 'SHOT_DISTANCE', 'SHOT_TYPE', 
                                 'LOC_X', 'LOC_Y', 
                                 'SHOT_ATTEMPTED_FLAG', 'SHOT_MADE_FLAG'])
    
    num_players = 300
    top_players_shotNum = df.PLAYER_NAME.value_counts()[:num_players]
    top_players_nameList = top_players_shotNum.index.tolist()

    train_players_df = {}
    test_players_df = {}
    playersID = {}
    for i, player in enumerate(set(top_players_nameList)):  
        player_df = df[df.PLAYER_NAME == player]
        train_players_df[player], test_players_df[player] = \
                sklearnMS.train_test_split(player_df, test_size=0.2, random_state=randSeed)
        playersID[player] = player_df.PLAYER_ID.unique()[0]
        
    return train_players_df, test_players_df, top_players_nameList, playersID

#######################

def gen_players_shotHist(players_df, top_players_nameList, flag='SHOT_ATTEMPTED_FLAG'):
    bins, binRange = ([25,18], [[-250,250], [-47.5,312.5]])
    
    player_shotHist = {}
    for i, player in enumerate(top_players_nameList):  
        temp = players_df[player]
        hist2d, xedges, yedges, binnumber = scipy.stats.binned_statistic_2d(temp.LOC_X, temp.LOC_Y, 
                                                                            temp[flag],
                                                                            statistic='sum',
                                                                            bins=bins, 
                                                                            range=binRange)
        player_shotHist[player] = hist2d.flatten() 
    return player_shotHist, (bins, binRange, xedges, yedges, binnumber)
        
    
# =============================================================================
# FUNCTIONS: Plot histograms
# =============================================================================    
        
    
def acquire_playerPic(PlayerID):
    url = "http://stats.nba.com/media/players/230x185/"+str(PlayerID)+".png"
    response = requests.get(url)
    img = Image.open(BytesIO(response.content))
    return img
        


def plot_shotHist(player_shotHist, player, binDat, fileName, 
                  title='', norm_Opt='linear', plot_size=(5,5)):   
    bins, binRange, xedges, yedges, binnumber = binDat
    Xn_v = player_shotHist
    temp = np.array(Xn_v, dtype='float')/np.sum(Xn_v)
    shotHist_v = np.reshape(temp, bins)
    ##########
    extent = np.min(xedges), np.max(xedges), np.max(yedges), np.min(yedges)
    
    fig = plt.figure(figsize=plot_size)
    ax = plt.axes()
    
    plot_court.draw_court(outer_lines=True, lw=1.5)
    
    
    cmap = plt.cm.magma_r
    if norm_Opt == 'log':
        plt.imshow(shotHist_v.T, cmap=cmap, 
                   norm=colors.LogNorm(vmin=1e-4, vmax=1e-1),
                   alpha=.85, extent=extent)
    else:
        plt.imshow(np.ma.masked_where(shotHist_v.T == 0., shotHist_v.T), 
                   cmap=cmap, alpha=.85, extent=extent)
    
    ax.set_xlim([-300,300])
    ax.set_ylim([-100,500])
    ax.grid('off')
    ax.axis('off')
    ax.set_title('%s: %s'%(player, title), fontsize=15)
#    plt.axis('off')
    fig.tight_layout()
    fig.savefig(fileName, dpi=700)


#%%
    

# =============================================================================
# 
# =============================================================================

def gen_shotHist_plots(player, phi=30, seed=546682):
    dirName = 'SHOT_ATTEMPTED_FLAG/shotHist_LGCP_phi%d_seed%d/'%(phi,seed)
    fileName = 'norm_lambda_%s.txt'%player
    
    outfileName = 'lgcp_shotHist_%s.png'%player
    plot_shotHist(np.loadtxt(dirName + fileName), player, binDat, 
                  outfileName, title='LGCP', norm_Opt='log')
    
    outfileName = 'raw_shotHist_%s.png'%player
    plot_shotHist(players_shotHist_train[player], player, binDat, 
                  outfileName, title='raw', norm_Opt='log')


# =============================================================================
# 
# =============================================================================



# =============================================================================
# 
# =============================================================================


if __name__ == '__main__':
    
    #arg1: phi (the spatial correlation length of shot behavior)
    #arg2: where to start on top_players_nameList
    
    phi = float(sys.argv[1])
    if len(sys.argv) >= 3:
        i = int(sys.argv[2])
    else:
        i = 0
    if len(sys.argv) >= 4:
        flag_name = str(sys.argv[3])
    else:
        flag_name = 'SHOT_ATTEMPTED_FLAG'
    if len(sys.argv) >= 5:
        season_string = str(sys.argv[4])
    else:
        season_string = '2016-17'
    
    ###################################################
    

    randSeed = 546682

    train_players_df, test_players_df, top_players_nameList, playersID = \
            train_test_split_player_shotInfo(season_string, randSeed = randSeed)
    playersName = dict((v,k) for k,v in playersID.items())
    # playersID-- key: name, value: ID
    # playersName-- key: ID, value: name
    
    
            
    players_shotHist_train, binDat = gen_players_shotHist(train_players_df, 
                                                          top_players_nameList,
                                                          flag=flag_name)
    players_shotHist_test, binDat = gen_players_shotHist(test_players_df, 
                                                         top_players_nameList,
                                                         flag=flag_name)
    bins, binRange, xedges, yedges, binnumber = binDat
    
    

    
    lgcp.run(top_players_nameList[i:], players_shotHist_train, 
             binDat, randSeed, 
             phi2=phi**2, flag=flag_name)
    
    LL = np.zeros((len(top_players_nameList), np.prod(bins)))
    directory = flag_name + '/shotHist_LGCP_phi%d_seed%d'%(phi, randSeed)
    for i, player in enumerate(top_players_nameList):
        lambdaN_v = np.loadtxt(directory + '/lambda_%s.txt'%(player))
        norm_lambdaN_v = lambdaN_v / np.sum(lambdaN_v)
        LL[i,:] = norm_lambdaN_v[:]
    

    

    n_features = 10
    W_norm, H_norm = nmf.run(LL, n_features=n_features)

    np.savetxt('W_norm_%s.txt' %flag_name, W_norm)
    np.savetxt('H_norm_%s.txt' %flag_name, H_norm)
    with open('top_players_nameList.pickle', 'wb') as fp:
        pickle.dump(top_players_nameList, fp)

    # for i in range(n_features):
    #     plot_shotHist(H_norm[i,:], 'NMF %d'%i, binDat, 'shotMade_basisVec_%d.png'%i, 
    #                   title='', norm_Opt='linear', plot_size=(5,5))



#bins, binRange = ([25,18], [[-250,250], [-47.5,312.5]])
#temp = df[df.PLAYER_NAME == player]
#
#
#
#sigma2 = 1e3