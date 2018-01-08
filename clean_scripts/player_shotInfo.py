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
import sklearn.model_selection as sklearnMS
from sklearn.decomposition import NMF

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
def train_test_split_player_shotInfo(randSeed = 546682):
    shots_path = os.path.abspath(os.path.join(os.getcwd(), os.pardir)) + '/data/shots/shots.csv'
    shots_dat = pd.read_csv(shots_path)
    
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
                                                                            statistic='count',
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
        
    
#def acquire_playerPic(PlayerID, zoom, offset=(250,400)):
#    def save_pic(playerID):
#        f = open(PlayerID+".png", 'wb')
#        f.write(request.urlopen("http://stats.nba.com/media/players/230x185/"+PlayerID+".png").read())
#        f.close()
#    
#    fn = get_sample_data(PlayerID+".png", asfileobj=False)
#    arr_img = plt.imread(fn, format='png')
#    imagebox = OffsetImage(arr_img, zoom=zoom)
#    #img.set_offset(offset)
#    img = osb.AnnotationBbox(imagebox, offset,xycoords='data',pad=0.0, box_alignment=(1,0), frameon=False)
#    return img

def plot_shotHist(player_shotHist, player, binDat, playersID, fileName, 
                  title='', norm_Opt='linear', plot_size=(5,5)):   
    bins, binRange, xedges, yedges, binnumber = binDat
    Xn_v = player_shotHist
    temp = np.array(Xn_v, dtype='float')/np.sum(Xn_v)
    shotHist_v = np.reshape(temp, bins)
    ##########
    extent = np.min(xedges), np.max(xedges), np.max(yedges), np.min(yedges)
    
    fig = plt.figure(figsize=plot_size)
    ax = plt.axes()
    cmap = plt.cm.magma_r
    if norm_Opt == 'log':
        plt.imshow(shotHist_v.T, cmap=cmap, 
                   norm=colors.LogNorm(vmin=1e-4, vmax=1e-1),
                   alpha=.85, extent=extent)
    else:
        plt.imshow(np.ma.masked_where(shotHist_v.T == 0., shotHist_v.T), 
                   cmap=cmap, alpha=.85, extent=extent)
    
    plot_court.draw_court(outer_lines=True, lw=1.5)
    
#    zoom = np.float(plot_size[0])/(12.0*2)
#    img = acquire_playerPic(playersID[player])
#    im = np.array(img).astype(np.float) / 255
#    print(im.shape)
#    fig.figimage(im, 0, fig.bbox.ymax - img.size[1])
#    imagebox = osb.OffsetImage(img, zoom=zoom)
#    xy=(250,400)
#    img_ann = osb.AnnotationBbox(imagebox, xy,
#                                 xybox=(120., -80.),
#                                 xycoords='data',
#                                 pad=0.0, boxcoords="offset points", 
#                                 frameon=False)
#    ax.add_artist(img_ann)
    
    ax.set_xlim([-300,300])
    ax.set_ylim([-100,500])
    ax.grid('off')
    ax.axis('off')
    ax.set_title('%s: %s'%(player, title), fontsize=15)
#    plt.axis('off')
    fig.tight_layout()
    fig.savefig(fileName, dpi=700)


#%%
    
randSeed = 546682


train_players_df, test_players_df, top_players_nameList, playersID = \
        train_test_split_player_shotInfo(randSeed = randSeed)
playersName = dict((v,k) for k,v in playersID.items())
# playersID-- key: name, value: ID
# playersName-- key: ID, value: name


flag_name = 'SHOT_ATTEMPTED_FLAG'


        
players_shotHist_train, binDat = gen_players_shotHist(train_players_df, 
                                                      top_players_nameList,
                                                      flag=flag_name)
players_shotHist_test, binDat = gen_players_shotHist(test_players_df, 
                                                     top_players_nameList,
                                                     flag=flag_name)
bins, binRange, xedges, yedges, binnumber = binDat

#
#player = 'James Harden'
#fileName = 'raw_shotHist_%s.png'%player.replace(' ', '')
##print(fileName)
#plot_shotHist(players_shotHist_train[player], player, binDat, playersID, 
#              fileName, title='raw histogram', norm_Opt='linear')


lgcp.run(top_players_nameList, players_shotHist_train, 
         binDat, randSeed, 
         phi2=float(sys.argv[0])**2, flag=flag_name)





#bins, binRange = ([25,18], [[-250,250], [-47.5,312.5]])
#temp = df[df.PLAYER_NAME == player]
#
#
#
#sigma2 = 1e3