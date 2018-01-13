

import scipy
import numpy as np

import os
import lgcp_func as lgcp

# =============================================================================
# 
# =============================================================================

def transpose_vec(vec):
	if len(vec.shape) == 1:		# it is a row vec
		return np.array([row_vec]).T
	elif len(vec.shape) == 2 and vec.shape[2] == 1:	# it is a col vec
		return vec.T[0]
	else:
		print('Error: vec not the right size')

def 

def find_boxes_Xfrac_counts(x, player_shotHist):
    # find indices of histogram that account for at least 90% of the shots
    # from the given player
    tot_n_shots = np.sum(player_shotHist)

    
    [[i, ]]
    
    

    
def calc_spatial_FGpercent(unnorm_LL_attempt, unnorm_LL_made):
    lgcp_FGpercent = np.zeros(unnorm_LL_attempt.shape)
    lgcp_FGpercent[:,:] = unnorm_LL_made[:,:] / unnorm_LL_attempt[:,:]
    
    