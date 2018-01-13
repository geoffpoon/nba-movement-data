

import scipy
import numpy as np

import os

# =============================================================================
# 
# =============================================================================
    
def ln_factorial(n):
    # Use the log(gamma function)
    return scipy.special.gammaln(n + 1)

def ln_prior(nbins, zn_v, logdet_cov_K, inv_cov_K):
    part1 = - (nbins/2.) * np.log(2 * np.pi) - (0.5 * logdet_cov_K)
    part2 = -0.5 * np.dot(zn_v, np.dot(inv_cov_K, zn_v))
    return part1 + part2

def lambdaN_func(z0, zn_v):
    return np.exp(z0 + zn_v)

def ln_lambdaN_func(z0, zn_v):
    return z0 + zn_v

def ln_likelihood(z0, zn_v, Xn_v):
    part1 = -lambdaN_func(z0, zn_v)
    part2 = Xn_v * ln_lambdaN_func(z0, zn_v)
    part3 = -ln_factorial(Xn_v)
    #print(np.sum(part1), np.sum(part2), np.sum(part3))
    #print(part3)
    return np.sum(part1 + part2 + part3)

def ln_postprob(z, Xn_v, logdet_cov_K, inv_cov_K, nbins):
    z0 = z[0]
    zn_v = z[1:]
    return ln_prior(nbins, zn_v, logdet_cov_K, inv_cov_K) + ln_likelihood(z0, zn_v, Xn_v)

def cov_func(dist_matrix, sigma2, phi2):
    return sigma2 * np.exp( -(dist_matrix**2) / (2 * phi2) )    

def run(top_players_nameList, players_shotHist_train, 
        binDat, randSeed, 
        phi2=30.**2, sigma2=1e3, flag='SHOT_ATTEMPTED_FLAG'):
    import time
    
    bins, binRange, xedges, yedges, binnumber = binDat
    nbins = np.prod(bins)
    
    # Creating the grid we will use for analysis
    XX, YY = np.meshgrid(xedges, yedges)
    binX_flat = XX.T[:-1,:-1].flatten()
    binY_flat = YY.T[:-1,:-1].flatten()
    binXY = np.column_stack((binX_flat.T, binY_flat.T))
    dist_matrix = scipy.spatial.distance_matrix(binXY, binXY)
    
    cov_K = cov_func(dist_matrix, sigma2, phi2)
    # np.linalg.det under/overflows for very small/large values of det, so slogdet is more robust
    sign, logdet_cov_K = np.linalg.slogdet(cov_K)
    inv_cov_K = np.linalg.inv(cov_K)
    
    #-------------------------------------------
    
    directory = flag + '/shotHist_LGCP_phi%d_seed%d'%(phi2**0.5, randSeed)
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    print('Computing LL, normalized Lambda for each player')
    print('================================================')
    print('================================================')
    for i, player in enumerate(top_players_nameList):
        try:
            lambdaN_v = np.loadtxt(directory + '/lambda_%s.txt'%(player))
            if np.all(lambdaN_v == lambdaN_v[0]):
                print(player, 'BAD')
            else:
                print(player, 'DONE')
        except:
            print('================================================')
            start_time = time.time()
            
            Xn_v = players_shotHist_train[player]
            z0_guess = np.log(np.mean(Xn_v))
            zn_v_guess = np.zeros(len(Xn_v))
            z_guess = np.append(z0_guess, zn_v_guess)
        
            neg_logLike = lambda *args: -ln_postprob(*args)
            result = scipy.optimize.minimize(neg_logLike, z_guess, 
                                             args=(Xn_v, logdet_cov_K, inv_cov_K, nbins))
            z_MaxLike = result["x"]
            z0_MaxLike = z_MaxLike[0]
            zn_MaxLike = z_MaxLike[1:]
            lambdaN_v = np.exp(z0_MaxLike + zn_MaxLike)
            norm_lambdaN_v = lambdaN_v / np.sum(lambdaN_v)
        
            print(player)
            print("------  %s seconds ------" %(time.time() - start_time))
            if np.all(norm_lambdaN_v == norm_lambdaN_v[0]):
                print(player, 'BAD')
            np.savetxt(directory + '/lambda_%s.txt'%(player), lambdaN_v)
#            np.savetxt(directory + '/norm_lambda_%s.txt'%(player), norm_lambdaN_v)