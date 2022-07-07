
"""
	Code that implements the benchmarking metrics described in the paper: "Explainable AI for Trees: From Local Explanations to Global Understanding"
	
	As decribed in this paper, these metrics can be used to compare the performance of different XAI methods against one another. However, they do not provide a
	ground truth to explicitly evaluate the failthfulness of a single XAI method, ie how well an XAI method should do if its is perfect. Therefore, this functionality 
	has been implemented in the "optimal_benchmarks()" method below. NB it should be noted that this method is extrememly memory demanding and requires complexity (M!)(2^M),
	with M being the number of inputs.
 
	Also note that this code has been primiraly developed for the work described in the paper "Resource Reservation in Sliced Networks: An Explainable Artificial Intelligence (XAI) Approach" and we
	offer no guarantee of its reliability when used in any other contexts or settings. For examples of how to use each metric, as well as computing final areas under each curve, please see lines 437-577
	in the main.py script found in this repository.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import itertools
from tqdm import tqdm

def powerset(iterable):
    s = list(iterable)
    return itertools.chain.from_iterable(itertools.combinations(s,r) for r in range(len(s)+1))

       
def shapley_local_acc(y_model, shap_values,  shap_null, plot_dist=False):
    """
        function to compute average normalised standard deviation across predictions using Eqs 6 - 14 from SHAP trees paper:
        y_model:        	output predictins of the model being explained
        shap_values:      	correspnding shapley values of each feature
        shap_null:      	the mean output of the model across the background set used to compute the shapley values i.e. shap_null = model.predict(x_train[:background_size]).mean()
    """
   
    sigma = np.sqrt( np.mean(np.square( y_model - (np.sum(shap_values , axis=1) +  shap_null) ) ) ) / np.sqrt( np.mean( np.square(y_model) ) )
    
    if plot_dist is True:
        temp = np.sqrt( np.square( y_model - (np.sum(shap_values , axis=1) +  shap_null) ) ) / np.sqrt( np.mean( np.square(y_model) ) )
        plt.hist(temp, bins=20)
        plt.xlabel('normalised standard deviation between Shapley values and predictions')
    
    if sigma < 10E-6:
        ans = 1.0
    elif sigma < 0.01:
        ans = 0.9
    elif sigma < 0.05:
        ans = 0.75
    elif sigma < 0.1:
        ans = 0.6
    elif sigma < 0.2:
        ans = 0.4
    elif sigma < 0.3:
        ans = 0.3
    elif sigma < 0.5:
        ans = 0.2
    elif sigma < 0.7:
        ans = 0.1
    else:
        ans = 0.0
        
    return ans, sigma

def optimal_benchmarks(predict_func, features, shap_values, mask, metrics='all', y_test=None, num_fracs=11):   

    """
    module to compute the optimal achieving curves for multiple XAI benchmarks, i.e. curves that result when SHAP values are 100% accurate
    features:       (numpy) set of input features corresponding to the shapley values 
    shap_values:    (numpy or dataframe) set of shapley values for x_test set
    mask:           1D array with mean values of each input feature  (usually taken from training set) 
    num_fracs       determines the fractional step size to consider in each masking operation
    metrics         list of the metrics to compute optimal curves for, i.e., 'keep_positive', 'keep_negative', 'remove_positive' etc.
    y_test          if computing keep/remove absolute metrics, need to pass corresponding ground truth of each feature
    
    output:         pandas dataframe with rows -> fraction of features included in model, columns -> average output of model for each metric
    
    metrics = ['keep_positive', 'keep_negative']
             """
    if metrics is 'all':
        metrics = ['keep_positive', 'keep_negative', 'remove_positive', 'remove_negative', 'keep_absolute', 'remove_absolute']
             
    num_feats = len(mask)           
    if len(mask) < num_fracs:       # make sure the number of fractions isnt larger than the number of features as this could lead to repeated mapping
        num_fracs = len(mask) + 1   # the extra fraction accounts for first point of each curve, where 0 features are kept/removed
        
    frac_nums = (np.round(np.linspace(0, len(mask), num_fracs))).astype(int)    # the integer amount of features used in each fraction
    frac_locs = frac_nums / frac_nums.max()                                     # the fractional step size (x ticks when plotting results)

    results = list() # use a list of dataframes to store the result of each feature (rows of each DF) and each fraction (cols of each DF) for each metric 
    key = {}
    for i,metric in enumerate(metrics):
        key[metric] = i
        results.append(pd.DataFrame(columns=frac_locs))
    
    # first compute a reusable masking array that iterates over all possible combinations of rank (ie which features are more important than others) AND sign (e.g whether SHAP values are negative or positive)           
    ranks = np.reshape(list( itertools.permutations( np.arange(num_feats)+1, num_feats )  ), (-1,num_feats))        # for each rank order, compute the entire sign matrix, resulting size = (ranks)*(signs) = (M!)(2^M)
    signs = np.reshape(list( itertools.product( [0,1], repeat=num_feats )  ), (-1,num_feats))   
    perm_mask = np.reshape(list(itertools.product( ranks, signs, repeat=1 )), (-1,num_feats*2))                     # resulting size = (ranks)*(signs) = (M!)(2^M)
    perm_mask = np.multiply(perm_mask[:,:num_feats],perm_mask[:,num_feats:])
    perm_mask = np.unique(perm_mask, axis=0)                                                                        # remove any duplicate redundant rows                                                                                                                        
    temp = np.sort(perm_mask, axis=1)                                                                               # all other redundant rows share a unique property that the difference between two consecutive numbers is larger than 1
    temp = temp[:,1:] - temp[:,:-1]
    perm_mask = np.delete(perm_mask, np.where(temp>1)[0],0)                                                         
    perm_mask[perm_mask==0] = -1                                                                                    # set all zeros to -1
    # note that this mask can be used as proxy shap values corresponding to all possible combinations concievable by a single row of data, where all positive combinations are transversed BUT all negative numbers are -1
  
    expanded_mask = np.broadcast_to(mask, np.shape(perm_mask) ).copy()          # expand the mask to the size of the permutation mask, used when masking features at each fraction number
    temp_best_masked = np.zeros((len(perm_mask), len(frac_nums)))               # matrix to store intemediate result of each feature across all fractions as features are unmasked
    temp_best_unmask = np.zeros((len(perm_mask), len(frac_nums)))               # matrix to store intemediate result of each feature across all fractions as features are masked
    abs_perm_locs = np.where(np.all(perm_mask!=-1, axis=1))                     # keep track of the permuttions where all features are positive - used by the absolute keep/remove metrics
    
    for i in tqdm(range(len(features))):    # for each feature, we use this proxy SHAP values to mask/unmask features in all conceivable combinations and at each considered step size of fractions
        
        expanded_data = np.broadcast_to(features[i,:], np.shape(perm_mask) ).copy()     # expand the data to the size of the permutation mask, used when unmasking features at each fraction number
        
        temp_best_masked[:,0] = predict_func( mask.reshape((1,-1)) )                    # first datum equal to masking all features 
        temp_best_unmask[:,0] = predict_func( features[i,:].reshape((1,-1)) )           # first datum equal to the features themselves
      
        for N in range(1, len(frac_nums)):                                          # for each fraction, get model ouput at all permutations. choose permuation which yields best area for each metric case
            x_temp_masked = expanded_mask.copy()                                    # create a temporary dataset where every feature is masked at start, and unmasked as the number of fractions increases
            x_temp_unmask = expanded_data.copy()                                    # create a temporary dataset where every feature is unmasked at start, and masked as the number of fractions increases
            
            top_N_val = np.sort(perm_mask)[:,-frac_nums[N]].reshape(-1,1)           # get the 'proxy' shap value of the Nth highest element in each sample, NB doesnt matter if we are looking for most neg SHAP values as we just want to preserve rank ordering
            top_N_locs = np.where( (perm_mask>=top_N_val)&(perm_mask>0) )           # get locs of the elements which are positive and greater than or equal to the Nth value 
            
            x_temp_masked[top_N_locs] =  expanded_data[top_N_locs]                  # using these locs, replace the top masked features with their real values   
            x_temp_unmask[top_N_locs] =  expanded_mask[top_N_locs]                  # using these locs, replace real values of features with their masked values
           
            temp_best_masked[:,N] = predict_func( x_temp_masked ).reshape(-1,)      # make predictions for each permutation as more features have been unmasked
            temp_best_unmask[:,N] = predict_func( x_temp_unmask ).reshape(-1,)      # make predictions for each permutation as more features have been masked
          
        # now calculate the area covered by each row, row with highest area will be recorded as being the optimal for that feature
        if 'keep_positive' in metrics:  #  np.min(temp_best_masked, axis=1)[:,np.newaxis]
            results[key['keep_positive']].loc[i,:] = temp_best_masked[np.argmax( np.trapz(temp_best_masked-np.amin(temp_best_masked), axis=1) ),:]
            
        if 'keep_negative' in metrics:  
            results[key['keep_negative']].loc[i,:] = temp_best_masked[np.argmin( np.trapz(temp_best_masked-np.amin(temp_best_masked), axis=1) ),:]
            
        if 'remove_positive' in metrics:  
            results[key['remove_positive']].loc[i,:] = temp_best_unmask[np.argmin( np.trapz(temp_best_unmask-np.amin(temp_best_unmask), axis=1) ),:]
            
        if 'remove_negative' in metrics: 
            results[key['remove_negative']].loc[i,:] = temp_best_unmask[np.argmax( np.trapz(temp_best_unmask-np.amin(temp_best_unmask), axis=1) ),:]
        
        if 'keep_absolute' in metrics:  # NB original SHAP paper tries to maximise accuracy for classification BUT for regression we want to minimse RMSE 
            temp = np.square(temp_best_masked[abs_perm_locs] - y_test[i])
            results[key['keep_absolute']].loc[i,:] = temp[np.argmin( np.trapz(temp, axis=1) ),:] 
            
        if 'remove_absolute' in metrics:  # NB original paper tries to maximise accuracy for classification BUT for regression we want to minimse RMSE 
            temp = np.square(temp_best_unmask[abs_perm_locs] - y_test[i]) 
            results[key['remove_absolute']].loc[i,:] = temp[np.argmax( np.trapz(temp, axis=1) ),:]
            
    return results, key
    
    
        
def keep_positive_mask(predict_func, features, shap_values, mask, num_fracs=11):   

    """
    Measure ability of explanation method to identify features that increases the model output
    features:       (numpy) set of input features corresponding to the shapley values 
    shap_values:    (numpy or dataframe) set of shapley values for x_test set
    mask:           1D array with mean values of each input feature  (usually taken from training set) 
    num_fracs       determines the fractional step size to consider in each masking operation
    output:         pandas series with indices -> fraction of features included in model, elements -> average output of model     
    """
    
    # make sure the number of fractions isnt larger than the number of features as this could lead to repeated mapping
    if len(mask) < num_fracs:
        num_fracs = len(mask) + 1
        
    frac_nums = (np.round(np.linspace(0, len(mask), num_fracs))).astype(int)    # the integer amount of features used in each fraction
    frac_locs = frac_nums / frac_nums.max()                                     # store the x tick locations for plotting result of each fraction
    output = np.zeros_like(frac_locs)                                           # store the result of each fraction
    expanded_mask = np.broadcast_to(mask, np.shape(features) ).copy()           # expand the mask to the same size as the features   
    output[0] = predict_func( mask.reshape((1,-1)) )                            # get first datum by masking all inputs to model - common across predictions
            
    # for each fraction, i.e x tick on the plot, we mask the relevant amount of 'best' features, as indicated by the shap value
    # to avoid a possible scenario where multiple shap values are identical, we first add a small amount of noise to the shap values 
    shap_values = shap_values + np.random.uniform(0, 1, np.shape(shap_values)) * 1e-8
    for N in range(1, len(frac_nums)):  
        x_temp = expanded_mask.copy()                                           # create a temporary dataset where every feature is masked
        top_N_val = np.sort(shap_values)[:,-frac_nums[N]].reshape(-1,1)         # get the shapley value of the Nth highest element in each sample
        top_N_locs = np.where( (shap_values>=top_N_val)&(shap_values>0) )       # get locs of the elements which are positive and greater than or equal to the Nth value 
        x_temp[top_N_locs] =  features[top_N_locs]                              # using these locs, replace the top masked features with their real values        
        output[N] = np.mean( predict_func( x_temp ) )                           # make predictions for each sample and take average output as the datum for current fraction
    
    return pd.DataFrame({'keep_positive':output}, index=frac_locs)
       
        
def keep_negative_mask(predict_func, features, shap_values, mask, num_fracs=11):       
    """
    Measure ability of explanation method to identify features which decreases the model output
    x_test:         numpy dataset of input features, for which shapley values have already been computed for
    shap_values:    dataset of shapley values for x_test set
    mask:           1D array with mean values of each input feature to the model (taken from training set) 
    output:         pandas series with indices -> fraction of features included in model, elements -> average output of model
    """

    # make sure the number of fractions isnt larger than the number of features as this could lead to repeated mapping
    if len(mask) < num_fracs:
        num_fracs = len(mask) + 1
        
    frac_nums = (np.round(np.linspace(0, len(mask), num_fracs))).astype(int)
    frac_locs = frac_nums / frac_nums.max()
    output = np.zeros_like(frac_locs)                                           # store the result of each fraction
    expanded_mask = np.broadcast_to(mask, np.shape(features) ).copy()           # expand the mask to the same size as the features
    output[0] = predict_func( mask.reshape((1,-1)) )                            # get first datum by masking all inputs to model - common across predictions
  
    # for each fraction, i.e x tick on the plot, we mask the relevant amount of 'best' features, as indicated by the shap value
    # to avoid a possible scenario where multiple shap values are selected, we first add a small amount of noise to the shap values 
    shap_values = shap_values + np.random.uniform(0, 1, np.shape(shap_values)) * 1e-8
    for N in range(1, len(frac_nums)):      
        x_temp = expanded_mask.copy()                                           # create a temporary dataset where every feature is masked
        low_N_val = np.sort(shap_values)[:,frac_nums[N-1]].reshape(-1,1)        # get shapley value of Nth lowest element in each sample
        low_N_locs = np.where( (shap_values<=low_N_val)&(shap_values<0) )       # get locs of the elements which are negative and less then or equal to the Nth value
        x_temp[low_N_locs] =  features[low_N_locs]                              # using these locs, replace the top masked features with their real values 
        output[N] = np.mean( predict_func( x_temp ) )                           # make predictions for each sample and take average as datum for current fraction
    
    return pd.DataFrame({'keep_negative':output}, index=frac_locs)

def remove_positive_mask(predict_func, features, shap_values, mask, num_fracs=11):

    """
    Measure ability of explanation method to identify features which increases the model output by maksing the top features
    features:       numpy dataset of input features, for which shapley values have already been computed for
    shap_values:    dataset of shapley values for features set
    mask:           1D array with mean values of each input feature to the model (taken from training set) 
    output:         pandas series with indices -> fraction of features included in model, elements -> average output of model
    """        
        
    if len(mask) < num_fracs:     # make sure the number of fractions isnt larger than the number of features as this could lead to repeated mapping
        num_fracs = len(mask) + 1
        
    frac_nums = (np.round(np.linspace(0, len(mask), num_fracs))).astype(int)
    frac_locs = frac_nums / frac_nums.max()
    output = np.zeros_like(frac_locs)    
       
    # create a dataset where every negaitve effecting feature is masked with its average value:
    x_start = features.copy()                                                   # create copy of the features
    expanded_mask = np.broadcast_to(mask, np.shape(features) ).copy()           # expand the mask to the same size as the features
    #neg_locs = np.where( shap_values<=0 )                                       # get locs of the elements which are negative  
    #x_start[neg_locs] =  expanded_mask[neg_locs]                                # using these locs, mask the negative effecting features with their respective average values
    output[0] = np.mean( predict_func( x_start ) )                              # get first datum by taking average output of the model across test set
    
    shap_values = shap_values + np.random.uniform(0, 1, np.shape(shap_values)) * 1e-8    
    for N in range(1, len(frac_nums)):
        x_temp = x_start.copy()                                                 # create a temporary dataset where all negative inducing feats are masked
        top_N_val = np.sort(shap_values)[:,-frac_nums[N]].reshape(-1,1)         # get shap value of Nth highest element in each sample
        top_N_pos_locs = np.where( (shap_values>=top_N_val)&(shap_values>0) )   # get locs of the elements which are positive and greater then or equal to the Nth value 
        x_temp[top_N_pos_locs] = expanded_mask[top_N_pos_locs]                  # using these locs, mask the top features with their respective average values
        output[N] = np.mean(predict_func( x_temp ))                             # at this point, temp_x ->  copy of features with top N feats masked, use model to make predictions for each sample and take average as datum for current fraction
    
    return pd.DataFrame({'remove_positive':output}, index=frac_locs)


    
def remove_negative_mask(predict_func, features, shap_values, mask, num_fracs=11):       
    """
    Measure ability of explanation method to identify features which decreases the model output by masking features with most negative shapley values
    features:       numpy dataset of input features, for which shapley values have already been computed for
    shap_values:    dataset of shapley values for features set
    mask:           1D array with mean values of each input feature to the model (taken from training set) 
    
    output:         pandas series with indices -> fraction of features included in model, elements -> average output of model
    """

    if len(mask) < num_fracs:  # make sure the number of fractions isnt larger than the number of features as this could lead to repeated mapping
        num_fracs = len(mask) + 1
        
    frac_nums = (np.round(np.linspace(0, len(mask), num_fracs))).astype(int)
    frac_locs = frac_nums / frac_nums.max()
    output = np.zeros_like(frac_locs) 
             
    expanded_mask = np.broadcast_to(mask, np.shape(features) ).copy()                   # create a dataset where every negaitve inducing feature is masked with its average value
    x_start = features.copy()
    #pos_locs = np.where( shap_values>0 )                                                # get locs of the elements which are positive  
    #x_start[pos_locs] = expanded_mask[pos_locs]                                         # using these locs, mask the neg inducing features with their respective average values
    output[0] = np.mean( predict_func( x_start ) )                                      # get first datum by taking average output of the model across test set
   
    shap_values = shap_values + np.random.uniform(0, 1, np.shape(shap_values)) * 1e-8   # to avoid a possible scenario where multiple shap values are identical, we first add a small amount of noise to the shap values 
    for N in range(1, len(frac_nums)):
        x_temp = x_start.copy()                                                         # create a temporary dataset where every positive effecting feature is masked
        low_N_val = np.sort(shap_values)[:,frac_nums[N-1]].reshape(-1,1)                # get shapley value of Nth lowest element in each sample
        low_N_pos_locs = np.where( (shap_values<=low_N_val)&(shap_values<0) )           # get locs of the elements which are positive and greater then or equal to the Nth value
        x_temp[low_N_pos_locs] = expanded_mask[low_N_pos_locs]                          # using these locs, mask the lowest features with their respective average values
        output[N] = np.mean(predict_func( x_temp ))                                     # at this point, temp_x -> masked version of original features with all but top N feats masked,  make predictions for each sample and take average as datum for current fraction
        
    return pd.DataFrame({'remove_negative':output}, index=frac_locs)

    
def keep_absolute_mask(predict_func, features, shap_values, mask, y_test, num_fracs=11):   
    """
    Measure ability of explanation method to identify most important features by unmasking features with highest absolute values
    features:       numpy dataset of input features, for which shapley values have already been computed for
    shap_values:    dataset of shapley values for x_test set
    mask:           1D array with mean values of each input feature to the model (taken from training set) 
    output:         pandas series with indices -> fraction of features included in model, elements -> average output of model
    """     
    # make sure the number of fractions isnt larger than the number of features as this could lead to repeated mapping
    if len(mask) < num_fracs:
        num_fracs = len(mask) + 1
        
    frac_nums = (np.round(np.linspace(0, len(mask), num_fracs))).astype(int)
    frac_locs = frac_nums / frac_nums.max()
    output = np.zeros_like(frac_locs)                                           
        
    expanded_mask = np.broadcast_to(mask, np.shape(features) ).copy()
    output[0] = np.sqrt(np.mean(np.square(predict_func(expanded_mask) - y_test)))                   # get first datum by masking all inputs to model - common across predictions
     
    #shap_values = shap_values + np.random.uniform(0, 1, np.shape(shap_values)) * 1e-8              # to avoid a possible scenario where multiple shap values are identical, we first add a small amount of noise to the shap values
    abs_shap = np.absolute(shap_values)                                                             # take absolute values
    for N in range(1, len(frac_nums)):                                                              # for each fraction, i.e x tick on the plot, we mask the relevant amount of 'best' features, as indicated by the shap value
        x_temp = expanded_mask.copy()                                                               # create a temporary dataset where every feature is masked
        top_N_val = np.sort(abs_shap)[:,-frac_nums[N]].reshape(-1,1)                                # get shapley value of Nth highest absolute element in each sample
        top_N_pos_locs = np.where( (abs_shap>=top_N_val) )                                          # get locs of the elements which are greater then or equal to the Nth value 
        x_temp[top_N_pos_locs] = features[top_N_pos_locs]                                           # using these locs, replace the top masked features with their real values   
        
        # if more than one feature is exactly equal to the Nth value within a sample, there could be more than N feats selected at that sample
        # therefore, need to get locations of all features sharing the Nth value and delete those which are surplus (occuring last in sorted order) 
        # slunberg uses tie_breaking_noise = const_rand(X_train.shape[1], random_state) * 1e-6 to ensure no shap vals are ever the exact same
        top_N_pos_locs = np.array( top_N_pos_locs ) # first convert the locations to an array for easy handling       
        sample, num_feats = np.unique(top_N_pos_locs[0], return_counts=True) # for each sample, count no. of feats selected
        flags = np.where(num_feats>N)[0] # if any sample has more than N feats selected, need to do work to remove the surplus ones
        num_del = num_feats - N # calc. how many features need to be deleted from each sample
            
        for i in flags: # for each sample that has more than N feats selected
            # get locations of all featswhich are  equal to the Nth value
            temp_locs = np.array( np.where(abs_shap[i] == top_N_val[i]) )[0]     
            # change sample's features at the surplus locations back to their mask values
            x_temp[i,temp_locs[-num_del[i]:] ] = mask[ temp_locs[-num_del[i]:] ]
           
        
        output[N] = np.sqrt(np.mean(np.square(np.subtract(predict_func(x_temp), y_test))))     # at this point, temp_x -> masked version of original features with all but top N feats masked, make predictions for each sample, then compute rmse by comparing to actual labels for each sample

    return pd.DataFrame({'keep_absolute':output}, index=frac_locs)
    


def remove_absolute_mask(predict_func, features, shap_values, mask, y_test, num_fracs=11):
    """
    Measure ability of explanation method to identify features which increases the model output by maksing the top features
    x_test:         numpy dataset of input features, for which shapley values have already been computed for
    shap_values:    dataset of shapley values for x_test set
    mean_feats:     1D array with mean values of each input feature to the model (taken from training set) 
    output:         pandas series with indices -> fraction of features included in model, elements -> average output of model
    """        
    
    # make sure the number of fractions isnt larger than the number of features as this could lead to repeated mapping
    if len(mask) < num_fracs:
        num_fracs = len(mask) + 1
        
    frac_nums = (np.round(np.linspace(0, len(mask), num_fracs))).astype(int)
    frac_locs = frac_nums / frac_nums.max()
    output = np.zeros_like(frac_locs)                                           
        
    expanded_mask = np.broadcast_to(mask, np.shape(features) ).copy()
    output[0] = np.sqrt(np.mean(np.square(predict_func(features) - y_test)))                        # get first datum by masking all inputs to model - common across predictions
        
    shap_values = shap_values + np.random.uniform(0, 1, np.shape(shap_values)) * 1e-8               # add noise to shap values 
    abs_shap = np.absolute(shap_values)                                                             # take absolute
    for N in range(1, len(frac_nums)):
        x_temp = features.copy()                                                                    # create a temporary dataset where every feature is masked
        top_N_val = np.sort(abs_shap)[:,-frac_nums[N]].reshape(-1,1)                                # get shapley value of Nth highest absolute element in each sample
        top_N_pos_locs = np.where( (abs_shap>=top_N_val) )                                          # get locs of the elements which are greater then or equal to the Nth value 
        x_temp[top_N_pos_locs] = expanded_mask[top_N_pos_locs]                                      # using these locs, mask the top features with their real values 
        output[N] = np.sqrt(np.mean(np.square(predict_func(x_temp) - y_test)))     # at this point, temp_x -> masked version of original features with all but top N feats masked, make predictions for each sample, then compute rmse by comparing to actual labels for each sample

    return pd.DataFrame({'remove_absolute':output}, index=frac_locs)

