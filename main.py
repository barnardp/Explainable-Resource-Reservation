# -*- coding: utf-8 -*-
"""
Created on Mon Aug  3 01:49:19 2020

Source code for paper "Resource Reservation in Sliced Networks: An Explainable Artificial Intelligence (XAI) Approach"
 
"""
import os
os.chdir('')                    # location where utility files are located
save_loc = ''      				# location to save results
data_loc = '' 					# location of dataset

import tensorflow as tf
physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
from sklearn.metrics import mean_squared_error
from datetime import datetime
from tqdm import tqdm
import shap
import time
import random
import datetime as dt
import matplotlib.dates as mdates

import utility_funcs as utf
import xai_benchmark as xbm

np.random.seed(10)

tenant = 'Chrome'   # name of the tenant to regress on
pred_horizon = 1    # prediction horizon to regress on
# use lag groups to define the features used in the model can be single lags or composite lags: takes a vector where each element gives 
# the SIZE of the lags to use as one feature, eg [1,6] -> use lag_0 as first feature and take average of lags 2-7 as second feature
lag_groups = np.ones(1, dtype=int) # [1,1] #np.ones(2, dtype=int) 
num_lags=sum(lag_groups)

data = pd.read_csv(data_loc)    # read in the dataset
data['Time'] = pd.to_datetime(data['Time'],format='%Y-%m-%d %H:%M:%S')  # convert time column to datetime
time_axis    = pd.to_datetime(data['Time'],format='%Y-%m-%d %H:%M:%S')  # copy the time to a unique variable
time_inter = int((time_axis.iloc[1]-time_axis.iloc[0]).seconds/60)      # time interval between data in mins
day_len = int(24*60/time_inter)                                         # length of one day
SP_time_data = pd.DataFrame(); SP_time_data['Time'] = time_axis         # Extract general time related features from the data
temp = time_axis.dt.hour*60 + time_axis.dt.minute                                   # create a time of day feature where each timestamp is mapped to a minute with a day
SP_time_data['Time of Day (cos)']   = np.cos(temp * 2*np.pi/1440)                   # split the time of day into cos and sin components so that first min of day wraps around last min smoothly
SP_time_data['Time of Day (sin)']   = np.sin(temp * 2*np.pi/1440)                   # NB to do this we convert each minute into radians, i.e. 1440 mins in a day = 360 degrees
temp = time_axis.dt.dayofweek*24*60 + time_axis.dt.hour*60 + time_axis.dt.minute    # similarly, create time of week feature
SP_time_data['Time of Week (cos)']  = np.cos(temp * 2*np.pi/10080)    				# in this case there are 7*24*60 = 10080 mins in a week, each mapped across 360 degrees or 2Pi radians
SP_time_data['Time of Week (sin)']  = np.sin(temp * 2*np.pi/10080)   
SP_time_data['Users'] = data['{}_users'.format(tenant)]
               
app_names = (data.filter(like='_uplink').columns.str.split('_')).str[0]
num_apps = len(app_names) # total number of apps, including our own

# combine the uplink and downlink data into a new dataset
new_data = pd.DataFrame()
for i in app_names:
    uplink, downlink, _ = data.filter(regex=i).columns
    new_data[i] = data[uplink] + data[downlink]
    
del data, uplink, downlink          # clear up some space
new_data = abs(new_data/1000000)    # convert data to Mb

#temp = np.percentile(new_data.sum(axis=1), 99)
#new_data[new_data>temp] = temp

# specifiy how the data is split up into train, val and test sets, return indices for each interval
train_start = time_axis[time_axis == datetime(2014, 9, 1,00,00,00) ].index[0]
val_start  = time_axis[time_axis ==  datetime(2014,10,25,00,00,00) ].index[0] # make sure this is one week long, check with # days in the month choosen
test_start = time_axis[time_axis ==  datetime(2014,11, 1, 00,00,00) ].index[0]
test_end   = time_axis[time_axis ==  datetime(2014,11,30, 23,55,00) ].index[0] #time_axis[time_axis ==  datetime(2014,12, 31,23,int(60-time_inter*(pred_horizon+1)),00) ].index[0]    # NB end date refers to the last date at which we make our final prediction                    
(test_start- val_start)/(val_start- train_start)*100

C_max =  np.percentile(new_data[train_start:val_start].sum(axis=1), 80).round(0)          

# do offline simulation to create train/val sets, data in test set will be over-written during the online simulation
S_k, S_feedback, R_L_total = utf.offline_simulation(new_data.loc[train_start:], tenant, C_max, test_start)

""" ******************************************************************************************* """
U_genie = np.sum( S_feedback.loc[test_start:test_end,'U'], axis=0 )
S_total = np.sum( S_feedback.loc[test_start:test_end,'S'], axis=0 )
print("Percentage of undelivered Traffic by genie on test set:{} %".format( (100*(U_genie/S_total)).round(1) ) )

U_genie = np.sum( S_feedback.loc[:test_start-1,'U'], axis=0 )
S_total = np.sum( S_feedback.loc[:test_start-1,'S'], axis=0 )
print("Percentage of undelivered Traffic by genie on training set:{} %".format( (100*(U_genie/S_total)).round(1) ) )
del U_genie, S_total
""" ******************************************************************************************* """
# FS selection stage followed by creation of fully lagged sets for SUB and aggregated sets of SUB according to the grouping 
full_lags, aggr_lags, grp_lags, max_lag = utf.FS_get_best_lags(S_feedback, lag_groups=lag_groups, Feature_Selection = False)
     
# create a lagged version of the number of users too
temp = utf.create_lag_set(SP_time_data.loc[:,'Users'], lags=[num_lags-1], full_set = True, keep_na = False)

# create label column
y = utf.create_lag_set(S_feedback.loc[:,'Ropt'], lags=[-pred_horizon], keep_na = False); y = y.rename(columns={y.columns[0]:'Ropt'})
# combine everything into one dataset, automatically align and reset index col so that it aligns with the numpy versions later
x = pd.concat([SP_time_data.drop('Users', axis=1), temp, aggr_lags[0], aggr_lags[1], aggr_lags[2], y], axis=1, join='inner')  #  .reset_index(drop=True)
# remove label col from data
y = x[['Ropt']].copy()
x = x.drop(['Ropt'], axis=1)
# remove the time col, can be used later to map times to the indices in the dataset
time_axis = pd.to_datetime( x['Time'], format = '%Y-%m-%d %H:%M:%S' )
x = x.drop(['Time'], axis=1)

# standardise each column of the x set -> create copy for easier handling later
means = x.loc[:val_start].mean(axis=0)
std_devs = x.loc[:val_start].std(axis=0)
x_stnd = ((x-means)/std_devs).copy()

# create train, val and test sets  
train_data_X = x_stnd.loc[:val_start-1]            
train_data_Y = y.loc[:val_start-1]            
val_data_X   = x_stnd.loc[val_start:test_start-1]  
val_data_Y   = y.loc[val_start:test_start-1]  
test_data_X  = x_stnd.loc[test_start:test_end]   
test_data_Y  = y.loc[test_start:test_end]   


"""   ****************************** train DNN on the offline data ******************************   """
import tensorflow as tf

# can create a decaying weight function to assign greater importance to most recent time samples
# where weight of least important sample = 1.0 and weight halves every 7 days
#time_decay = np.exp(np.arange(len(train_data_X))/((60*24*7/10)/np.log(2)))

import utility_funcs as utf
model, params = utf.get_hyper_model(train_data_X, train_data_Y, val_data_X, val_data_Y, method='bayesian', batch_size=int(24*60/time_inter), save_path=save_loc )

# model = utf.create_DNN_model(len(x_stnd.columns))
np.random.seed(1)
tf.random.set_seed(1)
history = model.fit(train_data_X.values, train_data_Y.values, epochs=1000, verbose=2, batch_size=day_len,
                    validation_data=(val_data_X.values, val_data_Y.values), callbacks = [utf.early_stop])
#config = model.summary()
utf.plot_loss(history)
#print(params.values)


i = 7
exp = train_start+day_len*i
data = model.predict( x_stnd.loc[exp:exp+day_len] )
plt.figure(); 
plt.plot(time_axis.loc[exp:exp+day_len], train_data_Y.loc[exp:exp+day_len], label='Optimal')
plt.plot(time_axis.loc[exp:exp+day_len], data, label='Reserved')
plt.legend()
plt.ylabel('Mb')
plt.title('Offline Model Performance on Training set')
plt.xticks(rotation=45)

i = 5
exp = val_start+day_len*i
data = model.predict( x_stnd.loc[exp:exp+day_len] )
plt.figure(); 
plt.plot(time_axis.loc[exp:exp+day_len], val_data_Y.loc[exp:exp+day_len], label='Optimal')
plt.plot(time_axis.loc[exp:exp+day_len], data, label='Reserved')
plt.legend()
plt.ylabel('Mb')
plt.title('Offline Model Performance on Training set')
plt.xticks(rotation=45)


i = 5
exp = test_start+day_len*i
data = model.predict( x_stnd.loc[exp:exp+day_len] )
plt.figure(); 
plt.plot(time_axis.loc[exp:exp+day_len], test_data_Y.loc[exp:exp+day_len], label='Optimal')
plt.plot(time_axis.loc[exp:exp+day_len], data, label='Reserved')
plt.legend()
plt.ylabel('Mb')
plt.title('Offline Model Performance on Test set set')
plt.xticks(rotation=45)

y_pred_org = pd.DataFrame(model.predict(x_stnd.loc[test_start:test_end]))
# calculate RMSE on test set
print("RMSE of trained model is {} Mb".format( np.sqrt(mean_squared_error( test_data_Y, y_pred_org ) ).round(1) ))    
# calc rmse of naive prediction, ie by simply predicting most recent known traffic demand   
print("RMSE of naive solution is {} Mb".format( np.sqrt(mean_squared_error( test_data_Y, full_lags[0].loc[test_start:test_end,'S_0'] ) ).round(1) ))    



"""   ****************************** Switch to Online Reservations **********************   
    Use trained model to make actual reservations starting on the validation set and test set-> this yields new distribution of U,B signals
    The validation set will subsequently become our 'background' set for explaining the model
"""
 
y_online_pred = pd.Series([], dtype=float)
    
 # val set should contain fully online reservations, need to go back by the maximum lag of U,B as these features only get updated after that amount of iterations
 # depending on the prediction horizon, the end point should be x.index[-1]-pred_horizon
for curr_time in tqdm(range(x.index[0], test_end+1 ) ): 
    
               
    # make a reservation for future time = current time + horizon
    pred =  float(model.predict( np.expand_dims(x_stnd.loc[curr_time], axis=0) )) # make pred/reservation for future time = curr + pred_horizon
    
    if ( (pred + float(R_L_total.loc[curr_time + pred_horizon])) <= C_max ): # If MNOs capacity limit isnt reached, no sharing 
        A_S_k = pred # given reservations of each SP, calculate actual amount scheduled by MNO for our tenant
        MNO_spare = C_max - float(pred + R_L_total.loc[curr_time + pred_horizon])
    else: # tenants share resources
        A_S_k = C_max * float(pred / (pred + R_L_total.loc[curr_time + pred_horizon] ) ) # given reservations of each SP, calculate actual amount scheduled by MNO for our tenant
        MNO_spare = 0
        
    # calculate amount by which our tenant will be under-provisioned in the future
    S_k_under = np.maximum(S_k.loc[curr_time + pred_horizon] - A_S_k, 0).values

    B_k = 0
    if (MNO_spare > 0 and S_k_under > 0): # where best effort occurs
        B_k = np.minimum( S_k_under,  MNO_spare/num_apps ) 
        
    # calculate the future delivered and undelivered traffic of our tenant 
    D_k = np.minimum(A_S_k + B_k, S_k.loc[curr_time + pred_horizon])
    U_k = np.subtract( S_k.loc[curr_time + pred_horizon] , D_k )
        
    # first update new data into the fully lagged U and B sets, then transfer into online x set  
    full_lags[1].at[curr_time + pred_horizon,'U_0'] = U_k    # update U_0 at current time slot
    full_lags[2].at[curr_time + pred_horizon,'B_0'] = B_k    # update U_0 at current time slot
    # then update next time slots lags by shifting current row to right, ie so that at next time slot, the 'previous' lag at -1 is updated to be same as lag 0 now
    full_lags[1].loc[curr_time + pred_horizon + 1,:] = full_lags[1].loc[curr_time + pred_horizon,:].shift(1)  
    full_lags[2].loc[curr_time + pred_horizon + 1,:] = full_lags[2].loc[curr_time + pred_horizon,:].shift(1)     
    # use newly updated full lags at the predicted time slot to calculate updated aggregated features
    new_U_B = utf.update_aggr_lags(full_lags[1].loc[curr_time + pred_horizon,:], full_lags[2].loc[curr_time + pred_horizon,:], lag_groups, grp_lags[1], grp_lags[2])  
    # write the aggregated results into the x set(s), remember to standardise those placed into the online x set used for making predictions
    x.loc[curr_time + pred_horizon, new_U_B.index] = new_U_B         # write unstandardised result into original x set
    x_stnd.loc[curr_time + pred_horizon, new_U_B.index] = (new_U_B-means[new_U_B.index])/std_devs[new_U_B.index]   # write standardised result into online x set
    
    # store predicted value for later - NB already unstandadrised
    y_online_pred[curr_time] = pred

""" ************************************  SAVE  DATA ************************************ """
model.save('{}/model.h5'.format(save_loc))   
with open("{}/misc_data.pkl".format(save_loc), "wb") as f:
   pickle.dump([time_axis, time_inter, day_len, new_data, train_start, val_start, test_start, test_end],f)       
with open("{}/misc_data_2.pkl".format(save_loc), "wb") as f:
   pickle.dump([means, std_devs, params, params.values, y, train_data_X, train_data_Y, val_data_X, val_data_Y, test_data_X, test_data_Y],f)       
with open("{}/misc_data_3.pkl".format(save_loc), "wb") as f:
   pickle.dump([C_max, R_L_total, SP_time_data, S_feedback, x, x_stnd],f)  # NB save copy of x data before it is truncated   
with open("{}/misc_data_4.pkl".format(save_loc), "wb") as f:
   pickle.dump([app_names, num_apps, pred_horizon],f)

# remove the last index of the x and x_stnd data, as these surpass the test end date
x = x.loc[:test_end]
x_stnd = x_stnd.loc[:test_end]
with open("{}/online_data.pkl".format(save_loc), "wb") as f:
    pickle.dump([ x, x_stnd, y_online_pred, full_lags ],f) 
   
"""
model = tf.keras.models.load_model('{}/model.h5'.format(save_loc))    
x, x_stnd, y_online_pred, full_lags = pickle.load(open("{}/online_data.pkl".format(save_loc), "rb"))
time_axis, time_inter, day_len, new_data, train_start, val_start, test_start, test_end = pickle.load(open("{}/misc_data.pkl".format(save_loc), "rb"))
means, std_devs, params, params.values, y, train_data_X, train_data_Y, val_data_X, val_data_Y, test_data_X, test_data_Y = pickle.load(open("{}/misc_data_2.pkl".format(save_loc), "rb"))

C_max, R_L_total, SP_time_data, S_feedback, _, _ = pickle.load(open("{}/misc_data_3.pkl".format(save_loc), "rb"))
app_names, num_apps, pred_horizon = pickle.load(open("{}/misc_data_4.pkl".format(save_loc), "rb"))

""" 
""" ****************************************************************************************** """        

# compute various statistics about the online reservations
y_pred_org = pd.DataFrame(y_online_pred.loc[test_start:test_end])                                   # get the onine predictions made during the test portion of the data
R_naive = full_lags[0].loc[test_start:test_end,'S_0'].values.reshape(-1,1)                          # define naive reservation requests as being equal to the most recent traffic demand observed, i.e. NB the req using S at time (t) corresponds to the req made for time (t+horizon)

# calculate RMSE statistics
print("RMSE of trained model is {} Mb".format( np.sqrt(mean_squared_error( test_data_Y, y_pred_org ) ).round(1) ))                                                              # calculate RMSE on test set
print("RMSE of naive solution is {} Mb".format( np.sqrt(mean_squared_error( test_data_Y, R_naive ) ).round(1) ))                                                                # calc RMSE between optimal reservations and naive reservation

# calculate over/under reservation relative to optimal reservations
over_res = ( sum( np.multiply((y_pred_org.values-test_data_Y.values),(y_pred_org.values>test_data_Y.values)) ) / sum( (y_pred_org.values>test_data_Y.values) ) ).round(1)       # calculate over reservation
under_res = ( sum( np.multiply((y_pred_org.values-test_data_Y.values),(y_pred_org.values<test_data_Y.values)) ) / sum( (y_pred_org.values<test_data_Y.values) ) ).round(1)    # calculate under reservation
naive_over_res =  ( sum( np.multiply((R_naive-test_data_Y.values),(R_naive>test_data_Y.values)) ) / sum( (R_naive>test_data_Y.values) ) ).round(1)                              # calculate over-reservation of naive solution
naive_under_res = ( sum( np.multiply((R_naive-test_data_Y.values),(R_naive<test_data_Y.values)) ) / sum( (R_naive<test_data_Y.values) ) ).round(1)                            # calculate under reservation of naive solution
print('Over reservation of model: {}Mb   Under reservation of model: {}Mb'.format(float(over_res), float(under_res)))
print('Over reservation of Naive solution: {}Mb   Under reservation of Naive solution: {}Mb'.format(float(naive_over_res), float(naive_under_res)))

# calculate undelivered traffic statstics, NB for the naive solution, we need to simulate the behavior of the model using the naive reservation reqs as this effects the BE assigned to each tenant and ultimately the undelivered amounts 

# Simulate Network using Naive reservation reqs -> calculate actual reserved rss, then assigned BE and finally the undlivered traffic etc.
R_others = R_L_total.loc[test_start+1:test_end+1].values.reshape(-1,1)                                  # reservations of the other tenants at each timeslot
no_share = np.array( np.where( (R_naive + R_others) <= C_max ) )[0]                                 # where MNOs capacity limit isnt exceeded 
share = np.array( np.where((R_naive + R_others) > C_max ) )[0]                                      # where MNOs capacity limit is exceeded
A_S_k = np.zeros_like(R_naive)                                                                      # given reservations of each SP, calculate actual amount scheduled by MNO for our tenant
A_S_k[no_share] = R_naive[no_share]                                                                 # where total sum of reqs less than capacity, actual reserved = requested amount
A_S_k[share] = C_max * (R_naive[share] / (R_naive[share] + R_others[share]) )                       # where total sum of reqs exceed capacity, actual reserved = relative to all other requests

S_k = S_feedback.loc[test_start+1:test_end+1,'S'].values.reshape(-1,1)                              # the actual traffic demands seen during the test set
S_k_under = np.maximum(S_k - A_S_k, 0)                                                              # calculate amount by which our tenant is under-provisioned
MNO_spare = np.zeros_like(R_naive)                                                                  # calculate MNOs spare capacity before resources are assigned to underprovisioned tenants (zero where sharing occurs)    
MNO_spare[no_share] = C_max - (R_naive[no_share] + R_others[no_share])                              # only non zero where no sharing occurs       
S_k_best = np.zeros_like(R_naive)                                                                   # calculate the best effort traffic of our tenant(zero wherever total BE is zero) 
where_BE = np.array( np.where( np.multiply(MNO_spare, S_k_under) > 0) )[0]                          # where best effort occurs
S_k_best[where_BE] = np.minimum( S_k_under[where_BE],  MNO_spare[where_BE] / (len(app_names)) )    
D_k = np.minimum(A_S_k + S_k_best, S_k)                                                             # calculate delivered traffic of our tenant 
U_k = np.subtract( S_k , D_k )                                                                      # calculate undelivered traffic of our tenant 


# END of simulation. Now compare undelivered traffic from genie model and Naive method, NB S_feedback already contains the undelivered traffic due to genie reservations, while full_lags contains the undelivered from the online model
U_model = np.sum( full_lags[1].loc[test_start:test_end,'U_0'], axis=0 )
U_genie = np.sum( S_feedback.loc[test_start:test_end,'U'], axis=0 )         
U_naive = np.sum( U_k )
S_total = np.sum( S_feedback.loc[test_start:test_end,'S'], axis=0 )             # record the total amount of traffic seen during the test set so that the undelivered amounts can be given relative to the total

print("% of undelivered Traffic by model on test set:{} %".format( 100*(U_model/S_total) ) )
print("% of undelivered Traffic by genie on test set:{} %".format( 100*(U_genie/S_total) ) )
print("% of undelivered Traffic by naive on test set:{} %".format( 100*(U_naive/S_total) ) )

# can also count SLA violations, i.e., whenever there is undelivered traffic
print("No. of SLA violations by model: {}".format( np.count_nonzero(full_lags[1].loc[test_start:test_end,'U_0']) ) )
print("No. of SLA violations by genie: {}".format( np.count_nonzero(S_feedback.loc[test_start:test_end,'U'], axis=0 ) ) )
print("No. of SLA violations by naive: {}".format( np.count_nonzero(U_k) ) )

# can also compute mean undelivered traffic
print("Mean undelivered traffic of model: {}".format(  np.mean(full_lags[1].loc[test_start:test_end,'U_0'] ) ) )
print("Mean undelivered traffic of genie: {}".format( np.mean(S_feedback.loc[test_start:test_end,'U'], axis=0) ) )
print("Mean undelivered traffic of naive: {}".format( np.mean(U_k) ) )

del U_model, U_genie, S_total, over_res, under_res, naive_over_res, naive_under_res

# plot an sample reservation across time

i = 5
exp = test_start+day_len*i
plt.figure(); 
plt.plot(time_axis.loc[exp:exp+day_len], test_data_Y.loc[exp:exp+day_len], label='Optimal')
plt.plot(time_axis.loc[exp:exp+day_len], y_online_pred.loc[exp:exp+day_len], label='Reserved')
plt.legend()
plt.ylabel('Mb')
plt.title('Online Model Performance on Test set set')
plt.xticks(rotation=45)


"""     ****************************** Compute the Kernel Shap values ******************************    """

# Define the background set - Use the full validation week as the background 
feat_names = list(x.columns)
X_reference = x_stnd.loc[val_start:test_start-1]
background_size = len(X_reference)  # one week worth of samples
expected_val = float(np.mean(model.predict(X_reference)))

try:    # If shap values were previously calculated using the distributed method, load them here
    shap_kernel = pickle.load(open("{}/shap_kernel_test.pkl".format(save_loc), "rb"))[0]
    test_samps = shap_kernel.index
except: # else calculate them sequentially 
    test_samps = np.arange(test_start, test_start + 2000, step=4)   # define the test samples we want to explain
    shap_kernel = pd.DataFrame([], columns=feat_names)
    explainer = shap.KernelExplainer( model.predict, X_reference )
    shap_kernel = utf.calc_shap_values(explainer, shap_kernel, x_stnd, test_samps, background_size, file_name='shap_kernel_test', save_loc=save_loc,  batch_size=100)

""" ****************************** SAVE the SHAP values and related data ****************************** """
with open("{}/shapr_data.pkl".format(save_loc), "wb") as f:
    pickle.dump([expected_val, X_reference.values, x_stnd.loc[shap_kernel.index], test_data_Y.loc[shap_kernel.index], feat_names],f)  
with open("{}/shap_data.pkl".format(save_loc), "wb") as f:
    pickle.dump([feat_names, X_reference, background_size, expected_val, test_samps, shap_kernel],f)    
	
#feat_names, X_reference, background_size, expected_val, test_samps, shap_kernel = pickle.load(open("{}/shap_data.pkl".format(save_loc), "rb"))
""" ****************************** Visualise the SHAP values ****************************** """
# if we want to plot the feature values asociated with each shapley value, we first need to re-combine the cos/sin components of each time feature in the shapley values and the x data as well
x_combined, shap_combined = utf.combine_time_feats(x, shap_kernel)
new_feats = list(x_combined.columns)

# To make the visual plots easier to follow, we can also change the names of the features to better reflect their meanings
# to group the features into super features, we need to define a number of mapping objects for the explainer to use
feat_names_expanded = ['Active_Users', 'Traffic_Demand', 'Undelivered', 'Best_Effort', 'Time_of_Day','Time_of_Week']
feat_name_map = {'Active_Users':'Users_0','Traffic_Demand':'S_0', 'Undelivered':'U_0', 'Best_Effort':'B_0', 'Time_of_Day':'Time of Day','Time_of_Week':'Time of Week'}
x_combined.columns = feat_names_expanded
shap_combined.columns = feat_names_expanded


x_comb_categ = x_combined.copy().round(1)
temp_time = utf.MinToHour(time_axis.loc[x.index].dt.hour*60 + time_axis.loc[x.index].dt.minute) # convert timestamps to 'minutes since start of each day', i.e. will be aggregated
temp_time = [dt.datetime.strptime(elem, '%H:%M:%S') for elem in temp_time]  
temp_time = [temp_t.strftime('%H:%M') for temp_t in temp_time]
x_comb_categ['Time_of_Day'] = temp_time

temp_time = utf.DayOfWeek(time_axis.loc[x.index])                                                     # convert timestamps to day of week, ie aggregated across a single week
temp_time = [temp_t.strftime('%a %H:%M') for temp_t in temp_time]
x_comb_categ['Time_of_Week'] = temp_time

# now plot the explanation of a single test point
import shap # NB need to comment out line 233 of _general.py to plot times in waterfall plot
#shap.force_plot(expected_val, shap_combined.loc[test_start+144].values, x_comb_categ.loc[test_start+144], matplotlib=True)
shap.plots.waterfall_legacy(expected_val, shap_combined.loc[test_start+144].values, x_comb_categ.loc[test_start+144], list(x_comb_categ.columns))



plot_samps = shap_kernel.index[::1]   # define the test samples we want to plot, useful if we want to sub sample later
utf.FI_summary_plot(shap_combined.loc[plot_samps], x_combined.loc[plot_samps], background=x_combined.loc[val_start:test_start-1], title=None, units='Mb') # NB we use a modified version of the SHAP library's 'summary plot', which allows use to color the points relative to the background distribution
# shap.summary_plot(shap_combined.loc[plot_samps].values, x_combined.loc[plot_samps]) # can plot the SHAP library sumary plot for comparison


# plot SHAP PDPs of each combined feature, NB the SHAP PDPs are relative to the background dataset 
for feat in x_combined.columns:    
    shap.dependence_plot(feat, shap_combined.values, x_combined.loc[shap_kernel.index], interaction_index=None)


#We may also want to show a histogram of the x data (taken from the background set), its mean and the mean model output values alongside the PDPs, S_0: mean=26.75


model_predict_unstd = lambda x: model.predict( (x-means)/std_devs ).reshape(-1,)                                # function that takes in unstandardised x data and standardises them to makes predictions


for feat in feat_names_expanded:    
    if 'Time' not in feat: # don't plot the PDPs for the time features just yet
        utf.shap_PDP(feat, shap_combined, x.loc[shap_combined.index,feat_name_map[feat]], model_predict_unstd, 
                 units='Mb', x_units='Mb',  # title='PDP plot of {}'.format(feat),
                 feat_expected_value=(X_reference[feat_name_map[feat]]*std_devs[feat_name_map[feat]] + means[feat_name_map[feat]]).mean(), 
                 model_expected_value = None, plot_zero=True, nbins=30, ax_color='black') # plot line thru expected model value = 0 to see if mean value of feat has zero effect (like the linear case) 

feat = 'Time_of_Day'
utf.shap_PDP(feat, shap_combined, time_axis.loc[shap_combined.index], model_predict_unstd, 
         plot_time=True, t_format='day', rotation=30,
         units='Mb',  #title='PDP plot of {}'.format(feat),
         feat_expected_value = dt.datetime(1900, 1, 1, 12, 0), # middle of day occurs at dt.datetime(1900, 1, 1, 12, 0)
         model_expected_value = None, plot_zero=True) 
      
feat = 'Time_of_Week'
utf.shap_PDP(feat, shap_combined, time_axis.loc[shap_combined.index], model_predict_unstd, 
         plot_time=True, t_format='week', rotation=30,
         units='Mb', # title='PDP plot of {}'.format(feat),
         feat_expected_value = dt.datetime(2020, 6, 4, 12, 0), # middle of week occurs at dt.datetime(1900, 1, 1, 12, 0)
         model_expected_value = None, plot_zero=True) 
    


""" can also plot a conventional PDP of the features -> keep in mind that for this to be directly relatable to the SHAP PDPs, the x data here needs to be the same x set used
for the explainer background, Can still plot PDPs using the training, full validation or test set etc and compare against the PDP from the background set to see if its out of sample
data causes the model to perfom unexpectedly -> x.loc[test_samps]
Also note that it may be necessary to remove oultiers or large data points to better compare each plot
"""
fig, ax = shap.partial_dependence_plot('U_0', model_predict_unstd, X_reference, model_expected_value=True, feature_expected_value=True, ice=False, show=False)
ax.set_title('Partial Dependence Plot of U trace across background set')


""" ******************************** BENCHMARK EVAULUATIONS ****************************************** """
import xai_benchmark as xbm

np.random.seed(10)
eval_samps = np.random.randint(test_start, test_end, 100)     	# define the samples we will use to evaluate (common practice is to use 100 samples)
mean_feats = np.mean(X_reference.values, axis=0)            	# can also use mean x values across entire training set as reference point

sigma_test = xbm.shapley_local_acc(y_model=model.predict(x_stnd.loc[eval_samps]).reshape(-1,), shap_values=shap_kernel.loc[eval_samps], shap_null = expected_val)

keep_positive   = xbm.keep_positive_mask(   model.predict, x_stnd.loc[eval_samps].values, shap_kernel.loc[eval_samps].values,  mean_feats)
keep_negative   = xbm.keep_negative_mask(   model.predict, x_stnd.loc[eval_samps].values, shap_kernel.loc[eval_samps].values,  mean_feats)  
remove_positive = xbm.remove_positive_mask( model.predict, x_stnd.loc[eval_samps].values, shap_kernel.loc[eval_samps].values,  mean_feats)  
remove_negative = xbm.remove_negative_mask( model.predict, x_stnd.loc[eval_samps].values, shap_kernel.loc[eval_samps].values,  mean_feats)  
keep_absolute   = xbm.keep_absolute_mask(   model.predict, x_stnd.loc[eval_samps].values, shap_kernel.loc[eval_samps].values,  mean_feats, test_data_Y.loc[eval_samps].values) 
remove_absolute = xbm.remove_absolute_mask( model.predict, x_stnd.loc[eval_samps].values, shap_kernel.loc[eval_samps].values,  mean_feats, test_data_Y.loc[eval_samps].values) 

                     
optimal_metrics, metric_key = xbm.optimal_benchmarks(model.predict, x_stnd.loc[eval_samps].values, shap_kernel.loc[eval_samps].values, mean_feats, metrics='all', y_test=test_data_Y.loc[eval_samps].values)



keep_pos_random = pd.DataFrame([]); np.random.seed(10)
for i in range(100):  # create comparison graph by randomising shapley values
    keep_pos_random  = pd.concat([keep_pos_random, xbm.keep_positive_mask(model.predict, x_stnd.loc[eval_samps].values, np.random.randn(*x_stnd.loc[eval_samps].values.shape), mean_feats)], axis=1)

keep_neg_random = pd.DataFrame([]); np.random.seed(10)
for i in range(100):  # create comparison graph by randomising shapley values
    keep_neg_random  = pd.concat([keep_neg_random, xbm.keep_negative_mask(model.predict, x_stnd.loc[eval_samps].values, np.random.randn(*x_stnd.loc[eval_samps].values.shape), mean_feats)], axis=1)

rem_pos_random = pd.DataFrame([]); np.random.seed(10)
for i in range(100):  # create comparison graph by randomising shapley values
    rem_pos_random  = pd.concat([rem_pos_random, xbm.remove_positive_mask(model.predict, x_stnd.loc[eval_samps].values, np.random.randn(*x_stnd.loc[eval_samps].values.shape), mean_feats)], axis=1)
    
rem_neg_random = pd.DataFrame([]); np.random.seed(10)
for i in range(100):  # create comparison graph by randomising shapley values
    rem_neg_random  = pd.concat([rem_neg_random, xbm.remove_negative_mask(model.predict, x_stnd.loc[eval_samps].values, np.random.randn(*x_stnd.loc[eval_samps].values.shape), mean_feats)], axis=1)

keep_abs_random = pd.DataFrame([]); np.random.seed(10)
for i in range(100):  # create comparison graph by randomising shapley values
    keep_abs_random  = pd.concat([keep_abs_random, xbm.keep_absolute_mask(model.predict, x_stnd.loc[eval_samps].values, np.random.randn(*x_stnd.loc[eval_samps].values.shape), mean_feats, test_data_Y.loc[eval_samps].values)], axis=1)

rem_abs_random = pd.DataFrame([]); np.random.seed(10)
for i in range(100):  # create comparison graph by randomising shapley values
    rem_abs_random  = pd.concat([rem_abs_random, xbm.remove_absolute_mask(model.predict, x_stnd.loc[eval_samps].values, np.random.randn(*x_stnd.loc[eval_samps].values.shape), mean_feats, test_data_Y.loc[eval_samps].values)], axis=1)

"""
Compute the area under each curve of the optimal, model and random cases. Here we define the area under the curve as the are above the y=0 line, when each curve as been offset using the minimum value of the 3 curves
"""

baseline = min(keep_positive['keep_positive'].min(), np.mean(optimal_metrics[metric_key['keep_positive']], axis=0).min(), np.mean(keep_pos_random, axis=1).min() ) # set the baseline to be the min value of the optimal, shift all curves down by the baseline
np.trapz(keep_positive['keep_positive'] - baseline  ).round(2)
np.trapz(np.mean(optimal_metrics[metric_key['keep_positive']], axis=0) - baseline ).round(2)
np.trapz(np.mean(keep_pos_random, axis=1) - baseline).round(2)

baseline = min(keep_negative['keep_negative'].min(), np.mean(optimal_metrics[metric_key['keep_negative']], axis=0).min(), np.mean(keep_neg_random, axis=1).min())
np.trapz(keep_negative['keep_negative'] - baseline).round(2)
np.trapz(np.mean(optimal_metrics[metric_key['keep_negative']], axis=0) - baseline).round(2)
np.trapz(np.mean(keep_neg_random, axis=1) - baseline).round(2)

baseline = min(remove_positive['remove_positive'].min(), np.mean(optimal_metrics[metric_key['remove_positive']], axis=0).min(), np.mean(rem_pos_random, axis=1).min() )
np.trapz(remove_positive['remove_positive'] - baseline).round(2)
np.trapz(np.mean(optimal_metrics[metric_key['remove_positive']], axis=0) - baseline).round(2)
np.trapz(np.mean(rem_pos_random, axis=1)- baseline).round(2)

baseline = min(remove_negative['remove_negative'].min(), np.mean(optimal_metrics[metric_key['remove_negative']], axis=0).min(), np.mean(rem_neg_random, axis=1).min() )
np.trapz(remove_negative['remove_negative'] - baseline).round(2)
np.trapz(np.mean(optimal_metrics[metric_key['remove_negative']], axis=0) - baseline).round(2)
np.trapz(np.mean(rem_neg_random, axis=1) - baseline).round(2)

baseline = min(keep_absolute['keep_absolute'].min(), np.sqrt(np.mean(optimal_metrics[metric_key['keep_absolute']], axis=0)).min(), np.mean(keep_abs_random, axis=1).min())
np.trapz(keep_absolute['keep_absolute'] - baseline).round(2)
np.trapz(np.sqrt(np.mean(optimal_metrics[metric_key['keep_absolute']], axis=0))- baseline).round(2)
np.trapz(np.mean(keep_abs_random, axis=1) - baseline).round(2)

baseline = min(remove_absolute['remove_absolute'].min(), np.sqrt(np.mean(optimal_metrics[metric_key['remove_absolute']], axis=0)).min(), np.mean(rem_abs_random, axis=1).min() )
np.trapz(remove_absolute['remove_absolute'] - baseline).round(2)
np.trapz(np.sqrt(np.mean(optimal_metrics[metric_key['remove_absolute']], axis=0)) - baseline).round(2)
np.trapz(np.mean(rem_abs_random, axis=1) - baseline).round(2)


fig = plt.figure(figsize=(6,4)) # default=6,4
ax = fig.add_subplot(1,1,1, title='Keep Positive (Mask)') 
ax.plot(keep_positive['keep_positive'], label='Kernel SHAP' )
ax.plot(np.mean(optimal_metrics[metric_key['keep_positive']], axis=0), label='Optimal', color='red')
ax.plot(keep_pos_random.mean(axis=1), label='Random', color='blue', alpha=1.0)
ax.plot(keep_pos_random, color='blue', alpha=0.05)
ax.set_xlabel('Max fraction of features kept')
ax.set_ylabel('Mean model output (MB)')
ax.legend(loc='best', framealpha=0.3) 

fig = plt.figure(figsize=(6,4)) # default=6,4
ax = fig.add_subplot(1,1,1, title='Keep Negative (Mask)') 
ax.plot(keep_negative['keep_negative'], label='Kernel SHAP' )
ax.plot(np.mean(optimal_metrics[metric_key['keep_negative']], axis=0), label='Optimal', color='red')
ax.plot(keep_neg_random.mean(axis=1), label='Random', color='blue', alpha=1.0)
ax.plot(keep_neg_random, color='blue', alpha=0.05)
ax.set_xlabel('Max fraction of features kept')
ax.set_ylabel('Mean model output (MB)')
ax.legend(loc='best', framealpha=0.3) 

fig = plt.figure(figsize=(6,4)) # default=6,4
ax = fig.add_subplot(1,1,1, title='Remove Positive (Mask)') 
ax.plot(remove_positive['remove_positive'], label='Kernel SHAP' )
ax.plot(np.mean(optimal_metrics[metric_key['remove_positive']], axis=0), label='Optimal', color='red')
ax.plot(rem_pos_random.mean(axis=1), label='Random', color='blue', alpha=1.0)
ax.plot(rem_pos_random, color='blue', alpha=0.05)
ax.set_xlabel('Max fraction of features removed')
ax.set_ylabel('Mean model output (MB)')
ax.legend(loc='best', framealpha=0.3) 

fig = plt.figure(figsize=(6,4)) # default=6,4
ax = fig.add_subplot(1,1,1, title='Remove Negative (Mask)') 
ax.plot(remove_negative['remove_negative'], label='Kernel SHAP' )
ax.plot(np.mean(optimal_metrics[metric_key['remove_negative']], axis=0), label='Optimal', color='red')
ax.plot(rem_neg_random.mean(axis=1), label='Random', color='blue', alpha=1.0)
ax.plot(rem_neg_random, color='blue', alpha=0.05)
ax.set_xlabel('Max fraction of features removed')
ax.set_ylabel('Mean model output (MB)')
ax.legend(loc='best', framealpha=0.3) 
#np.trapz(np.mean(optimal_metrics[metric_key['remove_negative']], axis=0))
#np.trapz(remove_negative['remove_negative'])

fig = plt.figure(figsize=(6,4)) # default=6,4
ax = fig.add_subplot(1,1,1, title='Keep Absolute (Mask)') 
ax.plot(keep_absolute['keep_absolute'], label='Kernel SHAP' )
ax.plot(np.sqrt(np.mean(optimal_metrics[metric_key['keep_absolute']], axis=0)), label='Optimal', color='red')
ax.plot(keep_abs_random.mean(axis=1), label='Random', color='blue', alpha=1.0)
ax.plot(keep_abs_random, color='blue', alpha=0.05)
ax.set_xlabel('Max fraction of features kept')
ax.set_ylabel('Mean model RMSE (MB)')
ax.legend(loc='best', framealpha=0.3)  

fig = plt.figure(figsize=(6,4)) # default=6,4
ax = fig.add_subplot(1,1,1, title='Remove Absolute (Mask)') 
ax.plot(remove_absolute['remove_absolute'], label='Kernel SHAP' )
ax.plot(np.sqrt(np.mean(optimal_metrics[metric_key['remove_absolute']], axis=0)), label='Optimal', color='red') 
ax.plot(rem_abs_random.mean(axis=1), label='Random', color='blue', alpha=1.0)
ax.plot(rem_abs_random, color='blue', alpha=0.05)
ax.set_xlabel('Max fraction of features removed')
ax.set_ylabel('Mean model RMSE (MB)')
ax.legend(loc='best', framealpha=0.3) 



""" ********************************** SAVE DATA ********************************** """
with open("{}/benchmarking_data.pkl".format(save_loc), "wb") as f:
    pickle.dump([eval_samps, mean_feats, sigma_test, keep_positive, keep_negative, remove_positive, remove_negative, keep_absolute, remove_absolute, optimal_metrics, metric_key],f) 
  
with open("{}/benchmarking_random.pkl".format(save_loc), "wb") as f:
    pickle.dump([keep_pos_random, keep_neg_random, rem_pos_random, rem_neg_random, keep_abs_random, rem_abs_random],f) 


# EoF
 