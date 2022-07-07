
"""
Pre-Processing code for APP Usage Dataset - NB data is given in bytes -> code converts to bits
here, we read in the original dataset of app usage statitics, then for intervals of 5 mins, record the amount of data used within each interval
"""

import os
os.chdir('')  		# location where the dataset is stored

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import pickle
import math
from tqdm import tqdm

# User required settings ...
usage_type = 'full'       # set which type of usage needs to be stored in the new dataset, either uplink data, downlink or full
time_interval   = 30*60      # time interval that we want between the aggregated data, in SECONDS
true_end_date = datetime(2014,12,31,23,int(60 - time_interval/60),00) # set to datetime 2015,1,1,00,00,00 minus time_intrval

# rest of code is fixed 
usage_dict = {'uplink':'totalAmountBytesTX', 'downlink':'totalAmountBytesRX'}

start_date      = datetime(2014,1,1,00,00,00)               # starting date year,month,day,hr,min,sec
end_date        = datetime(2015,1,11,17,50,00)              # some of the data points run into 2015 so need get max time and truncate new dataset afterwards to cover just 2014

tot_secs  = (end_date - start_date).total_seconds() +  time_interval      # process everything in secs
time_axis = [start_date + timedelta(seconds=x) for x in range(0, int(tot_secs), int(time_interval))]

orig_data = pd.read_csv('shared_records.csv')

orig_data.totalAmountBytesTX.min()
orig_data.totalAmountBytesTX.idxmin()

temp = orig_data.totalAmountBytesTX
temp_2 = temp[temp<0]

new_data = pd.DataFrame()
new_data['Time'] = time_axis
    
# get names of apps in dataset    
apps = orig_data.appName.unique()

temp =  [[] for _ in range( len(time_axis) )] # len(time_axis)
user_list = pd.DataFrame()

# construct zero-initialized dataframe to store new data with cols -> time, uplink apps, downlink apps, app users
for i in apps:
    new_data[i+'_uplink'] = np.zeros(len(time_axis))
    new_data[i+'_downlink'] = np.zeros(len(time_axis))
    new_data[i+'_users'] = np.zeros(len(time_axis))
    user_list[i] = temp

   


for i in tqdm(range(0,len(orig_data))): # process each line in the original dataset
    
    # get start and end times of period, relative to initial start of dataset
    start_time = (datetime.strptime(orig_data.iloc[i].begin_timestamp,  '%Y-%m-%d %H:%M:%S') - start_date).total_seconds() 
    end_time   = start_time + orig_data.iloc[i].duration/1000.0      #( datetime.strptime(orig_data.iloc[i].end_timestamp,   '%Y-%m-%d %H:%M:%S') - start_date).total_seconds() 
    
    # total time of usage period in secs
    time = (end_time - start_time)
    
    # calculate average downlink/uplink usage rate (per second)
    up_rate   = orig_data[usage_dict['uplink']].iloc[i] / time
    down_rate = orig_data[usage_dict['downlink']].iloc[i] / time
    
    # uplink data in dataset contains negative values -> set to zero
    if(up_rate < 0):
        up_rate = 0.
        
    # get ID of current user    
    user = orig_data.loc[i, 'subscriber'] 
    
    # num intervals that usage period falls into, by finding first and last intervals
    int_start = math.trunc(start_time/time_interval)
    int_last  = math.trunc(end_time/time_interval)
    
    app_name = orig_data.iloc[i].appName
    
    # for each interval which the usage period falls into, add usage of that period into associated application
    for j in range(int_last - int_start + 1):
        
        # get length of time covered by current interval
        time_len = min(time_interval*(int_start+j+1), end_time) - start_time
        # shift 'starting' time upto start of next interval
        start_time = start_time + time_len   
        # add up/down link usage data to associated app & interval                                 
        new_data.at[int_start + j, app_name+'_uplink']  = new_data[app_name+'_uplink'].iloc[int_start + j] + time_len*up_rate
        new_data.at[int_start + j, app_name+'_downlink']  = new_data[app_name+'_downlink'].iloc[int_start + j] + time_len*down_rate
        
        # check if the current user is already listed within this time slot, if not, add their ID to the user list & increment # users
        temp = user_list.loc[int_start + j, app_name]
        if user not in temp:            
            temp.append(user)            
            user_list.at[int_start + j, app_name] = temp     
            # increment the amount of users using the app at current time
            new_data.at[int_start + j, app_name+'_users']  = new_data[app_name+'_users'].iloc[int_start + j] + 1
     
        
     
# check total amount of bytes add up before and after being decomposed into the applications - some small discrepancies may 
# arise due to rounding during (rate * period) calculations
print('sum of uplink data in new dataset:{}'.format(new_data[ apps+'_uplink'].sum().sum()/1000000000))   
print('sum of uplink data in old dataset:{}'.format(orig_data[usage_dict['uplink']].sum()/1000000000))   

print('sum of downlink data in new dataset:{}'.format(new_data[apps+'_downlink'].sum().sum()/1000000000  ))  
print('sum of downlink data in old dataset:{}'.format(orig_data[usage_dict['downlink']].sum()/1000000000 ))

# truncate data so that it only contains data from 2014, actual end date = 2015,1,1,00,00,00 minus time_intrval
new_data = new_data[: time_axis.index(true_end_date)+1 ]
# convert data from bytes to bits, ignore time col
new_data[new_data.columns[1:]] = new_data[new_data.columns[1:]]*8.0
    
# save to csv file
new_data.to_csv('{}_traffic_{}mins.csv'.format(usage_type, int(time_interval/60)), index=False)

"""
 Extra code: Visualise data statistics & Create final dataset used to model a single App plus   
"""
import matplotlib.pyplot as plt

plt.bar(apps, new_data[apps+'_uplink'].sum()/1000000000 )    
plt.xticks(rotation='vertical')  
plt.ylabel('Total {} (Gb)'.format(usage_type))  
plt.title('Uplink Statistics per App')
      
plt.bar(apps, new_data[apps+'_downlink'].sum()/1000000000 )    
plt.xticks(rotation='vertical')  
plt.ylabel('Total {} (Gb)'.format(usage_type))  
plt.title('Downlink Statistics per App')
  
# check sparsity of different apps (% of zeros per app) -> if an app is too sparse it may become difficult to fit a model to it
#sparsity = ( new_data[apps].isin([0]).sum() / len(new_data) ) * 100 
     
# create final dataset used when modeling a single App - removes uneccessary data
    
data = pd.read_csv('D:/Networks/Big Data/App Usage/{}_traffic_{}mins.csv'.format(usage_type, int(time_interval/60)))   # read in the dataset

# combine opera and opera mini
data['Opera_downlink'] = data['Opera_downlink'] + data['Opera Mini_downlink'] 
data['Opera_uplink'] = data['Opera_uplink'] + data['Opera Mini_uplink'] 
data['Opera_users'] = data['Opera_users'] + data['Opera Mini_users'] 
data = data.drop(['Opera Mini_downlink', 'Opera Mini_uplink', 'Opera Mini_users'], axis=1)
# delete Netflix data
data = data.drop(['Netflix_downlink', 'Netflix_uplink', 'Netflix_users'], axis=1)

# save to csv file
data.to_csv('App_traffic_{}mins.csv'.format(int(time_interval/60)), index=False)


# EoF