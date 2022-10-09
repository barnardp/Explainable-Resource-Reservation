"""
Utility functions for reservation code
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import tensorflow as tf
#physical_devices = tf.config.experimental.list_physical_devices('GPU')
#tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, LeakyReLU
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import optimizers, metrics
import kerastuner as kt
from kerastuner import HyperModel, Hyperband
import IPython
#from keras.losses import huber_loss
import datetime as dt
import matplotlib.dates as mdates
from datetime import datetime
from tensorflow.keras.losses import Huber
import matplotlib.colors as mcolors
import matplotlib.cm as cm
from shap.plots import colors, _labels
import keras_tuner as kt # use kerastuner for older versions of tensorflow
from shap import KernelExplainer
from tqdm import tqdm
import pickle



def offline_simulation(data, tenant, C_max, test_start, alpha = 0.05, std=0.0025, be_greedy = 0.1):
    """ 
        Perform an offline simulation between the tenants in the network and the MNO:
        Outputs:    Recorded traffic traces for reserved, best effort, unserved, and served traffic 
        S_k:        Our tenants usage demands (training set), pandas DataFrame
        L_k:        Usage demands of 'other' tenants, pandas DataFrame
        C_max:      MNOs maximum capacity, can use np.percentile(pd.concat([L_k, S_k], axis=1), 80)
        alpha:      probability of 'other' tenants under reserving
        std:        standard devs of the 'other' tenants reservation residuals, should be well below 1.0,
                    i.e. if original std = 1 then range of N(u,1) ~ [-4,4], but when scaling error by 
                    instaneous signal size, i.e. (S_k/M)*N(u,1), new dist becomes N(u,S_k^2 * 1^2 / M^2),
                    if M = 1/10, new std becomes 1/100 = 0.01.
        be_greedy   specify what fraction of the time our tenant should attempt to reserve nothing, i.e explore BE chances
        sell:       If true, other tenants who over-provisioned will loose their excess resources
        
        NB, each L_k tenant perfroms its own reservation policy, which is decribed by its own 
        alpha paramter (i.e. the % by which they under-provision), and standard deviation 
        (the average amount by which they under/over provision). i.e. assuming the nromal dist.
        strectches accross 8 stds, the average amount of under-reservation is given by L_k/
        
        sk = S_k.values.reshape(-1,)
        lk = L_k.values
                                                                                    
    """

    np.random.seed(10)
    
    # copy data, split into our own tenant and other tenants, convert all to numpy arrays
    L_k = data.copy()    
    S_k = L_k[tenant].values.reshape(-1,)   
    L_k = L_k.drop([tenant], axis=1)
    app_names = L_k.columns
    L_k = L_k.values
    
    #z_scores = st.norm.ppf(1 - alpha)     
    #R_L_k = np.maximum( 0, np.multiply( L_k, (1 + np.random.normal(z_scores*std, std, np.shape(L_k) ) ) ) ) # reservations of the other SPs
    R_L_k = L_k.copy() # In this case, just assume the other tenants are perfect predictors of their own traffic    
    
    # aggregate other tenants reservations into a single trace
    R_L_total = np.sum(R_L_k, axis=1)
    
    # assume our tenant makes unifromly random reservations for now, use percentile ONLY from training set to avoid leakage into test set
    #R_S_k = np.random.uniform(0, np.max(S_k[:test_start]), len(S_k))   #np.random.uniform(0, max(S_k), len(S_k))
    """
    # assume tenant makes reservations drawn from the distribution of their own traffic demand
    # to approx this distribution, uniformly draw random observation from their traffic demand trace (but leaving out test set part of it)
    indices = np.random.uniform(0, test_start, len(S_k)).astype(int)
    R_S_k = S_k[indices]
    # set some fraction to zero -> to help induce greater best effort
    indices = np.random.uniform(0, test_start, int(len(S_k)*be_greedy) ).astype(int)
    R_S_k[indices] = 0
    """
    
    
     # calculate the optimal reservation of our tenant 
    N = len(app_names)+1    
    reserve = np.array( np.where( (C_max - R_L_total)/N < S_k) )[0]
    R_opt = np.zeros_like(S_k) # zero wherever reserve is False    
    R_opt[reserve] = np.minimum(S_k[reserve], ( np.add(N*S_k[reserve], R_L_total[reserve]) - C_max) / (N-1) )  # in case that our own demand is higher than the total capacity (should never actually happen)
     
    
    # set our tenants reservations to be the optimal ones
    # NB if we randomly initialise the reservations and kept iteratively doing 
    # online evaluation -> online dataset -> re-train on new data then this would lead to same outcome
    R_S_k = R_opt  # S_k*0.75
    
    
    no_share = np.array( np.where( (R_S_k + R_L_total) <= C_max ) )[0]  # where MNOs capacity limit isnt exceeded 
    share = np.array( np.where((R_S_k + R_L_total) > C_max ) )[0]       # where MNOs capacity limit is exceeded
    # given reservations of each SP, calculate actual amount scheduled by MNO for our tenant
    A_S_k = np.zeros_like(R_S_k) 
    A_S_k[no_share] = R_S_k[no_share] 
    A_S_k[share] = C_max * (R_S_k[share] / (R_S_k[share] + R_L_total[share]) )      
    # calculate amount by which our tenant is under-provisioned
    S_k_under = np.maximum(S_k - A_S_k, 0)   
    # calculate the MNOs spare capacity before any resources are assigned to underprovisioned tenants
    MNO_spare = np.zeros_like(R_S_k) # zero where sharing occurs
    MNO_spare[no_share] = C_max - (R_S_k[no_share] + R_L_total[no_share]) # only non zero where no sharing occurs       
    # calculate the best effort traffic of our tenant
    S_k_best = np.zeros_like(R_S_k) # zero wherever total BE is zero
    where_BE = np.array( np.where( np.multiply(MNO_spare, S_k_under) > 0) )[0] # where best effort occurs
    S_k_best[where_BE] = np.minimum( S_k_under[where_BE],  MNO_spare[where_BE] / (len(app_names)+1) )   
    # calculate the delivered and undelivered traffic of our tenant  
    D_k = np.minimum(A_S_k + S_k_best, S_k)
    U_k = np.subtract( S_k , D_k ) 
    # calculate the optimal reservation of our tenant 
#    N = len(app_names)+1 
#    reserve = np.array( np.where( (C_max - R_L_total)/N < S_k) )[0]
#    R_opt = np.zeros_like(S_k) # zero wherever reserve is False    
#    R_opt[reserve] = np.minimum(S_k[reserve], ( np.add(N*S_k[reserve], R_L_total[reserve]) - C_max) / (N-1) )  # in case that our own demand is higher than the total capacity (should never actually happen)
            
    return pd.DataFrame(S_k, columns=[tenant], index=data.index), pd.DataFrame({'S':S_k,'U':U_k,'B':S_k_best,'Ropt':R_opt}, index=data.index), pd.DataFrame(R_L_total, index=data.index, columns=['R_L'])


def create_lag_set(data, lags, feat_name = None, full_set = False, keep_na = False):
    """ function to create lagged set of a univariate feature: shape = (#samples, #lags):
    data            -> column of data, should be dataFrame
    lags            -> numpy vector indicating which lags to use, i.e. [0,1,2,...,n]
    feat_name       -> name to give to each feature lag
    full_set        -> if true, will ignore lag vector and create the full lag set upto the highest lag in the lag vector
    keep_na         -> if true, will keep the na elements
    """
    X = data.copy()
    
    if isinstance(X, pd.DataFrame) is False:
        if isinstance(X, pd.Series) is True:
            X = pd.DataFrame(X)
        else:
            X = pd.DataFrame(np.array(X).T, columns = [feat_name])
            
    if feat_name is not None:
        X = X.rename({X.columns[0]:feat_name}, axis='columns')
                 
    X_new = pd.DataFrame()
         
    for features in X.columns:   
        if full_set is False:
            for i in lags:
                X_new['{}_{}'.format(features,i)] = X[features].shift(i)
        else:
            for i in range(0, max(lags)+1):
                X_new['{}_{}'.format(features,i)] = X[features].shift(i)
    
    if keep_na is False:
        X_new = X_new.dropna(axis=0)
    
    return X_new


def lag_corr(x,y, plot=False):
    """
    Function to compute autocorrelation equivilent of two different signals
    """
    X = x.copy()
    Y = y.copy()
    
    if isinstance(X, pd.Series) is True:
            X = pd.DataFrame(X)
    if isinstance(Y, pd.Series) is True:
            Y = pd.DataFrame(Y)        
    
    lag_corr = pd.Series([], dtype=float)
    
    for lag in X.columns:
        temp = pd.DataFrame(X[lag])
        lag_corr[lag] = pd.concat([temp, Y], axis=1, join='inner').corr(method = 'spearman').iloc[0,1]
    
    if plot is True:
        lags = np.arange(0,int(lag[2:])+1)
        plt.figure()
        plt.stem(lags, lag_corr, linefmt='black', basefmt='black')
        plt.title(' Correlation Plot of {} lags Vs Optimal Reservation'.format(lag[0]))
        plt.xlabel('lags')
        plt.ylabel('Correlation')    

    return lag_corr





def FS_get_best_lags(S_feedback, lag_groups, Feature_Selection = False, val_start=None, max_FS_lags=300, plot_stems=False):
    """
    This part of the code is still very much hard coded - it takes as inputs the time series features S,U,B and label
    and uses correlation to perform feature selection to get best lags of each feature. It then uses the defined groups 
    to create aggregated features
    """
    num_lags=sum(lag_groups)
    
    if Feature_Selection == 'correlation': # use correlation to determine optimal lages
        # first create univariate datasets where each feature is lagged up to some maximum 'max_FS_lags'
        S_lags = create_lag_set(S_feedback.loc[:val_start,'S'], lags=[0,max_FS_lags-1], feat_name = None, full_set = True, keep_na = False)
        B_lags = create_lag_set(S_feedback.loc[:val_start,'B'], lags=[0,max_FS_lags-1], feat_name = None, full_set = True, keep_na = False)
        U_lags = create_lag_set(S_feedback.loc[:val_start,'U'], lags=[0,max_FS_lags-1], feat_name = None, full_set = True, keep_na = False)
        
        # compute autocorrelation plots of each feature against the optimal label, can optionally plot the results
        S_lag_corr = lag_corr(S_lags,S_feedback['Ropt'], plot=plot_stems)
        U_lag_corr = lag_corr(U_lags,S_feedback['Ropt'], plot=plot_stems)
        B_lag_corr = lag_corr(B_lags,S_feedback['Ropt'], plot=plot_stems)
        
        # sort the results so that the most important lags are first
        S_lag_corr = S_lag_corr.abs().sort_values(ascending=False)
        U_lag_corr = U_lag_corr.abs().sort_values(ascending=False)
        B_lag_corr = B_lag_corr.abs().sort_values(ascending=False)
        
    else: # optionally, just define the best lags to be the most recent ones
        S_lag_corr = pd.Series(np.ones(num_lags), index=['S_{}'.format(i) for i in range(num_lags)]) 
        U_lag_corr = pd.Series(np.ones(num_lags), index=['U_{}'.format(i) for i in range(num_lags)]) 
        B_lag_corr = pd.Series(np.ones(num_lags), index=['B_{}'.format(i) for i in range(num_lags)])
        
        
    # using the predifined groups, bin the top lags into their groups and compute the average of each group -> new features
    S_grps = pd.Series([], dtype=float)
    U_grps = pd.Series([], dtype=float)
    B_grps = pd.Series([], dtype=float)
    
    counter = 0
    for grp in lag_groups: # store the index locations used to compute each aggregated group
        
        pointer = sum(lag_groups[:counter])
        S_grps[counter] = [int(sub.split('_')[1]) for sub in S_lag_corr.index[pointer : sum(lag_groups[:counter+1]) ].tolist()]  # S_lag_corr.index[pointer : sum(lag_groups[:counter+1]) ].tolist()
        U_grps[counter] = [int(sub.split('_')[1]) for sub in U_lag_corr.index[pointer : sum(lag_groups[:counter+1]) ].tolist()]  # U_lag_corr.index[pointer : sum(lag_groups[:counter+1])  ].tolist()
        B_grps[counter] = [int(sub.split('_')[1]) for sub in B_lag_corr.index[pointer : sum(lag_groups[:counter+1]) ].tolist()]  # B_lag_corr.index[pointer : sum(lag_groups[:counter+1])  ].tolist()
        counter = counter + 1
    
    
    # keep track of the largest index as will be needed during online testing -> need to offset test tart so that prior lags are relevant
    max_S = max([max(p) for p in S_grps])
    max_U = max([max(p) for p in U_grps])
    max_B = max([max(p) for p in B_grps])
    max_lag = max(max_S, max_U, max_B)
    
    # re-construct the lagged sets - will be used for trainging and initialising online simulation later, as well as background for SHAP (validaition set)
    S_lags = create_lag_set(S_feedback.loc[:,'S'], lags=[max_S], full_set = True, keep_na = False)
    U_lags = create_lag_set(S_feedback.loc[:,'U'], lags=[max_U], full_set = True, keep_na = False) 
    B_lags = create_lag_set(S_feedback.loc[:,'B'], lags=[max_B], full_set = True, keep_na = False)
    
    # aggregate lag groups into their means, to be used as final features to model
    aggr_S_lags = pd.DataFrame()
    aggr_U_lags = pd.DataFrame()
    aggr_B_lags = pd.DataFrame()
    
    counter = 0    
    for grp in lag_groups: # create the aggregated features by taking mean of each group
        
        cols = [ 'S_{}'.format(col) for col in S_grps[counter] ]
        S_means = S_lags.loc[:,cols].mean(axis=1)
        
        cols = [ 'U_{}'.format(col) for col in U_grps[counter] ]
        U_means = U_lags.loc[:,cols].mean(axis=1)
        
        cols = [ 'B_{}'.format(col) for col in B_grps[counter] ]
        B_means = B_lags.loc[:,cols].mean(axis=1)
        
        if grp == 1:
            aggr_S_lags['S_{}'.format( max(S_grps[counter]) )] = S_means
            aggr_U_lags['U_{}'.format( max(U_grps[counter]) )] = U_means
            aggr_B_lags['B_{}'.format( max(B_grps[counter]) )] = B_means
        else:    
            aggr_S_lags['S_{}-{}'.format(min(S_grps[counter]), max(S_grps[counter]))] = S_means
            aggr_U_lags['U_{}-{}'.format(min(U_grps[counter]), max(U_grps[counter]))] = U_means
            aggr_B_lags['B_{}-{}'.format(min(B_grps[counter]), max(B_grps[counter]))] = B_means
                
        counter = counter + 1
    

    return [S_lags, U_lags, B_lags], [aggr_S_lags, aggr_U_lags, aggr_B_lags], [S_grps, U_grps, B_grps], max_lag


#new_U_B = utf.update_aggr_lags(full_lags[1].loc[curr_time + pred_horizon,:], full_lags[2].loc[curr_time + pred_horizon,:], lag_groups, aggr_lags[1], aggr_lags[2])    
# then use updated features to make next prediction
#x_online.loc[curr_time + pred_horizon, new_U_B.index] = new_U_B


def update_aggr_lags(U_lag_row, B_lag_row, lag_groups, U_grps, B_grps):
    """ 
    Function to calculate the aggregated U and B lags - each time an online reservation is made, 
    new U and B values are recorded in a fully lagged dataset, then used to re-calculate aggregeted 
    features (used by the DNN) such as the mean of a set of these lags etc.
    """ 
    
    updated_U = pd.Series([], dtype=float)
    updated_B = pd.Series([], dtype=float)
    counter = 0    
    
    for grp in lag_groups:
        
        cols = [ 'U_{}'.format(col) for col in U_grps[counter] ]
        U_means = U_lag_row.loc[cols].mean()
        
        cols = [ 'B_{}'.format(col) for col in B_grps[counter] ]
        B_means = B_lag_row.loc[cols].mean()
        
        if grp == 1:
            updated_U['U_{}'.format( max(U_grps[counter]) )] = U_means
            updated_B['B_{}'.format( max(B_grps[counter]) )] = B_means
        else:    
            updated_U['U_{}-{}'.format(min(U_grps[counter]), max(U_grps[counter]))] = U_means
            updated_B['B_{}-{}'.format(min(B_grps[counter]), max(B_grps[counter]))] = B_means
                
        counter = counter + 1 
    
    return pd.concat([updated_U,updated_B])

early_stop = EarlyStopping(monitor = 'val_loss', mode = 'min', verbose = 1, patience = 50, restore_best_weights=True)

def create_DNN_model(x_shape):   
    model = Sequential()    
    model.add(Dense(25, activation='relu', input_shape=(x_shape,)))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(.3))   
    model.add(Dense(25,activation='relu')) ##relu
    model.add(Dropout(.3))    
    model.add(Dense(10))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(.2))     
    model.add(Dense(1, activation='relu'))
          
    loss_fn = tf.keras.losses.MeanAbsoluteError( reduction="auto", name="mean_absolute_error")
    #loss_fn = Huber( delta=1000 )
        
    model.compile( optimizer='adam', loss = loss_fn, metrics=[metrics.RootMeanSquaredError()] ) 
    
    return model


def create_linear_ANN(x_shape):   
    model = Sequential()    
    model.add(Dense(1, input_shape=(x_shape,)))    
    
    loss_fn = tf.keras.losses.MeanAbsoluteError( reduction="auto", name="mean_absolute_error")
    #loss_fn = Huber( delta=1000 )
    
    model.compile( optimizer='adam', loss = loss_fn, metrics=[metrics.RootMeanSquaredError()] ) 
    
    return model

class RegressionHyperModel(HyperModel):
    
    def __init__(self, input_shape):
        self.input_shape = input_shape
        
    def build(self, hp):
        model = Sequential()
        model.add(Dense(
            units=hp.Int('layer1_units', 16, 256, 16, default=16), 
            activation=hp.Choice('dense_activation_1', values=['relu', 'tanh', 'sigmoid'], default='relu'), 
            input_shape=self.input_shape))
        
        model.add(Dropout( 
            hp.Float('layer1_dropout', min_value=0.10, max_value=0.5, default=0.1, step=0.10) ))
        
        model.add(Dense( 
            units=hp.Int('layer2_units', 16, 256, 16, default=16),
                activation=hp.Choice('dense_activation_2', values=['relu', 'tanh', 'sigmoid'], default='relu') ))
        
        model.add(Dropout( 
            hp.Float('layer2_dropout', min_value=0.10, max_value=0.5, default=0.1, step=0.10) ))
        
        model.add(Dense( 
            units=hp.Int('layer3_units', 16, 128, 16, default=16),
                activation=hp.Choice('dense_activation_3', values=['relu', 'tanh', 'sigmoid'], default='relu') ))
        
        model.add(Dropout( 
            hp.Float('layer3_dropout', min_value=0.10, max_value=0.5, default=0.1, step=0.10) ))
        
        model.add(Dense(1, activation='relu'))
        
        # initialise class handle of huber loss function and pass delta as a config parameter
        loss_fn = Huber( delta=hp.Choice('huber_delta', values = [1.0, 10.0, 100.0, 250.0, 500.0] ) )
        
        model.compile( 
            optimizer='sgd', 
            loss = loss_fn, 
            metrics=[ metrics.RootMeanSquaredError() ] ) #
        
        return model
    
# Define the Huber loss so that it can be used with Keras
#from keras.losses import huber_loss
"""
def huber_loss_wrapper(**huber_loss_kwargs):
    def huber_loss_wrapped_function(y_true, y_pred):
        return huber_loss(y_true, y_pred, **huber_loss_kwargs)
    return huber_loss_wrapped_function    

"""


class ClearTrainingOutput(tf.keras.callbacks.Callback):
    def on_train_end(*args, **kwargs):
        IPython.display.clear_output(wait = True)
    

def get_hyper_model(x_train, y_train, x_val, y_val, method='bayesian', max_trials=20, epochs=25, batch_size=512, save_path=None, fname='KTuner', weights=None):
    """
    module to estimate best paramters of a model by using keras tuning 
    x_train...y_val:    data for training (panda objects)
    method:             method for tuning
    epochs:             max number of epochs to train each grid location for
    save_path:          the path directory to store the saved files, if none uses current default directory
    fname:              name of file to store trained models
    weights:            optional weights for each sample, ie weight most recent samples higher
    """
    
    if save_path is not None: # go to the specified directory
        import os
        orig_dir = os.getcwd()
        os.chdir(save_path)
    
    tf.random.set_seed(10)

    # create the hypermodel
    hypermodel = RegressionHyperModel((x_train.shape[1],))

    if method == 'hyperband':
        tuner = kt.tuners.Hyperband( hypermodel, seed=10, objective='val_loss', max_epochs=epochs, executions_per_trial=5, project_name=fname)
    elif method == 'bayesian':
        tuner = kt.tuners.BayesianOptimization( hypermodel, seed=10, objective='val_loss', max_trials=max_trials, executions_per_trial=5, project_name=fname)
    else: # do random search
        tuner = kt.tuners.RandomSearch( hypermodel, seed=10, objective='val_loss', max_trials=max_trials, executions_per_trial=5, project_name=fname)

    tuner.search(x_train.values, y_train.values, epochs=epochs, batch_size=batch_size, sample_weight = weights, 
                 validation_data=(x_val.values, y_val.values), verbose=0, callbacks = [ClearTrainingOutput()])
    
    
    #best_model = tuner.get_best_models(num_models=1)[0]
   
    # Get the optimal hyperparameters
    best_params = tuner.get_best_hyperparameters(num_trials = 1)[0]
    # Build the model with the optimal hyperparameters and train it on the data
    best_model = tuner.hypermodel.build(best_params)
    
    if save_path is not None: # return to the original directory
        os.chdir(orig_dir)
    
    return best_model, best_params


"""
    Function to plot model loss during training
"""
def plot_loss(history):
    
    plt.figure()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Training Loss', 'Validation Loss'], loc='upper right')
    plt.show()
    # plt.savefig(os.path.join(OUTPUT_PATH, 'train_vis_BS_'+str(BATCH_SIZE)+"_"+time.ctime()+'.png'))
    
    return



def calc_shap_values(explainer, result, x_input, indices, backgnd_size, file_name, save_loc=None,  batch_size=20):
    """ 
        function to compute shapley values in intervals/batches where each batch is saved as it is completed
        explainer:      kernel shap object that has already been initialised with background set, NB can be either from the 
                        SHAP library OR Alibi library, but if from Alibi will assume features have been grouped together
        result:         dataframe object where the shapley values will be saved in, may contain some shapley values already
        x_input:        x input data of which to calc the shapley values Dataframe
        indices:        the indices of x_input to compute shapley values of
        batch:          batch size between each interval
        file_name:      name of file used when saving batches
        save_loc:       location to the save shap values, if None will skip saving
    """
    
    test_samps = indices[~np.isin(indices, result.index)] # dont compute shapley values for indices that have already been computed
    if test_samps.size == 0:
        return result
    
    temp = result.copy()
    
    num_exps  = len(test_samps)
    num_batch = int((num_exps)/batch_size)  # number of complete batches to do
    remainder = num_exps%batch_size         # any remaining samples
    counter = 0
 
    for i in tqdm(range(num_batch)):
        
        samples = test_samps[counter:counter+batch_size]
        test_data = x_input.loc[samples]
        if str(type(explainer)).endswith("<class 'alibi.explainers.shap_wrappers.KernelShap'>"):            # if explainer is alibi object
            explanations = explainer.explain(test_data, nsamples=2**len(explainer.params['group_names']) )  # NB unlike SHAP, alibi automatically scales nsamples by the background size
            temp = temp.append( pd.DataFrame(explanations.shap_values[0], index=samples, columns=explainer.params['group_names']) )
        else:
            explanations = explainer.shap_values(test_data, nsamples=backgnd_size*(2**x_input.shape[1]), l1_reg='num_features({})'.format(x_input.shape[1]) )[0]
            temp = temp.append( pd.DataFrame(explanations, index=samples, columns=x_input.columns) )
        if save_loc is not None:
            with open("{}/{}.pkl".format(save_loc, file_name), "wb") as f:
                pickle.dump([temp],f) 
        counter += batch_size

    if remainder > 0:
        samples = test_samps[counter:counter+remainder]
        test_data = x_input.loc[samples]
        if str(type(explainer)).endswith("<class 'alibi.explainers.shap_wrappers.KernelShap'>"):            # if explainer is alibi object
            explanations = explainer.explain(test_data, nsamples=2**len(explainer.params['group_names']) )  # NB unlike SHAP, alibi automatically scales nsamples by the background size
            temp = temp.append( pd.DataFrame(explanations.shap_values[0], index=samples, columns=explainer.params['group_names']) )
        else:
            explanations = explainer.shap_values(test_data, nsamples=backgnd_size*(2**x_input.shape[1]), l1_reg='num_features({})'.format(x_input.shape[1]) )[0]
            temp = temp.append( pd.DataFrame(explanations, index=samples, columns=x_input.columns) )
        if save_loc is not None:
            with open("{}/{}.pkl".format(save_loc, file_name), "wb") as f:
                pickle.dump([temp],f) 
      
    return temp



def FI_summary_plot(shapley_values, feature_values, feature_names=None, background=None, title=None, xtitle=None, units=None, max_display = 30, alpha=1):
    """
    Plot the summary distributions of a feature importance method - similar to SHAP summary plot except with extra functionalities to allow plot 
    title, x label and mean & std deviation params of the background dataset to be used for creating the color scheme of the distributions.
    shapley_values          Pandas DataFrame or Numpy array with the importance scores of the explainer method, i.e shapley values for SHAP 
    feature_values          Pandas DataFrame or Numpy array with the feature values associated with each importance score (unormalised)
    feature_names           list of the feature names, if feature_values is a DataFrame the names can be extracted from its columns by leaving this blank
    background              Optional pandas DataFrame or Numpy array with the (unormalised) background used by the explainer. Used to adjust color range of plot.
    title                   optional title of the plot
    xtitle                  optional x label
    units                   As the shapley values are usually in units of the model output, it can be benificial to include the units in brackets on the x label axis
    max_display             param to control the amount of features shown in the plot
    alpha                   opacity of the colors
    """
    
    if str(type(shapley_values)).endswith("'pandas.core.frame.DataFrame'>"):
        shapley_values = shapley_values.values
        
    if str(type(feature_values)).endswith("'pandas.core.frame.DataFrame'>"):
        if feature_names is None:
            feature_names = list(feature_values.columns)
        feature_values = feature_values.values
           
    if str(type(background)).endswith("'pandas.core.frame.DataFrame'>"):
        background = background.values       
          
    if xtitle is None:
        if units is not None:
            xtitle = 'SHAP Value ({})'.format(units)
        else:
            xtitle = 'SHAP Value'
            
    np.random.seed(10)        
    feature_order = np.argsort(np.sum(np.abs(shapley_values), axis=0))          # determine the order of each feature in the plot by placing those with highest abs means on top
    feature_order = feature_order[-min(max_display, len(feature_order)):]
        
    #color = colors.blue_rgb
    color_bar_label='Relative feature value'    # NB. if the mean ans std_dev are None, the colors will be relative to the feature values associated with the shapley values
    axis_color="#333333"
    row_height = 0.4
    plt.gcf().set_size_inches(8, len(feature_order) * row_height + 2)   
    plt.title(title, fontsize=16)
        
    for pos, i in enumerate(feature_order):
        plt.axhline(y=pos, color="#cccccc", lw=0.5, dashes=(1, 5), zorder=-1)
        shaps = shapley_values[:, i]
        values = feature_values[:, i]
        
        inds = np.arange(len(shaps))
        np.random.shuffle(inds)
        values = values[inds]
        shaps = shaps[inds]
        
        N = len(shaps)
        nbins = 100
        quant = np.round(nbins * (shaps - np.min(shaps)) / (np.max(shaps) - np.min(shaps) + 1e-8)) # map each shapley value to a bin
        inds = np.argsort(quant + np.random.randn(N) * 1e-6)
        layer = 0
        last_bin = -1
        ys = np.zeros(N)
                   
        for ind in inds:
            if quant[ind] != last_bin:
                layer = 0
            ys[ind] = np.ceil(layer / 2) * ((layer % 2) * 2 - 1)
            layer += 1
            last_bin = quant[ind]
            
        ys *= 0.9 * (row_height / np.max(ys + 1))

        
        if background is None:                         
            vmin = np.nanpercentile(values, 1)              # trim the color range to avoid possible outliers that may cause it to collapse
            vmax = np.nanpercentile(values, 99)
            if vmin == vmax:
                vmin = np.min(values)
                vmax = np.max(values)      
            if vmin > vmax:                                     # fixes rare numerical precision issues
                vmin = vmax
    
            # plot any nan values in the interaction feature as grey
            nan_mask = np.isnan(values)
            plt.scatter(shaps[nan_mask], pos + ys[nan_mask], color="#777777", vmin=vmin,
                        vmax=vmax, s=16, alpha=alpha, linewidth=0,
                        zorder=3, rasterized=len(shaps) > 500)    
            # plot the non-nan values colored by the trimmed feature value
            cvals = values[np.invert(nan_mask)].astype(np.float64)
            cvals_imp = cvals.copy()
            cvals_imp[np.isnan(cvals)] = (vmin + vmax) / 2.0
            cvals[cvals_imp > vmax] = vmax
            cvals[cvals_imp < vmin] = vmin
            plt.scatter(shaps[np.invert(nan_mask)], pos + ys[np.invert(nan_mask)],
                        cmap=colors.red_blue, vmin=vmin, vmax=vmax, s=16,
                        c=cvals, alpha=alpha, linewidth=0,
                        zorder=3, rasterized=len(shaps) > 500)
        else: # use the background to center the color range
            vmin = np.nanpercentile(background[:, i], 1)              # trim the color range to avoid possible outliers that may cause it to collapse
            vmax = np.nanpercentile(background[:, i], 99)
            if vmin == vmax:
                vmin = np.min(background[:, i])
                vmax = np.max(background[:, i])
            if vmin > vmax:                                     # fixes rare numerical precision issues
                vmin = vmax             
            divnorm = mcolors.TwoSlopeNorm(vmin=vmin, vcenter=np.mean(background[:,i]), vmax=vmax) 
            # plot any nan values in the interaction feature as grey
            nan_mask = np.isnan(values)
            plt.scatter(shaps[nan_mask], pos + ys[nan_mask], color="#777777", norm=divnorm,
                        s=16, alpha=alpha, linewidth=0,
                        zorder=3, rasterized=len(shaps) > 500)
    
            # plot the non-nan values colored by the trimmed feature value
            cvals = values[np.invert(nan_mask)].astype(np.float64)
            cvals_imp = cvals.copy()
            cvals_imp[np.isnan(cvals)] = (vmin + vmax) / 2.0
            cvals[cvals_imp > vmax] = vmax
            cvals[cvals_imp < vmin] = vmin
            plt.scatter(shaps[np.invert(nan_mask)], pos + ys[np.invert(nan_mask)],
                        cmap=colors.red_blue, norm=divnorm, s=16,
                        c=cvals, alpha=alpha, linewidth=0,
                        zorder=3, rasterized=len(shaps) > 500)
                   
    # draw the color bar
    m = cm.ScalarMappable(cmap=colors.red_blue)
    m.set_array([0, 1])
    cb = plt.colorbar(m, ticks=[0, 1], aspect=1000)
    cb.set_ticklabels(['Low', 'High'])
    cb.set_label(color_bar_label, size=12, labelpad=0)
    cb.ax.tick_params(labelsize=11, length=0)
    cb.set_alpha(1)
    cb.outline.set_visible(False)
    bbox = cb.ax.get_window_extent().transformed(plt.gcf().dpi_scale_trans.inverted())
    cb.ax.set_aspect((bbox.height - 0.9) * 20)
    
    plt.axvline(x=0, color="#999999", zorder=-1)
    plt.gca().xaxis.set_ticks_position('bottom')
    plt.gca().yaxis.set_ticks_position('none')
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['left'].set_visible(False)
    plt.gca().tick_params(color=axis_color, labelcolor=axis_color)
    plt.yticks(range(len(feature_order)), [feature_names[i] for i in feature_order], fontsize=13)
   
    plt.gca().tick_params('x', labelsize=11)
    plt.ylim(-1, len(feature_order))
    plt.gca().tick_params('y', length=20, width=0.5, which='major')
    plt.gca().tick_params('x', labelsize=11)
    plt.ylim(-1, len(feature_order))
   
    plt.xlabel(xtitle)
    
    plt.show()
            
        
    return

# helper function to get mean of every nth row as a group
def groupedMean(array, period):
    result = [np.mean(array[x::period], axis=0) for x in range(period) ]
    return np.array(result)

# helper function to get median of every nth row as a group
def groupedMedian(array, period):
    result = [np.median(array[x::period], axis=0) for x in range(period) ]
    return np.array(result)

# helper function to convert time in minutes into hour format
def MinToHour(array): # convert minutes to hour:minute format
    return [str(dt.timedelta(minutes=int(i))) for i in array]

# helper function to remove year/month info of a datetime array and keep only day of week, hrs, mins info
def DayOfWeek(array): 
    array = array.dt.dayofweek*1440 + array.dt.hour*60 + array.dt.minute # keep only day of week, hrs, mins info
    day = (array/1440).astype(int) + 1 # add 1 so that range becomes 1-7, with 1 being a Monday
    hour = ((array%1440)/60).astype(int)
    minute = (array - (day-1)*1440 - hour*60).astype(int)

    #return ['2020-7-{} {}:{}:00'.format(day.iloc[i], hour.iloc[i],minute.iloc[i]) for i in range(len(array)) ]
    return [datetime(2020, 6, day.iloc[i], hour.iloc[i],minute.iloc[i],00) for i in range(len(array)) ] # convert to new datetime, where date is choosen so that the day number aligns with Monday being the 1st of the month

def combine_time_feats(x_data, shap_values):
    """
    helper function to combine the shapley values and x data of time based features decomposed into cos and sin components
    x_data          Pandas DataFrame of the (un-normalised/un-standardised) features used in the model.
    shap_values     Pandas DataFrame of the shapley values    
    """
    feat_names = list(x_data.columns)
    time_feats = [feat for feat in feat_names if 'Time'  in feat]       # get the time based features 

    x_combined = x_data.drop(time_feats, axis=1)
    # start with the time of day features, to get back the orginal time, we take arctan2(sin component, cos component) to get the 'angle' made by the time within the specific period (T). However,
    #  as numpy ranges everything between (-PI, +PI), we map the angle to (0,2pi) using (angle*2pi)%2pi. once we have the angle (in rads) we can get the actual time by multplying by (T/2pi).
    x_combined['Time of Day']  =  ((np.arctan2( x_data['Time of Day (sin)'], x_data['Time of Day (cos)'] ) + 2*np.pi) % (2*np.pi))*1440/(2*np.pi)
    x_combined['Time of Week'] =  ((np.arctan2( x_data['Time of Week (sin)'], x_data['Time of Week (cos)'] ) + 2*np.pi) % (2*np.pi))*10080/(2*np.pi)
        
    # combine the shapley components of each feature, as the shapley values are already in units of the model output, we can simply add them
    shap_combined = shap_values.drop(time_feats, axis=1)
    shap_combined['Time of Day'] = shap_values['Time of Day (sin)'] + shap_values['Time of Day (cos)']
    shap_combined['Time of Week'] = shap_values['Time of Week (sin)'] + shap_values['Time of Week (cos)']
    
    return x_combined, shap_combined


def shap_PDP(feat, shap_vals, x_data, model_predict, plot_time=False, t_format=None, rotation=0, title=None, units=None, x_units=None, feat_expected_value=None, model_expected_value=None, plot_zero=None, ax_color='black', nbins=30):
    """ function to plot the Shap PDP:
        feat:                   name of the feature to plot
        shap_vals               Dataframe object with the shapley values
        x_data                  Dataframe with the corresonding un-normalised x data of each the shapley value
        model_predict           predict function of the model, NB as x data is unstandardised, this may need to be a wrapper function or class
        plot_time               Specify as True when plotting time based data, in this case x_data should be pandas datetime vector
        t_format                specify the time format to be used, can be either be 'day' or 'week'
        rotation                used to rotate the x ticks of the pdp plot
        title                   title to use on plot
        units                   string giving the units of the shapley values
        x_units                 string giving the units of the x axis
        feat_expected_value     optional float giving the mean value of the feature being plot, should be the mean value of the feaute across the explainr background
        model_expected_value    optional float giving the mean output of the model, should be the mean output from the explainer background
        plot_zero               if true will plut the y = 0 line, useful alongside data mean to see of shap value changes at the intersection
    Notes: For a linear model, the Shap value for a feature (with a specific value=x) is given by its conventional PDP (at location x) minus the baseline value E[f(x)].
           This also means that when the feature is at its expected value (i.e. its mean) the shap PDP for a linear model intersects with 0 (i.e has shap value zero).
           Nb the x data corresponding to the shapley values are used for creating the scatter plot, while the expected outout and feature values are from the validation data
    """
    fig = plt.figure()
    ax = plt.gca()
    ax2 = ax.twinx()
    for pos in ['top', 'bottom', 'right', 'left']:
        ax.spines[pos].set_edgecolor(ax_color)
        ax2.spines[pos].set_edgecolor(ax_color)
    
    if plot_time is False:
        data = x_data.copy()
        x_min, x_max = np.min(data), np.max(data)
        ax2.hist(data, nbins, density=False, facecolor='black', alpha=0.2, range=(x_min,x_max) )     # plot histogram of the feature
        shap_vals = shap_vals[feat]
    else:                                                                                       # for time data, need to convert datetime to a numerical representations, such as minutes
        if t_format == 'day':     
            #shap_vals = (shap_vals['Time of Day (cos)']+shap_vals['Time of Day (sin)'])         # resultant shapley explanations for 'time of day' feature
            data = x_data.dt.hour*60 + x_data.dt.minute                                         # convert timestamps to 'minutes since start of each day'
            data = [dt.datetime.strptime(elem, '%H:%M:%S') for elem in MinToHour(data)]        
            x_min,x_max = np.min(data), np.max(data)
            ax2.hist(data,24,density=False,facecolor='black',alpha=0.2,range=(x_min,x_max) ) # plot histogram of the time data                                
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))                         # set the x tick format to show hours:mins                                 
            shap_vals = shap_vals[feat]
        elif t_format == 'week': 
            #data = x_data.dt.dayofweek*1440 + x_data.dt.hour*60 + x_data.dt.minute             # convert time to mins since start of week
            data = DayOfWeek(x_data)
            x_min,x_max = np.min(data), np.max(data)
            ax2.hist(data, 42, density=False, facecolor='black', alpha=0.2, range=(x_min,x_max) ) # plot histogram of the time data          
            #data = DayOfWeek(x_data) #[dt.datetime.strptime(elem, '%d:%H:%M') for elem in DayOfWeek(data)]          # convert to back to datetime format but keep only the dayofweek:hours:mins information
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%a %H:%M'))                      # set the x tick format to show days:hours:mins
            shap_vals = shap_vals[feat]
            #shap_vals = (shap_vals['Time of Week (cos)']+shap_vals['Time of Week (sin)'])         # resultant shapley explanations for 'time of week' feature
           
    # finally plot the PDP
    ax.set_xlim(x_min,x_max)
    ax.scatter(data, shap_vals, color='#1E88E5', s=10)   
    ax.set_ylabel('SHAP value ({})'.format(units))
    if x_units is  None:
        ax.set_xlabel(feat)
    else:
        ax.set_xlabel('{} ({})'.format(feat, x_units))
        
    ax.set_title(title)
    ymin,ymax = ax.get_ylim()
    img_height = ymax-ymin
        
    ax2.set_ylim(ymin,x_data.shape[0])
    ax2.yaxis.set_ticks([])
    plt.sca(ax)
    plt.xticks(rotation=rotation)
        
    if feat_expected_value is not None:
        ax3=ax.twiny()
        xmin,xmax = ax.get_xlim()
        ax3.set_xlim(xmin,xmax)  # NB this will essentially shift the axis, need to subtract xmin from any number being plot after this
        ax3.set_xticks([feat_expected_value])
        ax3.tick_params(axis='x', direction='in', labelsize=12, pad=-19, length=0)
        for pos in ['top', 'bottom', 'right', 'left']:
            ax3.spines[pos].set_edgecolor(ax_color)
        if plot_time is True:
            ax3.plot([(xmax-xmin)/2+xmin,(xmax-xmin)/2+xmin], [ymin,ymin+(img_height*0.90)], color="#999999", linestyle="--", linewidth=1, label='E[{}]'.format(feat))
            ax3.set_xticklabels(["E["+feat+"]"])   
        else:    
            ax3.plot([feat_expected_value-xmin,feat_expected_value-xmin], [ymin,ymin+(img_height*0.90)], color="#999999", linestyle="--", linewidth=1, label='E[{}]'.format(feat))
            ax3.set_xticklabels(["E["+feat+"]"])
                
    if model_expected_value is not None:
        ax4=ax.twinx()
        ymin,ymax = ax.get_ylim()
        xmin,xmax = ax.get_xlim()
        ax4.set_ylim(ymin,ymax)        
        ax4.set_yticks([model_expected_value])
        ax4.set_yticklabels(["E[f(x)]"])
        ax4.tick_params(axis='y', direction='in', labelsize=12, pad=-40)
        for pos in ['top', 'bottom', 'right', 'left']:
            ax4.spines[pos].set_edgecolor(ax_color)
        
        if t_format == 'day':
            xmin= dt.datetime(1900, 1, 1, 0,  0)
            xmax= dt.datetime(1900, 1, 1, 21, 0)
            ax4.plot([xmin,xmax], [model_expected_value,model_expected_value], color="#999999", linestyle="--", linewidth=1, label='E[f(x)]')
            ax4.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M')) 
        elif t_format == 'week':
            ax4.plot([xmin,xmax-1], [model_expected_value,model_expected_value], color="#999999", linestyle="--", linewidth=1, label='E[f(x)]')
            ax4.xaxis.set_major_formatter(mdates.DateFormatter('%a %H:%M')) 
            print(xmin, xmax)
        else:
            ax4.plot([xmin,xmax*0.85], [model_expected_value,model_expected_value], color="#999999", linestyle="--", linewidth=1, label='E[f(x)]')
            
    if plot_zero is True:
        xmin,xmax = ax.get_xlim()
        ymin,ymax = ax.get_ylim()
        ax5=ax.twinx()
        ax5.set_ylim(ymin,ymax)
        ax5.set_yticks([])
        for pos in ['top', 'bottom', 'right', 'left']:
            ax5.spines[pos].set_edgecolor(ax_color)  
        ax5.plot([xmin,xmax], [0,0], color="#999999", linestyle="--", linewidth=1)
        
    return