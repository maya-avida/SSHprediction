from signal import signal, SIGPIPE, SIG_DFL  
import os
import tensorflow as tf
from tensorflow import keras
import tensorflow_addons as tfa
import numpy as np
import copy
import time
from numba import cuda

from src.generators import *
from src.models import *
from src.losses import *
#This is the ConvLSTM model

def GetListOfFiles(dirName, ext = '.nc'):
    # create a list of file and sub directories 
    # names in the given directory 
    listOfFile = os.listdir(dirName)
    allFiles = list()
    # Iterate over all the entries
    for entry in listOfFile:
        # Create full path
        fullPath = os.path.join(dirName, entry)
        # If entry is a directory then get the list of files in this directory 
        if os.path.isdir(fullPath):
            allFiles = allFiles + GetListOfFiles(fullPath)
        else:
            if fullPath.endswith(ext):
                allFiles.append(fullPath)               
    return allFiles

n_t = 30
n_train = 1000
n_val = 80
region = 'gulf stream' # 'global' or 'gulf stream'
mode = 'SLA' # 'SLA-SST' or 'SST'
oneoutput=True #set this to True if you only want to predict the first day after the input period
mode2= 'future prediction' #Are we recreating Scott's problem (interpolation) or trying to predict future SSH (future prediction)
print(mode,mode2)

if mode2=='interpolation':
    model_weights_dir = '/home/jovyan/deep-learning-ssh-mapping-JAMES-paper/src/model_weights/'
    data_dir_gs='/home/jovyan/pre-processed/'
    if mode=='SLA-SST':
        experiment_name = 'convlstm_sla_sst_' + f'{n_t}days_{n_train}samples'
    elif mode=='SLA':
        experiment_name = 'convlstm_sla_' + f'{n_t}days_{n_train}samples'
    gs_mean_ssh=np.load("gs_mean_ssh.npy")
    gs_std_ssh=np.load("gs_std_ssh.npy")
    gs_mean_sst=np.load("gs_mean_sst.npy")
    gs_std_sst=np.load("gs_std_sst.npy")

elif mode2=='future prediction':
    model_weights_dir = '/home/jovyan/deep-learning-ssh-mapping-JAMES-paper/src/model_weights_future/'
    data_dir_gs = '/home/jovyan/pre-processed-future-fixed/'
    if mode=='SLA-SST':
        experiment_name = 'convlstm_sla_sst_future_' + f'{n_t}days_{n_train}samples'
    elif mode=='SLA':
        experiment_name = 'convlstm_sla_future_' + f'{n_t}days_{n_train}samples'
    gs_mean_ssh=np.load("gs_mean_ssh_future.npy")
    gs_std_ssh=np.load("gs_std_ssh_future.npy")
    gs_mean_sst=np.load("gs_mean_sst_future.npy")
    gs_std_sst=np.load("gs_std_sst_future.npy")

n_epochs = 100
        
batch_size = 25
# global statistics:
#global_mean_ssh = 0.064
#global_std_ssh = 0.0712
#global_mean_sst = 290.1
#global_std_sst = 1.433
#statistics:

if region == 'global':
    mean_ssh = global_mean_ssh
    std_ssh = global_std_ssh
    mean_sst = global_mean_sst
    std_sst = global_std_sst
    stats = (mean_ssh, std_ssh, mean_sst, std_sst)
    datadir = data_dir_global
elif region == 'gulf stream':
    mean_ssh = gs_mean_ssh
    std_ssh = gs_std_ssh
    mean_sst = gs_mean_sst
    std_sst = gs_std_sst
    stats = (mean_ssh, std_ssh, mean_sst, std_sst)
    datadir = data_dir_gs
else:
    print('invalid region')
    exit()

params = {'oneoutput': oneoutput,
            'dim': (n_t,128,128),
          'batch_size': batch_size,
          'n_channels': 1,
          'shuffle': True,
          'val':'training/',
          'datadir':datadir,
         'stats':stats}

params_val = {'dim': (n_t,128,128),
          'batch_size': batch_size,
          'n_channels': 1,
          'shuffle': False,
              'val':'validation/',
          'datadir':datadir,
              'oneoutput': oneoutput,
             'stats':stats}


#FIX
val_ids = []
for i in range(n_val):
    val_ids.append(f'{i}')
train_ids = []
for i in range(n_train):
    train_ids.append(f'{i}')

if mode == 'SLA':
    model = create_ConvLSTM_SLA(n_t,oneoutput)
    if oneoutput:
        model.compile(loss = tracked_mse_interp_grads_gulf_stream_oneoutput(), optimizer=keras.optimizers.Adam(lr = 5e-4))
    elif region == 'gulf stream':
        model.compile(loss = tracked_mse_interp_grads_gulf_stream(), optimizer=keras.optimizers.Adam(lr = 5e-4))
    elif region == 'global':
        model.compile(loss = tracked_mse_interp_grads_global(), optimizer=keras.optimizers.Adam(lr = 5e-4))
    model.summary()
    model.load_weights(model_weights_dir+experiment_name+'.h5')
    training_generator = DataGenerator_ssh_interp(train_ids, **params)
    validation_generator = DataGenerator_ssh_interp(val_ids, **params_val)


elif mode == 'SLA-SST':
    model = create_ConvLSTM_SLA_SST(n_t)
    if region == 'gulf stream':
        model.compile(loss = tracked_mse_interp_grads_gulf_stream(), optimizer=keras.optimizers.Adam(lr = 5e-4))
    elif region == 'global':
        model.compile(loss = tracked_mse_interp_grads_global(), optimizer=keras.optimizers.Adam(lr = 5e-4))
    model.summary()
    training_generator = DataGenerator_ssh_sst_interp(train_ids, **params)
    validation_generator = DataGenerator_ssh_sst_interp(val_ids, **params_val)
    
else:
    print('invalid mode')
    exit()

stopping = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta = 0, patience = 10, restore_best_weights=True)
saving = keras.callbacks.ModelCheckpoint(model_weights_dir+experiment_name+'.h5', save_weights_only=True, monitor='val_loss', mode = 'min',save_best_only= True)
lr = keras.callbacks.ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.1,
    patience=8,
    verbose=1,
    mode="auto",
    min_delta=0.0001,
    cooldown=0,
    min_lr=1e-4,
)
history_log = keras.callbacks.CSVLogger(experiment_name+'_log'+'.csv', separator=",", append=True)
callbacks = [stopping, saving, lr, history_log]
history = model.fit(training_generator, validation_data = validation_generator, epochs = n_epochs, callbacks = callbacks, use_multiprocessing=False,workers=1,max_queue_size=100)
val_loss = history.history['val_loss']
train_loss = history.history['loss']
np.save(experiment_name + 'history.npy', np.stack((train_loss, val_loss), axis = -1))
prediction = model.predict(validation_generator)
np.save(experiment_name + 'prediction.npy', prediction)
