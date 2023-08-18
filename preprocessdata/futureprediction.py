#IF MAKING TEST DATA, USE MAKETESTDATA INSTEAD
import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1" 
N_max_cpus = 5
os.environ["TF_NUM_INTEROP_THREADS"] = str(N_max_cpus)
import copy

# Code to pre-process the subsetted data into Keras-ready input-output pairs, save the pairs in batches of size ~100MB; each input file consists of 25 batches of 30 days each, and the output is the same 25 batches but 30 days in the future
#This script should be run twice; one with mode='validation' and one with mode='training'

import numpy as np
import datetime
import os
from scipy import stats
import random
import tensorflow as tf
import time
# function to list all files within a directory including within any subdirectories
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


# take available along-track altimetry, randomly select up to n_sats_max sats on each day to use as input, bin average input sats onto zero-padded grid
def bin_ssh(data_tracks, L_x,L_y, n, filtered = False):
    random.shuffle(data_tracks)
    tracks_in = np.concatenate(data_tracks[:len(data_tracks)], axis = 0)

    if filtered:
        input_grid, _,_,_ = stats.binned_statistic_2d(tracks_in[:,0], tracks_in[:,1], tracks_in[:,-2], statistic = 'mean', bins=n, range = [[-L_x/2, L_x/2],[-L_y/2, L_y/2]])
        input_grid = np.rot90(input_grid)
        input_grid[np.isnan(input_grid)] = 0
    else:
        input_grid, _,_,_ = stats.binned_statistic_2d(tracks_in[:,0], tracks_in[:,1], tracks_in[:,-1], statistic = 'mean', bins=n, range = [[-L_x/2, L_x/2],[-L_y/2, L_y/2]])
        input_grid = np.rot90(input_grid)

        input_grid[np.isnan(input_grid)] = 0    
    return input_grid
#save output sat(s) un-binned for use in loss function:
def bin_ssh_out(data_tracks_future, L_x,L_y, n, filtered = False):
    tracks_out=np.concatenate(data_tracks_future[:len(data_tracks_future)],axis = 0)

    if filtered:
        if len(data_tracks_fut)>1:
            output_tracks = np.stack((tracks_out[:,0],tracks_out[:,1],tracks_out[:,-2]),axis=-1)
            output_tracks[np.isnan(output_tracks)] = 0
        else:
            output_tracks = np.zeros((1,3))
    else:
        if len(data_tracks_fut)>1:
            output_tracks = np.stack((tracks_out[:,0],tracks_out[:,1],tracks_out[:,-1]),axis=-1)
            output_tracks[np.isnan(output_tracks)] = 0
        else:
            output_tracks = np.zeros((1,3))
    return output_tracks
    

sats_all = ['alg','tpn','tp','s3b','s3a','j3','j2n','j2g','j2','j1n','j1g','j1','h2b','h2ag','h2a','g2','enn','en','e2','e1g','al','c2','c2n','s3b','s6a','j3n','h2b']
# satellites_nrt = ['s3a','s3b','s6a','j3','j3n','al','c2n','h2b']

test_sats = ['alg','al'] # independent test satellite used for testing purposes, withhold from training data for all years

sats = [s for s in sats_all if s not in test_sats]


N_t = 30 # length of single input time series in days
n = 128 # no. grid points per side of domain
L_x = 960e3 # size of domain
L_y = 960e3  # size of domain
filtered = False # whether to use the 65km band-pass filtered or unfiltered SSH observations
sst_high_res = True # True = L4 MUR SST with MW+IR (highest spatial resolution but time-varying effective resolution since IR resolution depends on clouds), False = L4 MUR SST with just MW (lower res but more constant spatial resolution)
SST=True #Determines whether or not we preprocessed the SST data as well
test_year = 2017


regionoptions= "PATH" #INSERT PATH HERE
if SST:
    SSTdir="PATH" #INSERT PATH HERE
# THIS DEFINES THE TRAIN-VALIDATION-TEST SPLIT IN TERMS OF DATES (n.b. the sats in test_sats are withheld for any dates)
######################
start_date = datetime.date(2010,1,1)
end_date = datetime.date(2022,12,31)
n_days = (end_date-start_date).days + 1
val_dates = []
for t in range(73-30):
    val_dates.append(datetime.date(2010,1,1)+datetime.timedelta(days = 15+t))
    val_dates.append(datetime.date(2011,1,1)+datetime.timedelta(days = 73)+datetime.timedelta(days = 15+t))
    val_dates.append(datetime.date(2012,1,1)+datetime.timedelta(days = 2*73)+datetime.timedelta(days = 15+t))
    val_dates.append(datetime.date(2013,1,1)+datetime.timedelta(days = 3*73)+datetime.timedelta(days = 15+t))
    val_dates.append(datetime.date(2014,1,1)+datetime.timedelta(days = 4*73)+datetime.timedelta(days = 15+t))
    val_dates.append(datetime.date(2016,1,1)+datetime.timedelta(days = 15+t))
    val_dates.append(datetime.date(2018,1,1)+datetime.timedelta(days = 73)+datetime.timedelta(days = 15+t))
    val_dates.append(datetime.date(2019,1,1)+datetime.timedelta(days = 2*73)+datetime.timedelta(days = 15+t))
    val_dates.append(datetime.date(2020,1,1)+datetime.timedelta(days = 3*73)+datetime.timedelta(days = 15+t))
    val_dates.append(datetime.date(2021,1,1)+datetime.timedelta(days = 4*73)+datetime.timedelta(days = 15+t))
test_dates = []
for t in range(365):
    test_dates.append(datetime.date(test_year,1,1)+datetime.timedelta(days=t))

train_dates = []
for t in range(n_days-15):
    check_date = start_date+ datetime.timedelta(days=t)
    diffs_val = [np.abs((check_date-date).days) for date in val_dates]
    diffs_test = [np.abs((check_date-date).days) for date in test_dates]
    
    if (np.min(diffs_val)>=60) and (np.min(diffs_test)>=60): #made it 60 days to prevent input-output pair overlap
        train_dates.append(check_date)

###################
save_dir= #
mode = 'validation'
if mode == 'training':
    save_dir = save_dir+'training'
    n_batches = 2000
    batch_size = 25
elif mode == 'validation':
    save_dir = save_dir+'validation'
    n_batches = 1000
    batch_size = 25
    
currentbatchnum=0
for batch in range(n_batches):
    
    print(batch)
    batch_no = batch+currentbatchnum
    
    filename_invar = save_dir+f'/batch{batch_no}_invar.npy'
    filename_outvar = save_dir+f'/batch{batch_no}_outvar.npy'
    
    
    input_data_final = np.zeros((batch_size,N_t,n,n,2))
    output_npy = np.zeros((batch_size,N_t,1,3))
    max_length = 1
    regions = []

    for sample in range(batch_size):
        print(sample)
        trying=True
        while trying:
            #Randomly selections a date and a region in the available dates
            if mode=='training':
                available_dates = train_dates
            elif mode=='validation':
                available_dates = val_dates
            if mode == 'training' or mode=='validation':
                mid_date = random.choice(available_dates)
                r = np.random.choice(regionoptions)
            raw_dir = f'/dat1/smart1n/aviso-data/global training data/raw/{r}/'
            regions.append(r)
                
                
            files_raw = os.listdir(raw_dir)

            files_tracks = [f for f in files_raw if 'tracks' in f]
            files_tracks = [f for f in files_tracks if not any(substring in f for substring in test_sats)] # removes the test sat for all years

            output_data_final = []
            n_tot = []
            n_tot_fut=[]
            for t_loop in range(N_t):
                #We now loop through each of the 30 days that we are making samples for; date_loop_fut is 30 days ahead of date_loop
                date_loop = mid_date - datetime.timedelta(days = N_t/2-t_loop)
                date_loop_fut=mid_date + datetime.timedelta(days = N_t/2+t_loop)

                #Finds the files that correspond to date_loop and date_loop future
                ssh_files = [f for f in files_tracks if f'{date_loop}' in f]
                ssh_files_fut = [f for f in files_tracks if f'{date_loop_fut}' in f]
            
                n_tot.append(len(ssh_files)) # number of sats passing over on that day
                n_tot_fut.append(len(ssh_files_fut)) # number of sats passing over on that day
                
                #This loads the pre-processed SST file (if it exists), otherwise it loads an array of zeroes
                if SST and date_loop>datetime.date(2017,12,15):
                    try:
                        sst_loop=np.load(SSTdir+f'{r}/night{date_loop}.npy')
                    except:
                        sst_loop = np.zeros((n,n))
                    else:
                        sst_loop=np.flip(sst_loop,axis=0)
                        sst_loop[np.isnan(sst_loop)]=0
                else:
                    sst_loop = np.zeros((n,n))
                    
                    
                data_tracks = []
                data_tracks_fut=[]
                #This now bins the input data and preprocesses the track in the future
                for f in ssh_files:
                    try:
                        data_tracks.append(np.load(raw_dir+f)[1:,:])
                    except: 
                        data_tracks.append(np.zeros((1,3)))
                for f in ssh_files_fut:
                    try:
                        data_tracks_fut.append(np.load(raw_dir+f)[1:,:])
                    except: 
                        data_tracks_fut.append(np.zeros((1,3)))
                if len(data_tracks)>0:
                    input_ssh = bin_ssh(data_tracks,L_x,L_y, n, filtered)
                else:
                    input_ssh = np.zeros((n,n))
                if len(data_tracks_fut)>0:
                    output_ssh = bin_ssh_out(data_tracks_fut,L_x,L_y, n, filtered)
                else:
                    output_fut = np.zeros((1,3))
                input_data_final[sample,t_loop,:,:,0] = input_ssh
                if SST:
                    input_data_final[sample,t_loop,:,:,1] = sst_loop
                output_data_final.append(output_ssh)
            
            lengths = []
            #This section of code makes sure all the output track for each batch can fit in the output numpy array by making it larger if necessary
            for i in range(len(output_data_final)):
                lengths.append(output_data_final[i].shape[0])
            if max(lengths)>max_length:
                oldoutput=copy.deepcopy(output_npy)
                output_npy=np.zeros((batch_size,N_t,max(lengths),3))
                output_npy[:,:,:max_length,:]=oldoutput
                max_length=max(lengths)
            for i in range(N_t):
                output_npy[sample,i,:lengths[i],:] = output_data_final[i]
            # condition to exclude examples with insufficient satellites:
            if (np.sum(n_tot)/N_t>1):
                trying = False

    np.save(filename_invar, input_data_final)
    np.save(filename_outvar,output_npy)