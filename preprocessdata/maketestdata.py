#THIS FILE MAKES THE TESTING DATA; ALL THE PREPROCESSING IS IDENTICAL, EXCEPT IT IS CENTERED ON ONE REGION AND THE BATCHING IS DONE CHRONOLOGICALLY, NOT RANDOMLY
import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1" 
N_max_cpus = 5
os.environ["TF_NUM_INTEROP_THREADS"] = str(N_max_cpus)
import copy
#This makes the test data
import numpy as np
import datetime
import os
from scipy import stats
import random
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


# take available along-track altimetry, randomly select up to n_sats_max sats on each day to use as input, bin average input sats onto zero-padded grid, save output sat(s) un-binned for use in loss function:
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




save_dir = '../testdata/'
batch_size = 5
n_batches = int(365/batch_size)
SSTdir=""
    
currentbatchnum=0
index=-1
for batch in range(n_batches):
    #This batches the data in batches of 5, each sample corresponing to a single date
    print(batch)
    batch_no = batch+currentbatchnum
    
    filename_invar = save_dir+f'/batch{batch_no}_invar.npy'
    filename_outvar = save_dir+f'/batch{batch_no}_outvar.npy'
    
    
    input_data_final = np.zeros((batch_size,N_t,n,n,2))
    output_npy = np.zeros((batch_size,N_t,1,3))
    max_length = 1
    regions = []

    for sample in range(batch_size):
        index+=1
        trying=True
        while trying:
            
            mid_date=datetime.date(test_year,1,1)+datetime.timedelta(index)
            raw_dir = './'
                
                
            files_raw = os.listdir(raw_dir)

            files_tracks = [f for f in files_raw if 'tracks' in f]
            if sst_high_res:
                files_sst = [f for f in files_raw if 'sst_' in f]
            else:
                files_sst = [f for f in files_raw if 'sst_' in f]

            output_data_final = []
            n_tot = []
            n_tot_fut=[]
            for t_loop in range(N_t):
                date_loop = mid_date - datetime.timedelta(days = N_t/2-t_loop)
                date_loop_fut=mid_date + datetime.timedelta(days = N_t/2+t_loop)

                ssh_files = [f for f in files_tracks if f'{date_loop}' in f]
                print(ssh_files)
                sst_files = [f for f in files_sst if f'{date_loop}' in f]
                ssh_files_fut = [f for f in files_tracks if f'{date_loop_fut}' in f]
                print(ssh_files_fut)

                n_tot.append(len(ssh_files)) # number of sats passing over on that day
                n_tot.append(len(ssh_files)) # number of sats passing over on that day
                n_tot_fut.append(len(ssh_files_fut)) # number of sats passing over on that day
                n_tot_fut.append(len(ssh_files_fut)) # number of sats passing over on that day
                try:
                    sst_loop=np.load(SSTdir+f'test/night{date_loop}.npy')
                    print('success')
                except:
                    sst_loop = np.zeros((n,n))
                else:
                    sst_loop=np.flip(sst_loop,axis=0)
                    sst_loop[np.isnan(sst_loop)]=0
                
                    
                    
                data_tracks = []
                data_tracks_fut=[]
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
                input_data_final[sample,t_loop,:,:,1] = sst_loop
                output_data_final.append(output_ssh)
            
            lengths = []
            for i in range(len(output_data_final)):
                lengths.append(output_data_final[i].shape[0])
            if max(lengths)>max_length:
                oldoutput=copy.deepcopy(output_npy)
                output_npy=np.zeros((batch_size,N_t,max(lengths),3))
                output_npy[:,:,:max_length,:]=oldoutput
                max_length=max(lengths)
            for i in range(N_t):
                output_npy[sample,i,:lengths[i],:] = output_data_final[i]
            sst_total = input_data_final[sample,:,:,:,1]
            if (np.sum(n_tot)/N_t>1):
                trying = False
    np.save(filename_invar, input_data_final)
    np.save(filename_outvar,output_npy)