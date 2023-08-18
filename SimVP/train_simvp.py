import os
import sys
import copy
# os.environ["CUDA_VISIBLE_DEVICES"]=""
os.environ['TMPDIR'] = '/dat1/smart1n/tmp'

sys.path.append('/dat1/smart1n/OpenSTL/openstl')

from models.simvp_model import SimVP_Model_no_skip
from models.simvp_model import SimVP_Model_no_skip_sst
import numpy as np
# from src.models import *
from pytorch_losses import *
import torch
from torch import optim
# import torch.utils.data as data
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F

import matplotlib.pyplot as plt

import csv
# from apex.optimizers import FusedAdam

class LossLoggerCallback:
    def __init__(self, filename):
        self.filename = filename
        self.train_losses = []
        self.val_losses = []

    def __call__(self, epoch, train_loss, val_loss):
        self.train_losses.append(train_loss)
        self.val_losses.append(val_loss)
        
        # Save the losses to a CSV file
        with open(self.filename, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Epoch', 'Train Loss', 'Val Loss'])
            for i in range(len(self.train_losses)):
                writer.writerow([i+1, self.train_losses[i], self.val_losses[i]])
        
        print(f'Epoch {epoch}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}')



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # Specify the GPU device

batch_size=25
n_t = 30

mode = 'SLA-SST' # 'SLA-SST' or 'SST'
mode2= 'future prediction'
hyperparamtuning=False #If set to True, this runs training many times with randomly chosen hyperparameters (you can set the ranges below), with a smaller batch size, to find the best hyperparameters
pretrained=False #If you've run this experiment before and you want to load the old weights, set this parameter to True
weighted=False #If set to True, it weights the first few days of the loss function more heavily than the last few days


#FILL IN THE DATA DIRECTORY AND WHERE YOU CALCUALTED THE MEAN AND STANDARD DEVIATION
if mode=='SLA-SST':
    data_dir_gs = '/home/jovyan/preproSST/'
    gs_mean_ssh=np.load("../../deep-learning-ssh-mapping-JAMES-paper/gs_mean_ssh_futureSST.npy")
    gs_std_ssh=np.load("../../deep-learning-ssh-mapping-JAMES-paper/gs_std_ssh_futureSST.npy")
    gs_mean_sst=np.load("../../deep-learning-ssh-mapping-JAMES-paper/gs_mean_sst_futureSST.npy")
    gs_std_sst=np.load("../../deep-learning-ssh-mapping-JAMES-paper/gs_std_sst_futureSST.npy")
elif mode=='SLA':
    data_dir_gs = '/home/jovyan/prepro-new/'
    gs_mean_ssh=np.load("../../deep-learning-ssh-mapping-JAMES-paper/gs_mean_ssh_future_new_regions.npy")
    gs_std_ssh=np.load("../../deep-learning-ssh-mapping-JAMES-paper/gs_std_ssh_future_new_regions.npy")
    gs_mean_sst=np.load("../../deep-learning-ssh-mapping-JAMES-paper/gs_mean_sst_future_new_regions.npy")
    gs_std_sst=np.load("../../deep-learning-ssh-mapping-JAMES-paper/gs_std_sst_future_new_regions.npy")

    
mean_ssh = gs_mean_ssh
std_ssh = gs_std_ssh
mean_sst = gs_mean_sst
std_sst = gs_std_sst
stats = (mean_ssh, std_ssh, mean_sst, std_sst)
datadir = data_dir_gs

params = {'dim': (n_t,128,128),
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
             'stats':stats}

#This is the dataloader
class SSH_Dataset(Dataset):
    def __init__(self, list_IDs,mode,batch_size=25,dim=(6,128,128), n_channels=1, shuffle=True, val = 'training', datadir = '30 days 1920km 128x128/training_processed/', stats = (0,1,0,1)):
        self.mode=mode
        self.dim = dim
        self.batch_size = batch_size
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.shuffle = shuffle
        self.val = val
        self.datadir = datadir
        self.stats = stats
    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, idx):
        mean_ssh, std_ssh, mean_sst, std_sst = self.stats
        
        # Initialization
        
        X1 = np.empty((self.batch_size, self.dim[0], 128, 128, 1))
        X2 = np.empty((self.batch_size, self.dim[0], 128, 128, 1))

        # X2 = np.empty((self.batch_size, self.dim[0], 128, 128, 1))
        # X3 = np.empty((self.batch_size, 30, 128, 128, 1))
        # Y = np.empty((self.batch_size, *self.dim, 1))
        Y_length = []
        Y_list = []
        if mode=="SLA-SST":
            
            ID=self.list_IDs[idx] #ID
            
            input_ = np.load(self.datadir + self.val +'batch' + ID + "_invar.npy")
            
            ssh = copy.deepcopy(input_[:,:,:,:,0])
            ssh[np.isnan(ssh)] = 0
            ssh[ssh!=0] = (ssh[ssh!=0]-mean_ssh)/std_ssh

            sst = copy.deepcopy(input_[:,:,:,:,1])
            sst[np.isnan(sst)] = 0

            sst[sst<273]=0
            sst[sst!=0] = (sst[sst!=0]-mean_sst)/std_sst


            X1[:,:,:,:,0] = ssh[:,:self.dim[0],:,:]
            X2[:,:,:,:,0] = sst[:,:self.dim[0],:,:]

            ssh_out = np.load(self.datadir + self.val +'batch' + ID + "_outvar.npy")

            ssh_out = ssh_out[:self.dim[0],]

            ssh_out[np.isnan(ssh_out)] = 0
            x = copy.deepcopy(ssh_out[:,:,:,0])
            x[x!=0] = ((x[x!=0]+0.5*960e3)/960e3)*(128-1)
            y = copy.deepcopy(ssh_out[:,:,:,1])
            #OUTPUT DIMENSIONS: sample, day, maximum number of satellites, x y and SSH

            y[y!=0] = ((-y[y!=0]+0.5*960e3)/960e3)*(128-1)
            sla = copy.deepcopy(ssh_out[:,:,:,2])
            sla[sla!=0] = (sla[sla!=0]-mean_ssh)/std_ssh
            outvar = np.stack((x,y,sla),axis = -1)
            Y=outvar

            X = np.stack([X1,X2],axis=2)# simvp wants (batch size, time, channels, height, width)

            invar = torch.from_numpy(X).half()
            outvar = torch.from_numpy(Y).half()
            return invar, outvar
        elif mode=="SLA":
            ID=self.list_IDs[idx] #ID
            input_ = np.load(self.datadir + self.val +'batch' + str(ID) + "_invar.npy")
            ssh = copy.deepcopy(input_[:,:,:,:,0])
            ssh[np.isnan(ssh)] = 0
            ssh[ssh!=0] = (ssh[ssh!=0]-mean_ssh)/std_ssh


            X1[:,:,:,:,0] = ssh[:,:self.dim[0],:,:]

            ssh_out = np.load(self.datadir + self.val +'batch' + str(ID) + "_outvar.npy")

            ssh_out = ssh_out[:self.dim[0],]

            ssh_out[np.isnan(ssh_out)] = 0
            x = copy.deepcopy(ssh_out[:,:,:,0])
            x[x!=0] = ((x[x!=0]+0.5*960e3)/960e3)*(128-1)
            y = copy.deepcopy(ssh_out[:,:,:,1])
            #OUTPUT DIMENSIONS: sample, day, maximum number of satellites, x y and SSH

            y[y!=0] = ((-y[y!=0]+0.5*960e3)/960e3)*(128-1)
            sla = copy.deepcopy(ssh_out[:,:,:,2])
            sla[sla!=0] = (sla[sla!=0]-mean_ssh)/std_ssh
            outvar = np.stack((x,y,sla),axis = -1)
            Y=outvar

            X = X1

            X=np.swapaxes(X,2,4)#(Batch size, frames, channels, columns, rows)
            X=np.swapaxes(X,3,4) #batches, frames, channels, rows, columns

            invar = torch.from_numpy(X).half()
            outvar = torch.from_numpy(Y).half()

            return invar, outvar
                
        ##########
        
        
    


weight_dir ='/home/jovyan/OpenSTL/openstl/weights/'#FILL IN WHERE YOU WANT THE WEIGHTS TO BE

n_t = 30
L_x = 960e3
L_y = 960e3
n=128
batch_size = 25 # DON'T CHANGE, THIS IS FIXED IN THE PRE-PROCESSING TO BE 1 BATCH PER FILE

torch.cuda.empty_cache()
lr=0.001

#SET TO THE NUMBER OF BATCHES YOU WANT
if hyperparamtuning:
    n_train_batches = 200 #training batches
    n_val=10 #validation batches
else:  
    n_train_batches = 2000#int(ns*1e3/batch_size)
    n_val=500
    
n_train=n_train_batches
n_val_batches = n_val

#Load the training and validation data
val_ids = []
for i in range(n_val):
    val_ids.append(f'{i}')
train_ids = []
for i in range(n_train):
    train_ids.append(f'{i}')
train_dataset = SSH_Dataset(train_ids,mode,**params)
train_dataloader = DataLoader(train_dataset, batch_size=None, num_workers=0, shuffle=True, pin_memory=True)
val_dataset = SSH_Dataset(val_ids,mode,**params_val)
val_dataloader = DataLoader(val_dataset, num_workers=0, batch_size=None, shuffle=False, pin_memory=True)


maxpatience=5 #this is the number of epochs that the validation loss can have plateaued before the training is cut off
if not hyperparamtuning:
    
    hyper_params=np.load('newregionbest_hyperparameters.npy') #AFTER HYPERPARAMETERTUNING, FILL IN WITH THE HYPERPARAMETERS YOU SAVED
    l2=hyper_params[0] #\lambda_2, the regularization
    dropout=hyper_params[1]
    droppath=hyper_params[2]
    if weighted:
        #Fill in with your specific experiment name
        experiment_name = f'simvp_weighted_l2_'+str(l2)+'_dropout'+str(dropout)+'_droppath_'+str(droppath)
    else:
        experiment_name = f'simvp_l2_'+str(l2)+'_dropout'+str(dropout)+'_droppath_'+str(droppath)
    if mode=='SLA-SST':
        experiment_name=experiment_name+'SST'
    
    num_epochs = 100 #FILL IN WITH THE NUMBER OF EPOCHS
    
    callback = LossLoggerCallback(experiment_name+'log.csv')#logs the loss in a csv file
    #Load the model
    if mode=='SLA':
        model = SimVP_Model_no_skip(in_shape=(n_t,1,128,128),model_type='gsta',hid_S=8,hid_T=128,drop=hyper_params[1],drop_path=hyper_params[2]).to(device)
    elif mode=='SLA-SST':
        model=SimVP_Model_no_skip_sst(in_shape=(n_t,2,128,128),model_type='gsta',hid_S=8,hid_T=128,drop=hyper_params[1],drop_path=hyper_params[2]).to(device)
    #Load the parameters (if applicable)
    if pretrained:
        model.load_state_dict(torch.load(weight_dir+experiment_name+".pt"))
    #Optimizer and scheduler; can choose a different learning rate scheduler if desired
    optimizer = optim.Adam(model.parameters(), lr=lr,weight_decay=hyper_params[0])
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=lr, steps_per_epoch=len(train_dataloader), epochs=num_epochs)
    
    best_val_loss = float('inf')  # Initialize the best validation loss

    use_amp = True

    scaler = torch.cuda.amp.GradScaler(enabled=use_amp) # to allow use of floating point 16 training, slightly faster
    patience=0
    for epoch in range(num_epochs):
        if patience>maxpatience:
            break
        model.train()
        train_loss = 0.0
        num_batches = 0
        for torch_input_batch, torch_output_batch in train_dataloader:
            optimizer.zero_grad(set_to_none=True)
            if mode=='SLA-SST':
                torch_input_batch = torch_input_batch.squeeze().cuda()
                torch_output_batch = torch_output_batch.squeeze().cuda()
            elif mode=='SLA':
                torch_input_batch = torch_input_batch.squeeze(0).cuda() #If mode == 'SLA' then there is only one input channel, which we don't want to squeeze; the goal here is just to make sure that the input is the shape that SimVP expects: (batch size, days, channels, height, width)
                torch_output_batch = torch_output_batch.squeeze(0).cuda()
            with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=use_amp):
                torch_input_batch=torch_input_batch.half()
                outputs = model(torch_input_batch)
                if weighted:
                    loss = torch_tracked_mse_interp_weighted(outputs, torch_output_batch)
                else:
                    loss = torch_tracked_mse_interp(outputs, torch_output_batch)
            loss=loss.half()
            grads=scaler.scale(loss)
            grads.float().backward()
            scaler.unscale_(optimizer)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            train_loss += loss.item()
            num_batches += 1

            # Print progress
            if num_batches%100==1:
                print(f'Train Epoch: {epoch+1}/{num_epochs} '
                      f'Batch: {num_batches}/{n_train_batches} '
                      f'Loss: {train_loss/num_batches:.4f}')

        train_loss /= num_batches

        # Validation phase
        model.eval()
        val_loss = 0.0
        num_val_batches = 0

        with torch.no_grad():
            for torch_input_batch, torch_output_batch in val_dataloader:

                if mode=='SLA-SST':
                    torch_input_batch = torch_input_batch.squeeze().cuda()
                    torch_output_batch = torch_output_batch.squeeze().cuda()
                elif mode=='SLA':
                    torch_input_batch = torch_input_batch.squeeze(0).cuda()
                    torch_output_batch = torch_output_batch.squeeze(0).cuda()
                with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=use_amp):
                    val_preds = model(torch_input_batch)
                    if weighted:
                        val_loss += torch_tracked_mse_interp_weighted(val_preds, torch_output_batch).item()
                    else:
                        val_loss += torch_tracked_mse_interp(val_preds, torch_output_batch).item()
                num_val_batches += 1
                if num_val_batches%10==1:
                    print(f'Validation Loss {num_val_batches}/{n_val_batches}: {val_loss/num_val_batches:.4f}')

        val_loss /= num_val_batches
        callback(epoch + 1, train_loss, val_loss)
        # Check if validation loss improved
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint = {"model": model.state_dict(),
              "optimizer": optimizer.state_dict(),
              "scaler": scaler.state_dict()}
            torch.save(model.state_dict(), weight_dir+experiment_name+".pt")
            patience=0
        else:
            patience+=1
elif hyperparamtuning:
    from random import uniform
    import random
    #hyperparameter tuning hyperhyperparameters -- fill in with the desired range
    trialnum=20
    dropouts=[0,0.5]
    droppaths=[0,0.5]
    logl2s=[-6,-1]
    loglrs=[-4,-2]
    absolute_best=np.load('newregionbest_loss.npy') #if you've already run hyperparameter tuning and are looking for even better hyperparameters, fill in the corresponding npy file, otherwise initialize this to float('inf')
    for trial in range(trialnum):
        patience=0
        num_epochs = 10 #total epochs 
        
        #Randomly choose hyperparameters
        l2=0 if random.random()<0.2 else 10**uniform(logl2s[0],logl2s[1]) #This sets regularization to 0 20% of the time
        dropout=uniform(dropouts[0],dropouts[1])
        droppath=uniform(droppaths[0],droppaths[1])
        lr=10**uniform(loglrs[0],loglrs[1])
        hyper_params=np.array([l2,dropout,droppath,lr])
        if weighted:
            experiment_name = f'simvp_weighted_l2_'+str(l2)+'_dropout'+str(dropout)+'_droppath_'+str(droppath)

        else:  
            experiment_name = f'simvpl2_'+str(l2)+'_dropout'+str(dropout)+'_droppath_'+str(droppath)
        experiment_name=experiment_name+f'_lr{lr}'
        callback = LossLoggerCallback(experiment_name+'log.csv')
        if mode=='SLA':
            model = SimVP_Model_no_skip(in_shape=(n_t,1,128,128),model_type='gsta',hid_S=8,hid_T=128,drop=hyper_params[1],drop_path=hyper_params[2]).to(device)
        elif mode=='SLA-SST':
            model=SimVP_Model_no_skip_sst(in_shape=(n_t,2,128,128),model_type='gsta',hid_S=8,hid_T=128,drop=hyper_params[1],drop_path=hyper_params[2]).to(device)
        optimizer = optim.Adam(model.parameters(), lr=lr,weight_decay=l2)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=lr, steps_per_epoch=len(train_dataloader), epochs=num_epochs)

        best_val_loss = float('inf')  # Initialize the best validation loss

        use_amp = True

        scaler = torch.cuda.amp.GradScaler(enabled=use_amp) # to allow use of floating point 16 training, slightly faster

        for epoch in range(num_epochs):
            if patience>maxpatience:
                break
            model.train()
            train_loss = 0.0
            num_batches = 0
            for torch_input_batch, torch_output_batch in train_dataloader:
                optimizer.zero_grad(set_to_none=True)
                if mode=='SLA-SST':
                    torch_input_batch = torch_input_batch.squeeze().cuda()
                    torch_output_batch = torch_output_batch.squeeze().cuda()
                elif mode=='SLA':
                    torch_input_batch = torch_input_batch.squeeze(0).cuda()
                    torch_output_batch = torch_output_batch.squeeze(0).cuda()
                with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=use_amp):
                    torch_input_batch=torch_input_batch.half()
                    outputs = model(torch_input_batch)
                    if weighted:
                        loss = torch_tracked_mse_interp_weighted(outputs, torch_output_batch)
                    else:
                        loss = torch_tracked_mse_interp(outputs, torch_output_batch)
                loss=loss.half()
                grads=scaler.scale(loss)
                grads.float().backward()
                scaler.unscale_(optimizer)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()

                train_loss += loss.item()
                num_batches += 1

                # Print progress
                if num_batches%100==1:
                    print(f'Train Epoch: {epoch+1}/{num_epochs} '
                          f'Batch: {num_batches}/{n_train_batches} '
                          f'Loss: {train_loss/num_batches:.4f}')

            train_loss /= num_batches

            # Validation phase
            model.eval()
            val_loss = 0.0
            num_val_batches = 0

            with torch.no_grad():
                for torch_input_batch, torch_output_batch in val_dataloader:

                    if mode=='SLA-SST':
                        torch_input_batch = torch_input_batch.squeeze().cuda()
                        torch_output_batch = torch_output_batch.squeeze().cuda()
                    elif mode=='SLA':
                        torch_input_batch = torch_input_batch.squeeze(0).cuda()
                        torch_output_batch = torch_output_batch.squeeze(0).cuda()
                    with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=use_amp):
                        val_preds = model(torch_input_batch)
                        if weighted:
                            val_loss += torch_tracked_mse_interp_weighted(val_preds, torch_output_batch).item()
                        else:
                            val_loss += torch_tracked_mse_interp(val_preds, torch_output_batch).item()
                    num_val_batches += 1
                    if num_val_batches%10==1:
                        print(f'Validation Loss {num_val_batches}/{n_val_batches}: {val_loss/num_val_batches:.4f}')

            val_loss /= num_val_batches
            callback(epoch + 1, train_loss, val_loss)
            torch.save(model.state_dict(), weight_dir+experiment_name+f'epoch:{epoch}.pt')
            # Check if validation loss improved
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                checkpoint = {"model": model.state_dict(),
                  "optimizer": optimizer.state_dict(),
                  "scaler": scaler.state_dict()}
                torch.save(model.state_dict(), weight_dir+experiment_name+"best.pt")
                patience=0
            else:
                patience+=1
        if best_val_loss<absolute_best:
            absolute_best=best_val_loss
            np.save('newregionbest_hyperparameters.npy',hyper_params) #Change this to wherever you want to save the hyperparameters
            np.save('newregionbest_loss.npy',absolute_best)#Change this to wherever you want to save the best loss