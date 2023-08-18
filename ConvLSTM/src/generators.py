import tensorflow as tf
from tensorflow import keras
import numpy as np
import copy
#No major changes here, I just changed it so it can handle the already batched data and if oneoutput=True, it only returns the first day of output data
class DataGenerator_ssh_sst_interp(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, list_IDs, batch_size=32, dim=(6,128,128), n_channels=1, shuffle=True, val = 'train', datadir = '30 days 1920km 128x128/training_processed/', stats = (0,1,0,1)):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.shuffle = shuffle
        self.val = val
        self.datadir = datadir
        self.stats = stats
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return len(self.list_IDs)

    def __getitem__(self, index):
        'Generate one batch of data'

        # Find list of IDs
        list_ID = self.list_IDs[index]

        # Generate data
        X, Y = self.__data_generation(list_ID)

        return X, Y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        ID=list_IDs_temp
        mean_ssh, std_ssh, mean_sst, std_sst = self.stats
        
        # Initialization
        
        X1 = np.empty((self.batch_size, self.dim[0], 128, 128, 1))
        X2 = np.empty((self.batch_size, self.dim[0], 128, 128, 1))

        # X2 = np.empty((self.batch_size, self.dim[0], 128, 128, 1))
        # X3 = np.empty((self.batch_size, 30, 128, 128, 1))
        # Y = np.empty((self.batch_size, *self.dim, 1))
        Y_length = []
        Y_list = []

        # Generate data
        # Store sample
        #INPUT DIMENSIONS: sample,day,x,y,0
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
        
        X = [X1,X2]
        return X, Y
    
class DataGenerator_ssh_interp(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, list_IDs, batch_size=32, dim=(6,128,128), n_channels=1, shuffle=True, val = 'train', datadir = '30 days 1920km 128x128/training_processed/', stats = (0,1,0,1),oneoutput=False):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.shuffle = shuffle
        self.val = val
        self.datadir = datadir
        self.stats = stats
        self.on_epoch_end()
        self.oneoutput=oneoutput

    def __len__(self):
        'Denotes the number of batches per epoch'
        return len(self.list_IDs)

    def __getitem__(self, index):
        'Generate one batch of data'

        # Find list of IDs
        list_ID = self.list_IDs[index]

        # Generate data
        X, Y = self.__data_generation(list_ID)

        return X, Y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        ID=list_IDs_temp
        mean_ssh, std_ssh, mean_sst, std_sst = self.stats
        
        # Initialization
        
        X1 = np.empty((self.batch_size, self.dim[0], 128, 128, 1))
        X2 = np.empty((self.batch_size, self.dim[0], 128, 128, 1))

        # X2 = np.empty((self.batch_size, self.dim[0], 128, 128, 1))
        # X3 = np.empty((self.batch_size, 30, 128, 128, 1))
        # Y = np.empty((self.batch_size, *self.dim, 1))
        Y_length = []
        Y_list = []

        # Generate data
        # Store sample
        #INPUT DIMENSIONS: sample,day,x,y,0
        input_ = np.load(self.datadir + self.val +'batch' + ID + "_invar.npy")
        ssh = copy.deepcopy(input_[:,:,:,:,0])
        ssh[np.isnan(ssh)] = 0
        ssh[ssh!=0] = (ssh[ssh!=0]-mean_ssh)/std_ssh

        
        X1[:,:,:,:,0] = ssh[:,:self.dim[0],:,:]
        
        ssh_out = np.load(self.datadir + self.val +'batch' + ID + "_outvar.npy")
        if self.oneoutput:
            ssh_out = ssh_out[:self.dim[0],]

            ssh_out[np.isnan(ssh_out)] = 0
            x = copy.deepcopy(ssh_out[:,0,:,0])
            x[x!=0] = ((x[x!=0]+0.5*960e3)/960e3)*(128-1)
            y = copy.deepcopy(ssh_out[:,0,:,1])
            #OUTPUT DIMENSIONS: sample, day, maximum number of satellites, x y and SSH

            y[y!=0] = ((-y[y!=0]+0.5*960e3)/960e3)*(128-1)
            sla = copy.deepcopy(ssh_out[:,0,:,2])
            sla[sla!=0] = (sla[sla!=0]-mean_ssh)/std_ssh
            outvar = np.stack((x,y,sla),axis = -1)
            Y=outvar

            X = X1
            return X, Y
        else:
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
            return X, Y