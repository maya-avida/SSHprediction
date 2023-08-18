import numpy as np
#calculates mean and standard deviation of the regions in question

def avgstd(batch_num,datadir,val="training/"):
    shape=np.load(datadir + val +'batch0_invar.npy').shape
    SSHtotal=0
    nonzeroSSHtotal=0
    SSTtotal=0
    nonzeroSSTtotal=0
    #calculates mean of SSH and SST
    for ID in range(batch_num):
        data=np.load(datadir + val +'batch' + str(ID) + "_invar.npy")
        SSH=(data[:,:,:,:,0]).flatten()
        SST=(data[:,:,:,:,1]).flatten()
        SSHtotal+=np.sum(SSH) #running total of SSH
        nonzeroSSHtotal+=np.sum(SSH!=0)#running total on nonzero SSH values
        SSTtotal+=np.sum(SST)
        nonzeroSSTtotal+=np.sum(SST!=0)
    SSHmean=SSHtotal/nonzeroSSHtotal
    SSTmean=SSTtotal/nonzeroSSTtotal
    for ID in range(batch_num):
        data=np.load(datadir + val +'batch' + str(ID) + "_invar.npy")
        SSH=(data[:,:,:,:,0]).flatten()
        SST=(data[:,:,:,:,1]).flatten()
        SSHtotal+=np.sum(((SSH-SSHmean)**2)*(SSH!=0))
        nonzeroSSHtotal+=np.sum(SSH!=0)
        SSTtotal+=np.sum(((SST-SSTmean)**2)*(SST!=0))
        nonzeroSSTtotal+=np.sum(SST!=0)
    SSHstd=np.sqrt(SSHtotal/nonzeroSSHtotal)
    SSTstd=np.sqrt(SSTtotal/nonzeroSSTtotal)
    return SSHmean, SSHstd, SSTmean, SSTstd
datadir = '/home/jovyan/preproSST/'
n_train=1000
gs_mean_ssh,gs_std_ssh,gs_mean_sst,gs_std_sst = avgstd(n_train,datadir)
print(gs_mean_ssh,gs_std_ssh,gs_mean_sst,gs_std_sst)
np.save("gs_mean_ssh_futureSST.npy", gs_mean_ssh)
np.save("gs_std_ssh_futureSST.npy", gs_std_ssh)
np.save("gs_mean_sst_futureSST.npy", gs_mean_sst)
np.save("gs_std_sst_futureSST.npy", gs_std_sst)