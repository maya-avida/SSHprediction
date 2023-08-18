#This takes the AVISO data and breaks it into daily data; it is step 1 in the preprocessing for the eddy observations
import xarray as xr
from datetime import timedelta, date, datetime
import numpy as np
filepath="/home/mavida/dat1/miniconda3/eddytracking/scottcopy/scottcopy.nc" #path to netcdf file with SSH
savepath="rawdata/"
n_days=30
dataset=xr.open_dataset(filepath)
print(dataset)
for t in range(n_days):
    name=savepath+str(t)+".nc"
    dailydata=dataset.isel(time=t)
    dailydata.to_netcdf(path=name, mode='w')
