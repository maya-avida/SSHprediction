#This file downloads the SST files for each day and then runs the code that converts the observations into a form that the futureprediction.py code recognizes
import os
from preprocessSST import prepro #This is where the actual preprocessing happens
import datetime
import numpy as np
import xarray as xr

#These are the numbered regions that we will be using 
regionoptions=np.load(".NPY FILE WITH THE REGIONS") #INSERT PATH HERE

#This part of the code just makes parallelization starting from 2017 easier; the first time you run the code in a terminal, it will process 12/15/2017 through 12/31/2018, the second time, 1/1/2019 through 12/31/2019, and each time afterwards adding a new year. If you want to run the code consecutively, just initialize a start_date and an end_date and set t=start_date
try:
    index=np.load('yearindex.npy')
    start_date=datetime.date(2018+index,1,1)
    end_date=min(datetime.date(2022,12,31),start_date+datetime.timedelta(days=366))
    np.save('yearindex.npy',index+1)
except:
    start_date=datetime.date(2017,12,15)
    end_date=datetime.date(2018,12,31)
    np.save('yearindex.npy',1)
t=start_date
print(t)
#These are the latitudes and longitudes for the GOES-16 satellite; this file is included in the google drive
latlonpath='latlon/G16_075_0_W.nc'
lat=xr.open_dataset(latlonpath)['lat'].values
lon=xr.open_dataset(latlonpath)['lon'].values
n=128 #the size of the grid
allowed=np.isnan(lat.flatten()+lon.flatten())==False #these are the locations where lat and lon aren't nan
#Just put whatever path you used in calculatexy.py
coordpath="PATH"#INSERT PATH HERE
satellite="GOES-1
while t<end_date:
    
    os.system(f'podaac-data-downloader -c ABI_G16-STAR-L2P-v2.70 -d ./SSTdataraw/ -sd {t}T20:00:00Z -ed {t+datetime.timedelta(1)}T20:00:00Z') #This downloads the raw data from PODAAC; for the code to work you need to make a folder inside whatever folder you're running this called 'SSTdataraw' and a folder called 'SSTpreprocessed'
    prepro(t,regionoptions,allowed,coordpath) #This calls the function that actually preprocesses the SST observations
    t+=datetime.timedelta(1)
    print(t)


   