import xarray as xr
import numpy as np
import scipy.stats
import datetime
import os
def generatefilename(day,month,year,hour,satellite):
    SSTpath="SSTdataraw/"
    if satellite=="GOES-13":
        date=year+month+day
        filenamemiddle="0000-OSISAF-L3C_GHRSST-SSTsubskin-GOES13-ssteqc_goes13_"
        filenameend="0000-v02.0-fv01.0.nc"
        filename=SSTpath+date+hour+filenamemiddle+date+"_"+hour+filenameend
    elif satellite=="GOES-16":
        fileend="0000-STAR-L2P_GHRSST-SSTsubskin-ABI_G16-ACSPO_V2.70-v02.0-fv01.0.nc"
        filename=SSTpath+year+month+day+hour+fileend
    return filename
def gridSSTdata(coords,allowed,filename,regionoptions,n=128):
    #this function takes in one SST file for a single hour of a single hour and returns a list of the binned SSTs corresponding to each region
    try:
        outputgrids=[]
        hourlyda=xr.open_dataset(filename)
        if allowed==None:
            allowedval=hourlyda['sea_surface_temperature'].values.flatten()
            valnotnan=(np.isnan(allowedval)==False)&(hourlyda['quality_level'].values.flatten()==5)#picks the values where val is not nan and where the quality level is 5

        else:
            allowedval=hourlyda['sea_surface_temperature'].values.flatten()[allowed] #This matches the SST values to the shape of the x and y values
            valnotnan=(np.isnan(allowedval)==False)&(hourlyda['quality_level'].values.flatten()[allowed]==5)#picks the values where val is not nan and where the quality level is 5
        val=allowedval[valnotnan]
        for i,r in enumerate(regionoptions):
            coord=coords[i]
            x=coord[0,:][valnotnan] #mathes the x values to the SST values
            y=coord[1,:][valnotnan]
            # bin the data onto a regular 7.5km grid in the x-y plane (the same grid we're reconstructing SSH on)
            n = 128 # number grid points
            L_x = 960e3 # size of box in m
            L_y = 960e3
            sst_grid, _,_,_ = scipy.stats.binned_statistic_2d(x, y, val, statistic = 'mean', bins=n, range = [[-L_x/2, L_x/2],[-L_y/2, L_y/2]]) #Bin the much denser SST observations onto the coarser grid used in the ML model
            sst_grid = np.rot90(sst_grid) #This makes the SST file show up correctly on plt.imshow
            outputgrids.append(np.flip(sst_grid,axis=0)) #When I plotted it, it only matched up with the gridded result when I flipped it 
        os.remove(filename) #Remove the file so it stops taking up space
        return outputgrids #output shape: region, x, y
    except OSError as err:
        #Just in case PODAAC failed to download the file
        print("OS error:", err)
        return np.nan*np.zeros((len(regionoptions),n,n))
def prepro(start_date,regionoptions,allowed,coordpath,satellite):
    #this function actually saves the files for a specific date, starting at hour 20 on start_date and finishing on hour 19 on start_date+1
    #This chunk of code just makes the filename
    n=128
    savedir='SSTpreprocessed/'
    t=start_date
    day=str(t.day).zfill(2)
    month=str(t.month).zfill(2)
    year=str(t.year).zfill(4)
    #This is where we load the x and y coordinates that we preprocessed with 'calculatexy.py'
    coords=[]
    if satellite=="GOES-16":
        for r in regionoptions:
            coords.append(np.load(coordpath+f'xy{r}.npy'))
    else:
        for r in regionoptions:
            coords.append(np.load(coordpath+f'GOES-13xy{r}.npy'))
    #Because temperatures vary between night and day, we separate the night SST and the day SST between two different files
    nightda=[]
    dayda=[]
    #Hour 20 on the day before until midnight — nightly data
    for hour_ in range(4):
        hour=str(hour_).zfill(2)
        filename=generatefilename(day,month,year,hour,satellite)
        nightda.append(gridSSTdata(coords,allowed,filename,regionoptions))
    t=t+datetime.timedelta(days = 1)
    day=str(t.day).zfill(2)
    month=str(t.month).zfill(2)
    year=str(t.year).zfill(4)
    #Midnight until 6 am — nightly data
    for hour_ in range(7):
        hour=str(hour_).zfill(2)
        filename=generatefilename(day,month,year,hour,satellite)
        nightda.append(gridSSTdata(coords,allowed,filename,regionoptions))
    #7 am until 7 pm — daily data
    for hour_ in range(7,20):
        hour=str(hour_).zfill(2)
        filename=generatefilename(day,month,year,hour,satellite)
        dayda.append(gridSSTdata(coords,allowed,filename,regionoptions))
    
    #Stack and average the nightly data and daily data
    dayda=np.stack(dayda) #output shape: hour, region, x, y
    nightda=np.stack(nightda) #output shape: hour, region, x, y
    daydamean=np.nanmean(dayda,axis=0) #output shape: region, x, y
    nightdamean=np.nanmean(nightda,axis=0) #output shape: region, x, y
    
    for count,r in enumerate(regionoptions):
        regionalday=np.squeeze(daydamean[count,:,:]) #shape: x,y
        regionalnight=np.squeeze(nightdamean[count,:,:]) #shape:x,y
        np.save(savedir+f'{r}/day{t}.npy',regionalday) #save the daily data
        np.save(savedir+f'{r}/night{t}.npy',regionalnight) #save the nightly data
