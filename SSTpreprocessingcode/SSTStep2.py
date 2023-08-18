#8/10/23: this script converts the latitude and longitude of the SST observations to the x and y used by the SST prepreprocessing code
import numpy as np
from projectcoords import ll2xyz
import xarray as xr
import pyproj

# Define the pyproj transformer objects used to transform coordinates between (lat,long,alt) and ECEF in both directions
transformer_ll2xyz = pyproj.Transformer.from_crs(
        {"proj":'latlong', "ellps":'WGS84', "datum":'WGS84'},
        {"proj":'geocent', "ellps":'WGS84', "datum":'WGS84'},
        )
transformer_xyz2ll = pyproj.Transformer.from_crs(
        {"proj":'geocent', "ellps":'WGS84', "datum":'WGS84'},
        {"proj":'latlong', "ellps":'WGS84', "datum":'WGS84'},
        )
#These are the numbered regions that we will be using 
region="Test"
if region=="Training":
    regionoptions=np.load("../newregions.npy")
elif region=="Test":
    regionoptions=["test"]
satellite="GOES-13" #GOES-16 or GOES 13
if satellite=="GOES-16":
    #These are the latitudes and longitudes for the GOES-16 satellite; this file is included in the google drive
    latlonpath='latlon/G16_075_0_W.nc'
    lat=xr.open_dataset(latlonpath)['lat'].values
    lon=xr.open_dataset(latlonpath)['lon'].values

    n=128 #The number of coordinates

    allowed=np.isnan(lat.flatten()+lon.flatten())==False #These are the locations where latitude and longitude aren't nan
    allowedlat=lat.flatten()[allowed] #The flattened array of nonnan latitudes
    allowedlon=lon.flatten()[allowed] #The flattened array of nonnan longitudes
    for r in regionoptions:
        print(r)
        try: #This just makes sure we haven't already made the .npy file with the x and y coordinates for this region 
            np.load(f'xy{r}.npy')
            print("Already made")
        except:
            #This chunk of code actually makes the f'xy{r}.npy' file for each region, which we will use in the preprocessing code to get the SST on the same grid as the SSH observations
            if region=="Training":
                coordpath = f'/dat1/smart1n/aviso-data/global training data/raw/{r}/'
            else:
                coordpath=f'/dat1/smart1n/aviso-data/data challenge region data/processed/38.0_-60.0/'
            coords_grid=np.load(coordpath+"coords.npy")
            lon0 = 0.25*(coords_grid[63,63,0]+coords_grid[64,63,0]+coords_grid[63,64,0]+coords_grid[64,64,0])
            lat0 = 0.25*(coords_grid[63,63,1]+coords_grid[64,63,1]+coords_grid[63,64,1]+coords_grid[64,64,1])
            #convert latitude and longitude to an x and y grid
            x,y,_ = ll2xyz(allowedlat, allowedlon, alt = 0, lat_org = lat0, lon_org = lon0, alt_org = 0, transformer = transformer_ll2xyz)
            np.save(f'xy{r}.npy',np.stack([x,y]))
elif satellite=="GOES-13":
    #This is the first day of 2017 for the GOES-13 satellite; this file is also included in the google drive
    latlonpath='latlon/20170101200000-OSISAF-L3C_GHRSST-SSTsubskin-GOES13-ssteqc_goes13_20170101_200000-v02.0-fv01.0.nc'
    latlonds=xr.open_dataset(latlonpath)
    lat=latlonds['lat'].values
    lon=latlonds['lon'].values
    t,lat,lon=np.meshgrid(np.arange(1),lat,lon,indexing='ij')

    n=128 #The number of coordinates

    allowed=np.isnan(lat.flatten()+lon.flatten())==False #These are the locations where latitude and longitude aren't nan
    allowedlat=lat.flatten()[allowed] #The flattened array of nonnan latitudes
    allowedlon=lon.flatten()[allowed] #The flattened array of nonnan longitudes
    for r in regionoptions:
        print(r)
        try: #This just makes sure we haven't already made the .npy file with the x and y coordinates for this region 
            np.load(f'GOES-13xy{r}.npy')
            print("Already made")
        except:
            #This chunk of code actually makes the f'xy{r}.npy' file for each region, which we will use in the preprocessing code to get the SST on the same grid as the SSH observations
            if region=="Training":
                coordpath = f'/dat1/smart1n/aviso-data/global training data/raw/{r}/'
            else:
                coordpath=f'/dat1/smart1n/aviso-data/data challenge region data/processed/38.0_-60.0/'
            coords_grid=np.load(coordpath+"coords.npy")
            lon0 = 0.25*(coords_grid[63,63,0]+coords_grid[64,63,0]+coords_grid[63,64,0]+coords_grid[64,64,0])
            lat0 = 0.25*(coords_grid[63,63,1]+coords_grid[64,63,1]+coords_grid[63,64,1]+coords_grid[64,64,1])
            #convert latitude and longitude to an x and y grid
            x,y,_ = ll2xyz(allowedlat, allowedlon, alt = 0, lat_org = lat0, lon_org = lon0, alt_org = 0, transformer = transformer_ll2xyz)
            np.save(f'GOES-13xy{r}.npy',np.stack([x,y]))