#!/usr/bin/env python
# coding: utf-8

# # Project lat/lon coords onto local orthonormal grid and bin

# In[19]:


import numpy as np
import matplotlib.pyplot as plt
import pyproj
import scipy
from scipy import stats


# In[2]:


# Define the pyproj transformer objects used to transform coordinates between (lat,long,alt) and ECEF in both directions
transformer_ll2xyz = pyproj.Transformer.from_crs(
        {"proj":'latlong', "ellps":'WGS84', "datum":'WGS84'},
        {"proj":'geocent', "ellps":'WGS84', "datum":'WGS84'},
        )
transformer_xyz2ll = pyproj.Transformer.from_crs(
        {"proj":'geocent', "ellps":'WGS84', "datum":'WGS84'},
        {"proj":'latlong', "ellps":'WGS84', "datum":'WGS84'},
        )


# In[3]:


# convert ECEF coords to lat,lon:
def xyz2ll(x,y,z, lat_org, lon_org, alt_org, transformer1, transformer2):

    # transform origin of local tangent plane to ECEF coordinates (https://en.wikipedia.org/wiki/Earth-centered,_Earth-fixed_coordinate_system)
    x_org, y_org, z_org = transformer1.transform( lon_org,lat_org,  alt_org,radians=False)
    ecef_org=np.array([[x_org,y_org,z_org]]).T

    # define 3D rotation required to transform between ECEF and ENU coordinates (https://gssc.esa.int/navipedia/index.php/Transformations_between_ECEF_and_ENU_coordinates)
    rot1 =  scipy.spatial.transform.Rotation.from_euler('x', -(90-lat_org), degrees=True).as_matrix()
    rot3 =  scipy.spatial.transform.Rotation.from_euler('z', -(90+lon_org), degrees=True).as_matrix()
    rotMatrix = rot1.dot(rot3)

    # transform ENU coords to ECEF by rotating
    ecefDelta = rotMatrix.T.dot(np.stack([x,y,np.zeros_like(x)],axis=-1).T)
    # add offset of all corrds on tangent plane to get all points in ECEF
    ecef = ecefDelta+ecef_org
    # transform to geodetic coordinates
    lon, lat, alt = transformer2.transform( ecef[0,:],ecef[1,:],ecef[2,:],radians=False)
    # only return lat, lon since we're interested in points on Earth. 
    # N.B. this amounts to doing an inverse stereographic projection from ENU to lat, lon so shouldn't be used to directly back calculate lat, lon from tangent plane coords
    # this is instead achieved by binning the data's lat/long variables onto the grid in the same way as is done for the variable of interest
    return lat, lon


# convert lat, lon to ECEF coords
def ll2xyz(lat, lon, alt, lat_org, lon_org, alt_org, transformer):

    # transform geodetic coords to ECEF (https://en.wikipedia.org/wiki/Earth-centered,_Earth-fixed_coordinate_system)
    x, y, z = transformer.transform( lon,lat, np.zeros_like(lon),radians=False)
    x_org, y_org, z_org = transformer.transform( lon_org,lat_org,  alt_org,radians=False)
    # define position of all points relative to origin of local tangent plane
    vec=np.array([[ x-x_org, y-y_org, z-z_org]]).T

    # define 3D rotation required to transform between ECEF and ENU coordinates (https://gssc.esa.int/navipedia/index.php/Transformations_between_ECEF_and_ENU_coordinates)
    rot1 =  scipy.spatial.transform.Rotation.from_euler('x', -(90-lat_org), degrees=True).as_matrix()
    rot3 =  scipy.spatial.transform.Rotation.from_euler('z', -(90+lon_org), degrees=True).as_matrix()
    rotMatrix = rot1.dot(rot3)    

    # rotate ECEF coordinates to ENU
    enu = rotMatrix.dot(vec)
    X = enu.T[0,:,0]
    Y = enu.T[0,:,1]
    Z = enu.T[0,:,2]
    return X, Y, Z

