import dask
import dask.dataframe as dd
import pandas as pd
import numpy as np
import geopandas as gpd
from shapely.geometry import Point, Polygon, shape, LinearRing
from shapely.ops import nearest_points
from geopy.distance import geodesic
from math import *
import os
import glob
import netCDF4
from netCDF4 import Dataset
from scipy import spatial
import numpy as np
from joblib import Parallel, delayed
import multiprocessing
from datetime import datetime
import xarray as xr
from tqdm import tqdm, tqdm_pandas, tqdm_notebook
import urllib
from datetime import datetime, timedelta
import urllib.request
import geopandas as gpd
from shapely.geometry import Point, Polygon, shape
import shapefile
from math import *
import cartopy.io.shapereader as shpreader
from json import loads
from time import sleep
from urllib.request import Request, urlopen
from tqdm import tqdm
from dask.diagnostics import ProgressBar


LON1 = -68
LON2 = -51
LAT1 = -51
LAT2 = -39



def dist(lat1, lon1, lat2, lon2):
    return np.sqrt( (lat2 - lat1)**2 + (lon2 - lon1)**2)



def haversine(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance between two points 
    on the earth (specified in decimal degrees)
    """
    # convert decimal degrees to radians 
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    # haversine formula 
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a)) 
    r = 6371 # Radius of earth in kilometers. Use 3956 for miles
    return c * r


# Assign EEZ indicator for GFW
def distance_to_eez(lon, lat):
    point = Point(lon, lat)
    d = pol_ext.project(point)
    p = pol_ext.interpolate(d)
    lon2 = list(p.coords)[0][0]
    lat2 = list(p.coords)[0][1]
    #p1, p2 = nearest_points(poly, point)
    #print(p1.wkt)
    #lon2 = p1.x
    #lat2 = p1.y
    return haversine(lon, lat, lon2, lat2)




# Assign EEZ indicator for GFW
def eez_check(lon, lat):
    pnts = gpd.GeoDataFrame(geometry=[Point(lon, lat)])
    check = pnts.assign(**{key: pnts.within(geom) for key, geom in polys.items()})
    return check.Argentina_EEZ.values[0]


# Get seascape lat/lon
def find_seascape(date, lat, lon):
    # lat = -51
    # lon = -68
    lat1 = lat - .15
    lat2 = lat + .15
    lon1 = lon - .15
    lon2 = lon + .15
    
    indat = sea[sea['date'] == date]
    indat = indat[(indat['lon'].values >= lon1) & 
                  (indat['lon'].values <= lon2) & 
                  (indat['lat'].values >= lat1) & 
                  (indat['lat'].values <= lat2)] 
    
    distances = indat.apply(lambda row: dist(lat, lon, row['lat'], row['lon']), axis=1)
    #rdat = pd.DataFrame({"seascape_class": [indat.loc[distances.idxmin(), 'seascape_class']], "seascape_prob": [indat.loc[distances.idxmin(), 'seascape_prob']]})
    #print(rdat)
    #return rdat
    return (indat.loc[distances.idxmin(), 'seascape_class'], indat.loc[distances.idxmin(), 'seascape_prob'])


# ---------------------------------------------------------------------
# Main function to process data
def process_days(dat):    
    
    check = 0
    
    # Date to save file as
    date = dat['date'].iat[0]
    
    try:        
        # ------------------------------------
        #print("1-Processing Seascapes")
        dat['seascape_class'], dat['seascape_prob'] = zip(*dat.apply(lambda row: find_seascape(row['date'], row['lat'], row['lon']), axis=1))
        check += 1        
    except:
        print(f"Failed: Seascape - {date}")    
        
    try:
        # ------------------------------------
        # print("6-Processing distance to EEZ line")
        dat['distance_to_eez_km'] = dat.apply((lambda x: distance_to_eez(x['lon'], x['lat'])), axis=1)
        check += 1
    except:
        print(f"Failed: Distance to EEZ {date}")

   
    try:
        # ------------------------------------
        # print("8-Assigning EEZ True or False")
        dat['eez'] = dat.apply((lambda x: eez_check(x['lon'], x['lat'])), axis=1)
        check += 1
    except:
        print(f"Failed: EEZ Check {date}")
        
    if check == 3:
        # ------------------------------------
        print(f"4-Save data to data/processed/10d/illegal_seascape/processed_{date}.feather")
        
        # Save data
        #print(dat.columns)
        outdat = dat.reset_index(drop=True)
        outdat.to_feather(f"data/processed/10d/illegal_seascape/processed_{date}.feather")
    else:
        print(f"Failed: {date}")
        
        
        
# Calculate distance to closest eez   
shpfile1 = gpd.read_file("data/EEZ/eez_v10.shp")
arg = shpfile1[shpfile1.Territory1 == 'Argentina'].reset_index(drop=True)  #268
polys = gpd.GeoSeries({'Argentina_EEZ': arg.geometry})
poly = arg.geometry[0]
pol_ext = LinearRing(poly.exterior.coords)


# Dats to filter
days = ['2016-02-26', '2016-03-05', '2016-02-18', '2016-02-02', '2016-03-31', '2016-01-25']

# SST
sst = pd.read_feather('data/patagonia_shelf_SST_8DAY_2012-2016.feather')
sst = sst[sst['date'].isin(days)]

# Seascapes
sea = pd.read_feather('data/patagonia_shelf_seascapes_8DAY_2012-2016.feather')
sea = sea[sea['date'].isin(days)]


# ------------------------------------
# Standard Parallel processing
gb = sst.groupby('date')
days = [gb.get_group(x) for x in gb.groups]

pool = multiprocessing.Pool(6)
pool.map(process_days, days)
pool.close()

# dat = days[0]
# dat = dat.iloc[1:50, :]
# process_days(dat)


# Combine processed files
files = glob.glob('data/processed/10d/illegal_seascape/*.feather')
files
list_ = []
for file in files:
    df = pd.read_feather(file)
    list_.append(df)
    mdat = pd.concat(list_, sort=False)

# # Manually remove veseels near land in eez
# #mdat.loc[:, 'eez'] = np.where(np.logical_and(mdat['lat1'] > -48, mdat['lon1'] < -65), True, mdat['eez'])
# #mdat.loc[:, 'eez'] = np.where(np.logical_and(mdat['lat1'] > -44, mdat['lon1'] < -63), True, mdat['eez'])
# #mdat.loc[:, 'eez'] = np.where(np.logical_and(mdat['lat1'] > -40, mdat['lon1'] < -61), True, mdat['eez'])

# Convert to data frame and save
mdat = mdat.reset_index(drop=True)
mdat.to_feather('data/illegal_seascape_data_model.feather')

