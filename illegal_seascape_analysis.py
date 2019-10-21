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
import calendarimport pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
import glob
from sklearn.preprocessing import MinMaxScaler
from statsmodels.discrete.discrete_model import Logit
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_validate
from sklearn.metrics import average_precision_score, accuracy_score, roc_curve, auc, precision_recall_curve, f1_score
from sklearn.ensemble import RandomForestClassifier
import sklearn.metrics as metrics
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit, GridSearchCV
from collections import deque
import calendar


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
days

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






# -----------------------------------------------------------------
# Random Forest Classification

# 8Day data
dat = pd.read_feather('data/full_gfw_10d_effort_model_data_8DAY_2012-01-01_2016-12-26.feather')

# Keep only Chinese vessels
dat = dat[(dat['flag'] == 'CHN') | (dat['flag'] == 'ARG')]

# If illegally operating inside EEZ (!= ARG)
dat.loc[:, 'illegal'] = np.where(((dat['eez'] == True) & (dat['fishing_hours'] > 0) & (dat['flag'] != 'ARG') ), 1, 0)

# Buffer by 2km
dat.loc[:, 'illegal_2km'] = np.where(((dat['illegal'] == True) & (dat['distance_to_eez_km'] >= 2)), 1, 0)

# Convert true/false eez to 0/1
dat.loc[:, 'illegal'] = dat.illegal.astype('uint8')
dat.loc[:, 'illegal_2km'] = dat.illegal_2km.astype('uint8')

sum(dat.illegal)/len(dat)
sum(dat.illegal_2km)/len(dat)

# Get year month
dat.loc[:, 'year'] = pd.DatetimeIndex(dat['date']).year
dat.loc[:, 'month'] = pd.DatetimeIndex(dat['date']).month

# Convert month number to name
dat.loc[:, 'month_abbr'] = dat.apply(lambda x: calendar.month_abbr[x['month']], 1)

# moddat = dat[['month_abbr', 'seascape_class', 'sst', 'eez', 'distance_to_eez_km', 
#               'lat1', 'lon1']].dropna().reset_index(drop=True)

moddat = dat[['illegal', 'sst', 'eez', 'distance_to_eez_km', 'seascape_class', 'month_abbr',
              'lat1', 'lon1']].dropna().reset_index(drop=True)

# Dummy variables for seascape and dummies
seascape_dummies = pd.get_dummies(moddat['seascape_class'], prefix='seascape').reset_index(drop=True)
month_dummies = pd.get_dummies(moddat['month_abbr']).reset_index(drop=True)

# Concat dummy variables
moddat = pd.concat([moddat, seascape_dummies, month_dummies], axis=1)

# Add seascapes available in model test data that is missing from moddat
moddat['seascape_1.0'] = 0
moddat['seascape_3.0'] = 0
moddat['seascape_5.0'] = 0
moddat['seascape_8.0'] = 0
moddat['seascape_11.0'] = 0
moddat['seascape_13.0'] = 0
moddat['seascape_20.0'] = 0
moddat['seascape_23.0'] = 0

dat.groupby('date').count()

# Get X, y
y = moddat[['illegal']].reset_index(drop=True)
y = y.ravel()

# Drop dummy variables and prediction
# moddat = moddat.drop(columns = ['month_abbr', 'illegal', 'seascape_class'])

moddat = moddat.drop(columns = ['illegal', 'month_abbr', 'seascape_class'])

# Build data for model
X = moddat
X.columns
X.head()
y.head()

# Classication object
clf = RandomForestClassifier(n_estimators = 100).fit(X, y)




# ------------------------------------------------------------------
# Test data set 
il_test = pd.read_feather('data/illegal_seascape_data_model.feather')

# Convert month
il_test['month'] = pd.DatetimeIndex(il_test['date']).month

# Convert month number to name
il_test.loc[:, 'month_abbr'] = il_test.apply(lambda x: calendar.month_abbr[x['month']], 1)

# Dummy variables for seascape and dummies
il_seascape_dummies = pd.get_dummies(il_test['seascape_class'], prefix='seascape').reset_index(drop=True)
il_month_dummies = pd.get_dummies(il_test['month_abbr']).reset_index(drop=True)

# Concat dummy variables
il_test = pd.concat([il_test, il_seascape_dummies, il_month_dummies], axis=1)

# Drop variables
il_test = il_test.drop(columns = ['month_abbr', 'month', 'seascape_class'])

il_test = il_test[['date', 'lon', 'lat', 'sst', 'seascape_prob', 'distance_to_eez_km', 'eez',
       'seascape_1.0', 'seascape_2.0', 'seascape_3.0', 'seascape_5.0',
       'seascape_7.0', 'seascape_8.0', 'seascape_11.0', 'seascape_12.0',
       'seascape_13.0', 'seascape_14.0', 'seascape_15.0', 'seascape_17.0',
       'seascape_20.0', 'seascape_21.0', 'seascape_23.0', 'seascape_25.0',
       'seascape_27.0', 'Feb', 'Jan', 'Mar']]

# Rename to match model data labels
il_test.columns = ['date', 'lon1', 'lat1', 'sst', 'seascape_prob', 'distance_to_eez_km', 'eez',
       'seascape_1.0', 'seascape_2.0', 'seascape_3.0', 'seascape_5.0',
       'seascape_7.0', 'seascape_8.0', 'seascape_11.0', 'seascape_12.0',
       'seascape_13.0', 'seascape_14.0', 'seascape_15.0', 'seascape_17.0',
       'seascape_20.0', 'seascape_21.0', 'seascape_23.0', 'seascape_25.0',
       'seascape_27.0', 'Feb', 'Jan', 'Mar']

# Create Month columns for missing data
il_test['Apr'] = 0
il_test['May'] = 0
il_test['Jun'] = 0
il_test['Jul'] = 0
il_test['Aug'] = 0
il_test['Sep'] = 0
il_test['Oct'] = 0
il_test['Nov'] = 0
il_test['Dec'] = 0

#il_test = il_test[['sst', 'eez', 'distance_to_eez_km', 'lat', 'lon']]

#il_test.columns = ['sst', 'eez', 'distance_to_eez_km', 'lat1', 'lon1']

# Drop NA for model
il_test = il_test.dropna()

# Get dates for output df
datet = il_test['date'].copy()

# Drop date
il_test = il_test.drop(columns = 'date')

# Test predictions
y_pred = clf.predict(il_test)
sum(y_pred)

# Get predict probabilities
y_proba = clf.predict_proba(il_test)

pred_0 = [el[0] for el in y_proba]
pred_1 = [el[1] for el in y_proba]

il_test['pred_0'] = pred_0
il_test['pred_1'] = pred_1
il_test['y_pred'] = y_pred
il_test['date'] = datet


# Feature importance
fea_import = pd.DataFrame({'variable': X.columns, 'importance': clf.feature_importances_})
fea_import = fea_import.sort_values('importance', ascending=False)
print(fea_import)


# save data
il_test = il_test.reset_index(drop=True)
il_test.to_feather('data/illegal_seascape_pred.feather')