#import ray
import pandas as pd
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

LON1 = -68
LON2 = -51
LAT1 = -51
LAT2 = -39


# Uncomment to reprocess GFW effort and Seascape data
#--------------------------------------------------------------------------------

# OC_8D = ['2012-01-01', '2012-01-09', '2012-01-17', '2012-01-25', '2012-02-02',
#  '2012-02-10', '2012-02-18', '2012-02-26', '2012-03-05', '2012-03-13',
#  '2012-03-21', '2012-03-29', '2012-04-06', '2012-04-14', '2012-04-22',
#  '2012-04-30', '2012-05-08', '2012-05-16', '2012-05-24', '2012-06-01',
#  '2012-06-09', '2012-06-17', '2012-06-25', '2012-07-03', '2012-07-11',
#  '2012-07-19', '2012-07-27', '2012-08-04', '2012-08-12', '2012-08-20',
#  '2012-08-28', '2012-09-05', '2012-09-13', '2012-09-21', '2012-09-29',
#  '2012-10-07', '2012-10-15', '2012-10-23', '2012-10-31', '2012-11-08',
#  '2012-11-16', '2012-11-24', '2012-12-02', '2012-12-10', '2012-12-18',
#  '2012-12-26', '2013-01-01', '2013-01-09', '2013-01-17', '2013-01-25',
#  '2013-02-02', '2013-02-10', '2013-02-18', '2013-02-26', '2013-03-06',
#  '2013-03-14', '2013-03-22', '2013-03-30', '2013-04-07', '2013-04-15',
#  '2013-04-23', '2013-05-01', '2013-05-09', '2013-05-17', '2013-05-25',
#  '2013-06-02', '2013-06-10', '2013-06-18', '2013-06-26', '2013-07-04',
#  '2013-07-12', '2013-07-20', '2013-07-28', '2013-08-05', '2013-08-13',
#  '2013-08-21', '2013-08-29', '2013-09-06', '2013-09-14', '2013-09-22',
#  '2013-09-30', '2013-10-08', '2013-10-16', '2013-10-24', '2013-11-01',
#  '2013-11-09', '2013-11-17', '2013-11-25', '2013-12-03', '2013-12-11',
#  '2013-12-19', '2013-12-27', '2014-01-01', '2014-01-09', '2014-01-17',
#  '2014-01-25', '2014-02-02', '2014-02-10', '2014-02-18', '2014-02-26',
#  '2014-03-06', '2014-03-14', '2014-03-22', '2014-03-30', '2014-04-07',
#  '2014-04-15', '2014-04-23', '2014-05-01', '2014-05-09', '2014-05-17',
#  '2014-05-25', '2014-06-02', '2014-06-10', '2014-06-18', '2014-06-26',
#  '2014-07-04', '2014-07-12', '2014-07-20', '2014-07-28', '2014-08-05',
#  '2014-08-13', '2014-08-21', '2014-08-29', '2014-09-06', '2014-09-14',
#  '2014-09-22', '2014-09-30', '2014-10-08', '2014-10-16', '2014-10-24',
#  '2014-11-01', '2014-11-09', '2014-11-17', '2014-11-25', '2014-12-03',
#  '2014-12-11', '2014-12-19', '2014-12-27', '2015-01-01', '2015-01-09',
#  '2015-01-17', '2015-01-25', '2015-02-02', '2015-02-10', '2015-02-18',
#  '2015-02-26', '2015-03-06', '2015-03-14', '2015-03-22', '2015-03-30',
#  '2015-04-07', '2015-04-15', '2015-04-23', '2015-05-01', '2015-05-09',
#  '2015-05-17', '2015-05-25', '2015-06-02', '2015-06-10', '2015-06-18',
#  '2015-06-26', '2015-07-04', '2015-07-12', '2015-07-20', '2015-07-28',
#  '2015-08-05', '2015-08-13', '2015-08-21', '2015-08-29', '2015-09-06',
#  '2015-09-14', '2015-09-22', '2015-09-30', '2015-10-08', '2015-10-16',
#  '2015-10-24', '2015-11-01', '2015-11-09', '2015-11-17', '2015-11-25',
#  '2015-12-03', '2015-12-11', '2015-12-19', '2015-12-27', '2016-01-01',
#  '2016-01-09', '2016-01-17', '2016-01-25', '2016-02-02', '2016-02-10',
#  '2016-02-18', '2016-02-26', '2016-03-05', '2016-03-13', '2016-03-21',
#  '2016-03-29', '2016-04-06', '2016-04-14', '2016-04-22', '2016-04-30',
#  '2016-05-08', '2016-05-16', '2016-05-24', '2016-06-01', '2016-06-09',
#  '2016-06-17', '2016-06-25', '2016-07-03', '2016-07-11', '2016-07-19',
#  '2016-07-27', '2016-08-04', '2016-08-12', '2016-08-20', '2016-08-28',
#  '2016-09-05', '2016-09-13', '2016-09-21', '2016-09-29', '2016-10-07',
#  '2016-10-15', '2016-10-23', '2016-10-31', '2016-11-08', '2016-11-16',
#  '2016-11-24', '2016-12-02', '2016-12-10', '2016-12-18', '2016-12-26']

# # ----------------------------------------------------------------------
# # Parse: GFW Effort Data -----------------------------------------------
# # Get global fish watch data
# GFW_DIR = '/data2/GFW_public/fishing_effort_10d/daily_csvs'

# # Load fishing vessel list
# vessels = pd.read_csv('/data2/GFW_public/fishing_vessels/fishing_vessels.csv')
# vessels.head()

# list_ = []
# outdat = pd.DataFrame()
# # # Append files in subdir
# for i in range(len(OC_8D)):
#     file_ = f"{GFW_DIR}/{OC_8D[i]}.csv"
#     #print(file_)
#     df = pd.read_csv(file_, index_col=None, header=0, low_memory=False)
    
#     # 10d
#     df['lat1'] = df['lat_bin']/10
#     df['lon1'] = df['lon_bin']/10
#     df['lat2'] = df['lat1'] + 0.10
#     df['lon2'] = df['lon1'] + 0.10
    
#     dat = df[(df['lon1'] >= LON1) & (df['lon2'] <= LON2) & (df['lat1'] >= LAT1) & (df['lat2'] <= LAT2)] 
#     outdat = pd.concat([outdat, dat])
#     print(OC_8D[i])

# outdat.head()

# # For 10d, attach vessel data
# outdat = outdat.merge(vessels, on='mmsi')

# # Assign EEZ indicator for GFW
# def eez_check(lon, lat):
#     pnts = gpd.GeoDataFrame(geometry=[Point(lon, lat)])
#     check = pnts.assign(**{key: pnts.within(geom) for key, geom in polys.items()})
#     return check.Argentina_EEZ.values[0]


# shpfile1 = gpd.read_file("data/EEZ/eez_v10.shp")
# arg = shpfile1[shpfile1.Territory1 == 'Argentina'].reset_index(drop=True)  #268

# #lat = -45
# #lon = -63

# # Get polygons to check
# polys = gpd.GeoSeries({'Argentina_EEZ': arg.geometry})

# # Get lat/lon for vessels
# outdat.loc[:, 'eez'] = outdat.apply(lambda x: eez_check(x['lon1'], x['lat1']), axis=1)

# outdat.head()

# outdat = outdat.reset_index(drop=True)
# outdat.to_feather("data/patagonia_shelf_gfw_effort_10d_data.feather")


# --------------------------------------------------------------------
# Parse: Seascape Data -----------------------------------------------

# https://cwcgom.aoml.noaa.gov/thredds/ncss/SEASCAPE_8DAY/SEASCAPES.nc?var=CLASS&var=P&north=-39&west=-68&east=-51&south=-51&disableProjSubset=on&horizStride=1&time_start=2012-01-01T12%3A00%3A00Z&time_end=2016-12-31T12%3A00%3A00Z&timeStride=1&addLatLon=true&accept=netcdf

# url = f"https://cwcgom.aoml.noaa.gov/thredds/ncss/SEASCAPE_8DAY/SEASCAPES.nc?var=CLASS&var=P&north=-39&west=-68&east=-51&south=-51&disableProjSubset=on&horizStride=1&time_start=2012-01-01T12%3A00%3A00Z&time_end=2016-12-31T12%3A00%3A00Z&timeStride=1&addLatLon=true&accept=netcdf"    

# # Download classes
# urllib.request.urlretrieve(url, filename = f"data/seascapes/seascapes_8D_CLASS_PROB_2012-2016.nc")    

# file = "data/seascapes/seascapes_8D_CLASS_PROB_2012-2016.nc"

# ds = xr.open_dataset(file)
# df = ds.to_dataframe()
# df = df.reset_index()
# df = df[(df['lon'] >= LON1) & (df['lon'] <= LON2)]
# df = df[(df['lat'] >= LAT1) & (df['lat'] <= LAT2)]
# df = df.reset_index(drop=True)

# # Issues with zero being letter O in seascape data
# df['date'] = df['time'].apply(lambda x: f"{x.year}" + f"-{x.month}".zfill(3) + f"-{x.day}".zfill(3))
# df = df[['date', 'lon', 'lat', 'CLASS', 'P']]
# df.columns = ['date', 'lon', 'lat', 'seascape_class', 'seascape_prob']
# df.to_feather('data/patagonia_shelf_seascapes_2012-2016.feather')

#Parse: SST Temperature -----------------------------------------------

# files = glob.glob('/data2/SST/8DAY/*.nc')

# rdat = pd.DataFrame()
# for file_ in files:
#     ds = xr.open_dataset(file_, drop_variables=['qual_sst', 'palette'])
#     df = ds.to_dataframe().reset_index()
#     df = df[(df['lon'] >= LON1) & (df['lon'] <= LON2)]
#     df = df[(df['lat'] >= LAT1) & (df['lat'] <= LAT2)]
#     df['date'] = ds.time_coverage_start
#     year = pd.DatetimeIndex(df['date'])[0].year
#     month = pd.DatetimeIndex(df['date'])[0].month
#     day = pd.DatetimeIndex(df['date'])[0].day
#     df['date'] = f"{year}" + f"-{month}".zfill(3) + f"-{day}".zfill(3)
#     df = df[['date', 'lon', 'lat', 'sst']]
#     rdat = pd.concat([rdat, df])
#     print(f"{year}" + f"-{month}".zfill(3) + f"-{day}".zfill(3))



# rdat = rdat.reset_index()
# rdat = rdat[['date', 'lon', 'lat', 'sst']]
# rdat.to_feather('data/patagonia_shelf_SST_2012-2016.feather')


# Parse: SST Gradient
# def sst_gradient(sst_dat):
#     sst_dat = sst_dat.sort_values('date')
    
#     # Check dates are consistent
#     check1 = sst_dat.date.iat[0] == '2012-01-01'
#     check2 = sst_dat.date.iat[-1] == '2016-12-26'
#     print(sst_dat.lon_lat.iat[0], check1, check2) if( (check1 == False) | (check2 == False)) else None
    
#     # Get sst and interpolate NA
#     sst = sst_dat['sst'].copy()
#     nans, x= np.isnan(sst), lambda z: z.to_numpy().nonzero()[0]
#     sst[nans] = np.interp(x(nans), x(~nans), sst[~nans])
    
#     # Get gradient
#     sst_dat.loc[:, 'sst_grad'] = np.gradient(sst)
#     return sst_dat

   
# rdat = pd.read_feather('data/patagonia_shelf_SST_2012-2016.feather')
# rdat.loc[:, 'lon_lat'] = rdat.lon.astype(str) + '_' + rdat.lat.astype(str)

# # 230 days in date
# # Keep on those locations with more 
# ccount = rdat.groupby('lon_lat')['sst'].apply(lambda x: x.isnull().sum()).reset_index()
# ccount = ccount[ccount.sst <= 100]
# rdat = rdat[rdat.lon_lat.isin(ccount.lon_lat)]

# # Get gradient for each location
# rdat = rdat.sort_values('date')
# rdat = rdat.groupby('lon_lat').apply(lambda x: sst_gradient(x))
# rdat = rdat.reset_index(drop=True)

# rdat = rdat[['date', 'lon', 'lat', 'sst', 'sst_grad']]

# rdat.to_feather('data/patagonia_shelf_SST_2012-2016.feather')

# Debug test
# test = rdat[rdat.lon_lat == '-67.97916412353516_-50.10417556762695']
# test.head()
# test2 = test.groupby('lon_lat').apply(lambda x: sst_gradient(x))
# test2 = test2.reset_index(drop=True)
# test2.head()




#Parse: SST4 Temperature -----------------------------------------------

# files = glob.glob('/data2/SST4/8DAY/*.nc')

# rdat = pd.DataFrame()
# for file_ in files:
#     ds = xr.open_dataset(file_, drop_variables=['qual_sst4', 'palette'])
#     df = ds.to_dataframe().reset_index()
#     df = df[(df['lon'] >= LON1) & (df['lon'] <= LON2)]
#     df = df[(df['lat'] >= LAT1) & (df['lat'] <= LAT2)]
#     df['date'] = ds.time_coverage_start
#     year = pd.DatetimeIndex(df['date'])[0].year
#     month = pd.DatetimeIndex(df['date'])[0].month
#     day = pd.DatetimeIndex(df['date'])[0].day
#     df['date'] = f"{year}" + f"-{month}".zfill(3) + f"-{day}".zfill(3)
#     df['date'] = pd.DatetimeIndex(df['date']) + pd.DateOffset(1)
#     df = df[['date', 'lon', 'lat', 'sst4']]
#     rdat = pd.concat([rdat, df])
#     print(df.date.iat[0])
    



# rdat = rdat.reset_index()
# rdat = rdat[['date', 'lon', 'lat', 'sst4']]
# rdat.to_feather('data/patagonia_shelf_SST4_2012-2016.feather')


# # Parse: SST4 Gradient
# def sst4_gradient(sst_dat):
#     #print(sst_dat.head())
#     sst_dat = sst_dat.sort_values('date')
    
#     # Check dates are consistent
#     #check1 = sst_dat.date.iat[0] == '2012-01-01'
#     #check2 = sst_dat.date.iat[-1] == '2016-12-26'
#     #print(sst_dat.lon_lat.iat[0], check1, check2) if( (check1 == False) | (check2 == False)) else None
    
#     # Get sst and interpolate NA
#     sst = sst_dat['sst4'].copy()
#     nans, x= np.isnan(sst), lambda z: z.to_numpy().nonzero()[0]
#     sst[nans] = np.interp(x(nans), x(~nans), sst[~nans])
    
#     # Get gradient
#     sst_dat.loc[:, 'sst4_grad'] = np.gradient(sst)
#     return sst_dat

   
# rdat = pd.read_feather('data/patagonia_shelf_SST4_2012-2016.feather')
# rdat.loc[:, 'lon_lat'] = rdat.lon.astype(str) + '_' + rdat.lat.astype(str)

# # 230 days in date
# # Keep on those locations with more 
# ccount = rdat.groupby('lon_lat')['sst4'].apply(lambda x: x.isnull().sum()).reset_index()
# ccount = ccount[ccount.sst4 <= 100]
# rdat = rdat[rdat.lon_lat.isin(ccount.lon_lat)]
# rdat = rdat.reset_index(drop=True)



# # Get gradient for each location
# rdat = rdat.sort_values('date')
# rdat = rdat.groupby('lon_lat').apply(lambda x: sst4_gradient(x))
# rdat = rdat.reset_index(drop=True)

# rdat = rdat[['date', 'lon', 'lat', 'sst4', 'sst4_grad']]

# rdat.to_feather('data/patagonia_shelf_SST4_2012-2016.feather')






# Parse: CHL -----------------------------------------------

# files = glob.glob('/data2/CHL/NC/8DAY/*.nc')

# rdat = pd.DataFrame()
# for file_ in files:
#     ds = xr.open_dataset(file_, drop_variables=['palette'])
#     df = ds.to_dataframe().reset_index()
#     df = df[(df['lon'] >= LON1) & (df['lon'] <= LON2)]
#     df = df[(df['lat'] >= LAT1) & (df['lat'] <= LAT2)]
#     df['date'] = ds.time_coverage_start
#     year = pd.DatetimeIndex(df['date'])[0].year
#     month = pd.DatetimeIndex(df['date'])[0].month
#     day = pd.DatetimeIndex(df['date'])[0].day
#     df['date'] = f"{year}" + f"-{month}".zfill(3) + f"-{day}".zfill(3)
#     df = df[['date', 'lon', 'lat', 'chlor_a']]
#     rdat = pd.concat([rdat, df])
#     print(f"{year}" + f"-{month}".zfill(3) + f"-{day}".zfill(3))

# rdat = rdat.reset_index()
# rdat = rdat[['date', 'lon', 'lat', 'chlor_a']]
# rdat.to_feather('data/patagonia_shelf_CHL_2012-2016.feather')

#--------------------------------------------------------------------------------

# Load processed files

# gfw data
# lat_bin southern edge of grid
# lon_bin western edge of grid
# lat2 northern edge
# lon2 eastern edge

# Global fish watch 10d processed data
# gfw = pd.read_feather("data/patagonia_shelf_gfw_effort_10d_data.feather")

# # seascape data
# sea = pd.read_feather("data/patagonia_shelf_seascapes_2012-2016.feather")

# # sea surface temp
# sst = pd.read_feather('data/patagonia_shelf_SST_2012-2016.feather')

# # sea surface temp
# sst4 = pd.read_feather('data/patagonia_shelf_SST4_2012-2016.feather')

# # Chlor
# chl = pd.read_feather('data/patagonia_shelf_CHL_2012-2016.feather')

#gfw.head()
#sea.head()
#sst.head()

# For each day
# in each year
# Find seascape for each fishing_hours

# gfw['year'] = pd.DatetimeIndex(gfw['date']).year
# sea['year'] = pd.DatetimeIndex(sea['date']).year
# sst['year'] = pd.DatetimeIndex(sst['date']).year
# chl['year'] = pd.DatetimeIndex(chl['date']).year

# gfw['month'] = pd.DatetimeIndex(gfw['date']).month
# sea['month'] = pd.DatetimeIndex(sea['date']).month
# sst['month'] = pd.DatetimeIndex(sst['date']).month
# chl['month'] = pd.DatetimeIndex(chl['date']).month

# gfw['day'] = pd.DatetimeIndex(gfw['date']).day
# sea['day'] = pd.DatetimeIndex(sea['date']).day
# sst['day'] = pd.DatetimeIndex(sst['date']).day
# chl['day'] = pd.DatetimeIndex(chl['date']).day

#gfw = gfw[(gfw['year'] == 2016) & (gfw['month'] == 1) & (gfw['day'] == 1)]
#sea = sea[(sea['year'] == 2016) & (sea['month'] == 1) & (sea['day'] == 1)]


#len(gfw)
#len(sea)

#gfw = gfw.head(50)
#sea = sea.head(50)


# def dist(lat1, lon1, lat2, lon2):
#     return np.sqrt( (lat2 - lat1)**2 + (lon2 - lon1)**2)



# # Get seascape lat/lon
# def find_seascape(lat, lon):
#     #lat = -51
#     #lon = -68
#     lat1 = lat - .5
#     lat2 = lat + .5
#     lon1 = lon - .5
#     lon2 = lon + .5
#     indat = sea[(sea['lon'].values >= lon1) & (sea['lon'].values <= lon2) & (sea['lat'].values >= lat1) & (sea['lat'].values <= lat2)] 
#     #distances = indat.apply(lambda row: dist(lat, lon, row['lat'], row['lon']), axis=1)
    
#     distances = indat.apply(
#         lambda row: dist(lat, lon, row['lat'], row['lon']), 
#         axis=1)
    
#     #rdat = pd.DataFrame({"seascape_class": [indat.loc[distances.idxmin(), 'seascape_class']], "seascape_prob": [indat.loc[distances.idxmin(), 'seascape_prob']]})
#     #print(rdat)
#     #return rdat
#     return (indat.loc[distances.idxmin(), 'seascape_class'], indat.loc[distances.idxmin(), 'seascape_prob'])



# def find_sst(lat, lon):

#     lat1 = lat - .5
#     lat2 = lat + .5
#     lon1 = lon - .5
#     lon2 = lon + .5

#     indat = sst[(sst['lon'].values >= lon1) & (sst['lon'].values <= lon2) & (sst['lat'].values >= lat1) & (sst['lat'].values <= lat2)] 
    
#     distances = indat.apply(
#         lambda row: dist(lat, lon, row['lat'], row['lon']), 
#         axis=1)

#     #distances = indat.apply(lambda row: dist(lat, lon, row['lat'], row['lon']), axis=1)
#     #print(indat.loc[distances.idxmin(), ['sst', 'lon', 'lat']])
#     return (indat.loc[distances.idxmin(), 'sst'], indat.loc[distances.idxmin(), 'sst_grad'])


# def find_sst4(lat, lon):

#     lat1 = lat - .5
#     lat2 = lat + .5
#     lon1 = lon - .5
#     lon2 = lon + .5

#     indat = sst4[(sst4['lon'].values >= lon1) & (sst4['lon'].values <= lon2) & (sst4['lat'].values >= lat1) & (sst4['lat'].values <= lat2)] 
    
#     distances = indat.apply(
#         lambda row: dist(lat, lon, row['lat'], row['lon']), 
#         axis=1)

#     #distances = indat.apply(lambda row: dist(lat, lon, row['lat'], row['lon']), axis=1)
#     #print(indat.loc[distances.idxmin(), ['sst', 'lon', 'lat']])
#     return (indat.loc[distances.idxmin(), 'sst4'], indat.loc[distances.idxmin(), 'sst4_grad'])



# def find_chlor(lat, lon):

#     lat1 = lat - .5
#     lat2 = lat + .5
#     lon1 = lon - .5
#     lon2 = lon + .5

#     indat = chl[(chl['lon'].values >= lon1) & (chl['lon'].values <= lon2) & (chl['lat'].values >= lat1) & (chl['lat'].values <= lat2)] 
    
#     distances = indat.apply(lambda row: dist(lat, lon, row['lat'], row['lon']), axis=1)
        
#     return indat.loc[distances.idxmin(), 'chlor_a']



# #@ray.remote
# def process_days(dat):
#     date = dat['date'].iat[0]

#     #print(f"Processing data for: {date}")

#     #print("1-Linking Effort and Seascape")
#     # Link seascape to effort
#     dat.loc[:, 'seascape_class'], dat.loc[:, 'seascape_prob'] = zip(*dat.apply(lambda row: find_seascape(row['lat1'], row['lon1']), axis=1))
    
    
#     #print("2-Linking Effort and SST")
#     # Link sst to effort
#     dat.loc[:, 'sst'], dat.loc[:, 'sst_grad'] = zip(*dat.apply(lambda row: find_sst(row['lat1'], row['lon1']), axis=1))

#     # SST 4
#     dat.loc[:, 'sst4'], dat.loc[:, 'sst4_grad'] = zip(*dat.apply(lambda row: find_sst4(row['lat1'], row['lon1']), axis=1))

#     #print("3-Linking Effort and CHL")
#     # Link sst to effort
#     dat.loc[:, 'chlor_a'] = dat.apply(lambda row: find_chlor(row['lat1'], row['lon1']), axis=1)

#     print(f"4-Save data to data/processed/10d/processed_{date}.feather")
#     # Save data
#     outdat = dat.reset_index(drop=True)
#     outdat.to_feather(f"data/processed/10d/processed_{date}.feather")
#     #print(f"{date}: COMPLETE")

#     #return outdat




# gb = gfw.groupby('date')
# days = [gb.get_group(x) for x in gb.groups]

# # Debug
# days = days[0].reset_index(drop=True)
# days = days.loc[1:5, :]
# dat = days
# process_days(days)

#dat = days[141]

#process_days(dat)

#test2 = sst[sst.date == '2012-01-01']

#test2.head()

#ray.init()
#1626474458
#1000000000
#0000
#results = ray.get([process_days.remote(i) for i in days])

#
#ray.shutdown()


# pool = multiprocessing.Pool(50)
# pool.map(process_days, days)
# pool.close()

# Combine processed files
# files = glob.glob('data/processed/10d/*.feather')
# files
# list_ = []
# for file in files:
#     df = pd.read_feather(file)
#     list_.append(df)
#     mdat = pd.concat(list_, sort=False)

# # Manually remove veseels near land in eez
# mdat.loc[:, 'eez'] = np.where(np.logical_and(mdat['lat1'] > -48, mdat['lon1'] < -65), True, mdat['eez'])
# mdat.loc[:, 'eez'] = np.where(np.logical_and(mdat['lat1'] > -44, mdat['lon1'] < -63), True, mdat['eez'])
# mdat.loc[:, 'eez'] = np.where(np.logical_and(mdat['lat1'] > -40, mdat['lon1'] < -61), True, mdat['eez'])

# mdat = mdat.reset_index(drop=True)
# mdat.to_feather('data/full_gfw_10d_effort_model_data_8DAY_2012-01-01_2016-12-26.feather')

#------------------------------------------------------------------
# Calculate distance to shore

def extract_geom_meta(country):
    '''
    extract from each geometry the name of the country
    and the geom_point data. The output will be a list
    of tuples and the country name as the last element.
    '''
    geoms = country.geometry
    coords = np.empty(shape=[0, 2])
    for geom in geoms:
        coords = np.append(coords, geom.exterior.coords, axis = 0)
    country_name = country.attributes["ADMIN"]
    return [coords, country_name]



def get_coastline(country_abb):
    ne_earth = shpreader.natural_earth(resolution = '10m',
                                    category = 'cultural',
                                    name='admin_0_countries')
    reader = shpreader.Reader(ne_earth)
    countries = reader.records()
    country = next(countries)
    # extract and create separate objects
    ctry = []
    for country in countries:
        #country_list.append(f"{country.attributes['ADM0_A3']}") 
        ctry.append(extract_geom_meta(country)) if country.attributes['ADM0_A3'] == country_abb else None
    lon = [item[0] for item in ctry[0][0]]
    lat = [item[1] for item in ctry[0][0]]
    df_ctry = pd.DataFrame({"country": country_abb, "lon": lon, "lat": lat})
    return df_ctry


def get_ports():
    ports = []
    ne_ports = shpreader.natural_earth(resolution = '10m',
                                    category = 'cultural',
                                    name='ports')
    reader = shpreader.Reader(ne_ports)                             
    ports = reader.records()
    port = next(ports)
    port_df = pd.DataFrame()
    for port in ports:
        geom_port = port.geometry
        geom_coords = geom_port.coords[:]
        lon = geom_coords[0][0]
        lat = geom_coords[0][1]
        port_name = port.attributes['name']
        odat = pd.DataFrame({'port': port_name, 'lon': [lon], 'lat': [lat]})
        port_df = pd.concat([port_df, odat])
    return port_df



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



def coast_dist(lon, lat):
    lat1 = lat - 5
    lat2 = lat + 5
    indat = coastline[(coastline['lat'].values >= lat1) & (coastline['lat'].values <= lat2)] 
    distances = indat.apply(lambda row: haversine(lon, lat, row['lon'], row['lat']), axis=1)
    return distances[distances.idxmin()]




def port_dist(lon, lat):
    indat = ports
    indat.loc[:, 'distance'] = indat.apply(lambda row: haversine(lon, lat, row['lon'], row['lat']), axis=1)
    indat = indat.sort_values('distance')
    return (indat['port'].iat[0], indat['distance'].iat[0])



def get_depth(lon, lat):
    request = Request('https://maps.googleapis.com/maps/api/elevation/json?locations={0},{1}&key={2}'.format(lat, lon, gkey))
    response = urlopen(request).read() 
    places = loads(response)
    return places['results'][0]['elevation']


# Get Google API key
api_key = open('Google_api_key.txt', 'r')
gkey = api_key.read()

# from pandarallel import pandarallel
# pandarallel.initialize(progress_bar=True)

dat = pd.read_feather('data/full_gfw_10d_effort_model_data_8DAY_2012-01-01_2016-12-26.feather')
#dat = dat.loc[0:5, :]

tqdm.pandas()

# Get depth in meters
dat['depth_m'] = dat.progress_apply(lambda x: get_depth(x['lon1'], x['lat1']), axis=1)
dat.head()

dat.to_feather('data/full_gfw_10d_effort_model_data_8DAY_2012-01-01_2016-12-26.feather')

# Distance to coastline
coastline = get_coastline('ARG')
dat['coast_dist_km'] = dat.progress_apply(lambda x: coast_dist(x['lon1'], x['lat1']), axis=1)
dat.head()

dat.to_feather('data/full_gfw_10d_effort_model_data_8DAY_2012-01-01_2016-12-26.feather')

# Distance to closests port
ports = get_ports()
dat['port'], dat['port_dist_km'] = zip(*dat.progress_apply(lambda x: port_dist(x['lon1'], x['lat1']), axis=1))
dat.head()

dat.to_feather('data/full_gfw_10d_effort_model_data_8DAY_2012-01-01_2016-12-26.feather')


mdat = pd.read_feather('data/full_gfw_10d_effort_model_data_8DAY_2012-01-01_2016-12-26.feather')
#mdat.to_feather('data/test.feather')

