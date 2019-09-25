import pandas as pd
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

#Fully processed data
# dat.to_feather('data/full_gfw_10d_illegal_model_data_DAILY_2012-01-01_2016-12-31.feather')

# 8Day data
dat = pd.read_feather('~/Projects/Seascape-and-fishing-effort/data/full_gfw_10d_effort_model_data_8DAY_2012-01-01_2016-12-26.feather')

# DAILY data
#dat = pd.read_feather('~/Projects/predicting-illegal-fishing/data/full_gfw_10d_illegal_model_data_DAILY_2012-01-01_2016-12-31.feather')

# Subset drifting longlines
#dat = dat[dat.geartype == 'drifting_longlines']

# Keep only vessel with fishing hours
#dat = dat[dat['fishing_hours'] > 0 ]

# Keep only Chinese vessels
dat = dat[(dat['flag'] == 'CHN') | (dat['flag'] == 'ARG')]

# If illegally operating inside EEZ (!= ARG)
dat.loc[:, 'illegal'] = np.where(((dat['eez'] == True) & (dat['fishing_hours'] > 0) & (dat['flag'] != 'ARG') ), 1, 0)

# Illegal if within 50km
dat.loc[:, 'illegal_5km'] = np.where(((dat['eez'] == True) & (dat['fishing_hours'] > 0 ) & (dat['distance_to_eez_km'] >= 5) & (dat['flag'] != 'ARG')), 1, 0)

# Illegal if within 100km
dat.loc[:, 'illegal_10km'] = np.where(((dat['eez'] == True) & (dat['fishing_hours'] > 0 ) & (dat['distance_to_eez_km'] >= 10) & (dat['flag'] != 'ARG')), 1, 0)

# Convert true/false eez to 0/1
dat.loc[:, 'illegal'] = dat.illegal.astype('uint8')
dat.loc[:, 'illegal_5km'] = dat.illegal_5km.astype('uint8')
dat.loc[:, 'illegal_10km'] = dat.illegal_10km.astype('uint8')

sum(dat.illegal)/len(dat)
sum(dat.illegal_5km)/len(dat)
sum(dat.illegal_10km)/len(dat)

# Get year month
dat.loc[:, 'year'] = pd.DatetimeIndex(dat['date']).year
dat.loc[:, 'month'] = pd.DatetimeIndex(dat['date']).month

# Convert month number to name
dat.loc[:, 'month_abbr'] = dat.apply(lambda x: calendar.month_abbr[x['month']], 1)





#-----------------------------------------------------
# Predicting illegal 5km out
# Linear model
# Get data frame of variables and dummy seascapes
moddat = dat[['illegal_5km', 'year', 'fishing_hours', 'month_abbr', 'seascape_class', 'sst', 'sst_grad', 'chlor_a', 'lon1', 'lat1', 'depth_m', 'coast_dist_km', 'port_dist_km', 'eez', 'distance_to_eez_km']].dropna().reset_index(drop=True)

# Dummy variables for seascape and dummies
seascape_dummies = pd.get_dummies(moddat['seascape_class'], prefix='seascape').reset_index(drop=True)
month_dummies = pd.get_dummies(moddat['month_abbr']).reset_index(drop=True)

# Concat dummy variables
moddat = pd.concat([moddat, seascape_dummies, month_dummies], axis=1)

# Get X, y
y = moddat[['year', 'illegal_5km']].reset_index(drop=True)

# Drop dummy variables and prediction
moddat = moddat.drop(columns = ['month_abbr', 'illegal_5km', 'seascape_class'])

# Build data for model
X = moddat
X.columns
X.head()
y.head()

# Cross-validate model
roc_dat = pd.DataFrame()
tpr_fpr = pd.DataFrame()
feffort = pd.DataFrame()
sdat = pd.DataFrame()
for year in range(2012, 2017):
    
    # Get training data
    X_train = X[X.year != year]
    y_train = y[y.year != year]

    # Get test data
    X_test = X[X.year == year]
    y_test = y[y.year == year]

    print(f"Training Years: {X_train.year.unique()} - Test Year: {X_test.year.unique()}")

    fishing_hours = X_test['fishing_hours']

    # Drop year
    X_train = X_train.drop(columns = ['year', 'fishing_hours'])
    X_test = X_test.drop(columns = ['year', 'fishing_hours'])

    # Set binary variable
    y_train = y_train['illegal_5km']
    y_test = y_test['illegal_5km']
    
    #clf = LogisticRegression().fit(X_train, y_train)
    clf = RandomForestClassifier(n_estimators=100).fit(X_train, y_train)

    # Get predicted probabilities for sensitivity analysis
    pred_proba = clf.predict_proba(X_test)
    proba = pred_proba[:, 1]

    # Get predictions and fishing hours
    y_pred = clf.predict(X_test)

    # data frame of true, pred, and fishing hours
    pred_dat = pd.DataFrame({'y_true': y_test, 'y_pred': y_pred, 'fishing_hours': fishing_hours})
    
    tf_dat = pred_dat[pred_dat['y_true'] == 1]
    tf = sum(tf_dat['fishing_hours'])

    # True positive
    fh_tpr_dat = pred_dat[(pred_dat['y_true'] == 1) & (pred_dat['y_pred'] == 1)]

    # False positive
    fh_fpr_dat = pred_dat[((pred_dat['y_true'] == 0) & (pred_dat['y_pred'] != 0)) | ((pred_dat['y_true'] == 1) & (pred_dat['y_pred'] != 1))]
    
    tpr_fishing_hours = sum(fh_tpr_dat['fishing_hours'])
    fpr_fishing_hours = sum(fh_fpr_dat['fishing_hours'])
    total_fishing_hours = sum(pred_dat['fishing_hours'])

    feffort_indat = pd.DataFrame({'year': [year], 'total_fishing': [tf], 'tpr_fishing': [tpr_fishing_hours], 'fpr_fishing_hours': [fpr_fishing_hours]})
    feffort = pd.concat([feffort, feffort_indat])
    
    # Precision-recall across thresholds
    # Precision = tp / ( tp + fp )
    # Recall = tn / ( tn + fp )
    precision, recall, thresholds = precision_recall_curve(y_test, proba)

    # Test predictions
    y_pred = clf.predict(X_test)

    # F1 score = 2 * precision * recall / precision + recall
    f1 = f1_score(y_test, y_pred)

    # calculate precision-recall AUC
    auc_m = auc(recall, precision)

    # calculate average precision score
    ap = average_precision_score(y_test, proba)
    print('f1=%.3f auc=%.3f ap=%.3f' % (f1, auc_m, ap))

    ddat = pd.DataFrame({'year': year, 'prec': precision, 'recall': recall, 'f1': f1, 'auc':auc_m, 'ap':ap})

    sdat = pd.concat([sdat, ddat])

# Save fishing hours data
feffort = feffort.reset_index(drop=True)
feffort = feffort.to_feather('data/predicted_5k_effort_data.feather')

# Save precision-recall data
sdat = sdat.reset_index(drop=True)
sdat.to_feather('data/illegal_5k_cross_val_dat.feather')






#-----------------------------------------------------
# Predicting illegal 5km out
# Linear model
# Get data frame of variables and dummy seascapes
moddat = dat[['illegal_10km', 'year', 'fishing_hours', 'month_abbr', 'seascape_class', 'sst', 'sst_grad', 'chlor_a', 'lon1', 'lat1', 'depth_m', 'coast_dist_km', 'port_dist_km', 'eez', 'distance_to_eez_km']].dropna().reset_index(drop=True)

# Dummy variables for seascape and dummies
seascape_dummies = pd.get_dummies(moddat['seascape_class'], prefix='seascape').reset_index(drop=True)
month_dummies = pd.get_dummies(moddat['month_abbr']).reset_index(drop=True)

# Concat dummy variables
moddat = pd.concat([moddat, seascape_dummies, month_dummies], axis=1)

# Get X, y
y = moddat[['year', 'illegal_10km']].reset_index(drop=True)

# Drop dummy variables and prediction
moddat = moddat.drop(columns = ['month_abbr', 'illegal_10km', 'seascape_class'])

# Build data for model
X = moddat
X.columns
X.head()
y.head()

# Cross-validate model
roc_dat = pd.DataFrame()
tpr_fpr = pd.DataFrame()
feffort = pd.DataFrame()
sdat = pd.DataFrame()
for year in range(2012, 2017):
    
    # Get training data
    X_train = X[X.year != year]
    y_train = y[y.year != year]

    # Get test data
    X_test = X[X.year == year]
    y_test = y[y.year == year]

    print(f"Training Years: {X_train.year.unique()} - Test Year: {X_test.year.unique()}")

    fishing_hours = X_test['fishing_hours']

    # Drop year
    X_train = X_train.drop(columns = ['year', 'fishing_hours'])
    X_test = X_test.drop(columns = ['year', 'fishing_hours'])

    # Set binary variable
    y_train = y_train['illegal_10km']
    y_test = y_test['illegal_10km']
    
    #clf = LogisticRegression().fit(X_train, y_train)
    clf = RandomForestClassifier(n_estimators=100).fit(X_train, y_train)

    # Get predicted probabilities for sensitivity analysis
    pred_proba = clf.predict_proba(X_test)
    proba = pred_proba[:, 1]

    # Get predictions and fishing hours
    y_pred = clf.predict(X_test)

    # data frame of true, pred, and fishing hours
    pred_dat = pd.DataFrame({'y_true': y_test, 'y_pred': y_pred, 'fishing_hours': fishing_hours})
    
    tf_dat = pred_dat[pred_dat['y_true'] == 1]
    tf = sum(tf_dat['fishing_hours'])

    # True positive
    fh_tpr_dat = pred_dat[(pred_dat['y_true'] == 1) & (pred_dat['y_pred'] == 1)]

    # False positive
    fh_fpr_dat = pred_dat[((pred_dat['y_true'] == 0) & (pred_dat['y_pred'] != 0)) | ((pred_dat['y_true'] == 1) & (pred_dat['y_pred'] != 1))]
    
    tpr_fishing_hours = sum(fh_tpr_dat['fishing_hours'])
    fpr_fishing_hours = sum(fh_fpr_dat['fishing_hours'])
    total_fishing_hours = sum(pred_dat['fishing_hours'])

    feffort_indat = pd.DataFrame({'year': [year], 'total_fishing': [tf], 'tpr_fishing': [tpr_fishing_hours], 'fpr_fishing_hours': [fpr_fishing_hours]})
    feffort = pd.concat([feffort, feffort_indat])
    
    # Precision-recall across thresholds
    # Precision = tp / ( tp + fp )
    # Recall = tn / ( tn + fp )
    precision, recall, thresholds = precision_recall_curve(y_test, proba)

    # Test predictions
    y_pred = clf.predict(X_test)

    # F1 score = 2 * precision * recall / precision + recall
    f1 = f1_score(y_test, y_pred)

    # calculate precision-recall AUC
    auc_m = auc(recall, precision)

    # calculate average precision score
    ap = average_precision_score(y_test, proba)
    print('f1=%.3f auc=%.3f ap=%.3f' % (f1, auc_m, ap))

    ddat = pd.DataFrame({'year': year, 'prec': precision, 'recall': recall, 'f1': f1, 'auc':auc_m, 'ap':ap})

    sdat = pd.concat([sdat, ddat])

# Save fishing hours data
feffort = feffort.reset_index(drop=True)
feffort = feffort.to_feather('data/predicted_10k_effort_data.feather')

# Save precision-recall data
sdat = sdat.reset_index(drop=True)
sdat.to_feather('data/illegal_10k_cross_val_dat.feather')