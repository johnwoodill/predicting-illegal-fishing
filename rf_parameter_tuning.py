from sklearn.model_selection import RandomizedSearchCV, cross_validate, cross_val_score
from sklearn.metrics import average_precision_score, accuracy_score, roc_curve, auc, precision_recall_curve, f1_score
from sklearn.ensemble import RandomForestClassifier
import sklearn.metrics as metrics
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit, GridSearchCV
import pandas as pd
import numpy as np
import calendar

# 8Day data
dat = pd.read_feather('data/full_gfw_10d_effort_model_data_8DAY_2012-01-01_2016-12-26.feather')

# Keep only Chinese vessels
dat = dat[(dat['flag'] == 'CHN') | (dat['flag'] == 'ARG')]

# If illegally operating inside EEZ (!= ARG)
dat.loc[:, 'illegal'] = np.where(((dat['eez'] == True) & (dat['fishing_hours'] > 0) & (dat['flag'] != 'ARG') ), 1, 0)

# Convert true/false eez to 0/1
dat.loc[:, 'illegal'] = dat.illegal.astype('uint8')

sum(dat.illegal)/len(dat)

# Get year month
dat.loc[:, 'year'] = pd.DatetimeIndex(dat['date']).year
dat.loc[:, 'month'] = pd.DatetimeIndex(dat['date']).month

# Convert month number to name
dat.loc[:, 'month_abbr'] = dat.apply(lambda x: calendar.month_abbr[x['month']], 1)

# Linear model
# Get data frame of variables and dummy seascapes
moddat = dat[['illegal', 'year', 'fishing_hours', 'month_abbr', 'seascape_class', 'sst', 'chlor_a', 'lon1', 'lat1', 'coast_dist_km', 'port_dist_km', 'eez', 'distance_to_eez_km']].dropna().reset_index(drop=True)

# Dummy variables for seascape and dummies
seascape_dummies = pd.get_dummies(moddat['seascape_class'], prefix='seascape').reset_index(drop=True)
month_dummies = pd.get_dummies(moddat['month_abbr']).reset_index(drop=True)

# Concat dummy variables
moddat = pd.concat([moddat, seascape_dummies, month_dummies], axis=1)

# Get X, y
y = moddat[['year', 'illegal']].reset_index(drop=True)

# Drop dummy variables and prediction
moddat = moddat.drop(columns = ['month_abbr', 'illegal', 'seascape_class'])

# Build data for model
X = moddat
X.columns
X.head()
y.head()



# Cross-validation data sets
cv1_train = X[X['year'] != 2012].index
cv1_test = X[X['year'] == 2012].index

cv2_train = X[X['year'] != 2013].index
cv2_test = X[X['year'] == 2013].index

cv3_train = X[X['year'] != 2014].index
cv3_test = X[X['year'] == 2014].index

cv4_train = X[X['year'] != 2015].index
cv4_test = X[X['year'] == 2015].index

cv5_train = X[X['year'] != 2016].index
cv5_test = X[X['year'] == 2016].index

custom_cv = [(np.array(cv1_train), np.array(cv1_test)),
             (np.array(cv2_train), np.array(cv2_test)), 
             (np.array(cv3_train), np.array(cv3_test)), 
             (np.array(cv4_train), np.array(cv4_test)),
             (np.array(cv5_train), np.array(cv5_test))]


# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]


# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
max_depth.append(None)

# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]

# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]

# Method of selecting samples for training each tree
bootstrap = [True, False]

# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}

print(random_grid)


X = X.drop(columns='year')
y = y.drop(columns = 'year')
y = y.illegal

clf = RandomForestClassifier(n_estimators=100)

rf_random = RandomizedSearchCV(estimator = clf, param_distributions = random_grid, 
                               n_iter = 100, cv = custom_cv, verbose=2, random_state=42, n_jobs = -1)

rf_random.fit(X, y)

print(rf_random.best_params_)

# n_estimators: 1600, min_samples_split: 2, min_samples_leaf:2, max_depth:40, bootstrap:True




