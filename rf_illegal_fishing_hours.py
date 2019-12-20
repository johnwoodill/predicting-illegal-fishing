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
from sklearn.model_selection import cross_validate, cross_val_score
from sklearn.metrics import mean_squared_error, average_precision_score, accuracy_score, roc_curve, auc, precision_recall_curve, f1_score
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import sklearn.metrics as metrics
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit, GridSearchCV
from collections import deque
import calendar
from sklearn.linear_model import LinearRegression

# 8Day data
dat = pd.read_feather('data/full_gfw_10d_effort_model_data_8DAY_2012-01-01_2016-12-26.feather')

# Keep only Chinese vessels
#dat = dat[(dat['flag'] == 'CHN') | (dat['flag'] == 'ARG')]

# If illegally operating inside EEZ (!= ARG)
dat.loc[:, 'illegal'] = np.where(((dat['eez'] == True) & (dat['fishing_hours'] > 0) & (dat['flag'] != 'ARG') ), 1, 0)

dat['illegal_fishing_effort'] = np.where((dat['illegal'] == True), dat['fishing_hours'], 0)

# Convert true/false eez to 0/1
dat.loc[:, 'illegal'] = dat.illegal.astype('uint8')

sum(dat.illegal)/len(dat)

# Get year month
dat.loc[:, 'year'] = pd.DatetimeIndex(dat['date']).year
dat.loc[:, 'month'] = pd.DatetimeIndex(dat['date']).month

# Convert month number to name
dat.loc[:, 'month_abbr'] = dat.apply(lambda x: calendar.month_abbr[x['month']], 1)


# Get data frame of variables and dummy seascapes
# moddat = dat[['illegal', 'year', 'fishing_hours', 'month_abbr', 'seascape_class', 'sst', 'sst_grad', 'chlor_a', 'lon1', 'lat1', 'depth_m', 'coast_dist_km', 'port_dist_km', 'eez', 'distance_to_eez_km']].dropna().reset_index(drop=True)

moddat = dat[['illegal_fishing_effort', 'year', 'month_abbr', 'seascape_class', 'sst', 
              'chlor_a', 'lon1', 'lat1', 'coast_dist_km', 'port_dist_km', 'eez', 
              'distance_to_eez_km']].dropna().reset_index(drop=True)

# moddat = dat[['illegal_fishing_effort', 'year', 'month_abbr', 'seascape_class', 'sst']].dropna().reset_index(drop=True)

# Dummy variables for seascape and dummies
seascape_dummies = pd.get_dummies(moddat['seascape_class'], prefix='seascape').reset_index(drop=True)
month_dummies = pd.get_dummies(moddat['month_abbr']).reset_index(drop=True)

# Concat dummy variables
moddat = pd.concat([moddat, seascape_dummies, month_dummies], axis=1)

# Get X, y
y = moddat[['year', 'illegal_fishing_effort']].reset_index(drop=True)

# Drop dummy variables and prediction
moddat = moddat.drop(columns = ['month_abbr', 'illegal_fishing_effort', 'seascape_class'])

# Build data for model
X = moddat
X.columns
X.head()
y.head()


# X = X.drop(columns='year')
# y = y.drop(columns='year').values


#clf = RandomForestRegressor(n_estimators = 100)
#cross_val_score(clf, X, y, cv=5)


# Cross-validate model
sdat = pd.DataFrame()
ddat = pd.DataFrame()
for year in range(2012, 2017):
    
    # Get training data
    X_train = X[X.year != year]
    y_train = y[y.year != year]

    # Get test data
    X_test = X[X.year == year]
    y_test = y[y.year == year]

    print(f"Training Years: {X_train.year.unique()} - Test Year: {X_test.year.unique()}")

    # Drop year
    X_train = X_train.drop(columns = ['year'])
    X_test = X_test.drop(columns = ['year'])

    # Set binary variable
    y_train = y_train['illegal_fishing_effort']
    y_test = y_test['illegal_fishing_effort']
    

    # Random Forest Regression
    clf = RandomForestRegressor(n_estimators = 100, random_state=123).fit(X_train, y_train)
    test_r2 = clf.score(X_test, y_test)
    train_r2 = clf.score(X_train, y_train)
       
    
    
    # Get predictions and fishing hours
    y_train_pred = clf.predict(X_train)
    y_train_true = y_train
    
    y_test_pred = clf.predict(X_test)
    y_test_true = y_test
    
    train_mse = mean_squared_error(y_train_true, y_train_pred)
    test_mse = mean_squared_error(y_test_true, y_test_pred)
    
    outdat = pd.DataFrame({'year': [year], 
                           'Train_MSE': round(train_mse, 4),
                           'Test_MSE': round(test_mse, 4),
                           'Train Coefficient of Det.': [test_r2],
                           'Test Coefficient of Det.': [train_r2]})
    sdat = pd.concat([sdat, outdat])
    print(f"Train MSE: {round(train_mse, 4)} ------ Test MSE: {round(test_mse, 4)}")
    
    outdat2 = pd.DataFrame({'year': year,
                            'y_true': y_test_true,
                            'y_pred': y_test_pred,
                            'lat': X_test.lat1,
                            'lon': X_test.lon1})
    ddat = pd.concat([ddat, outdat2])
    
print(sdat)

# Save fishing hours data
feffort = ddat.reset_index(drop=True)
feffort.to_feather('data/predicted_effort_data.feather')

# Feature importance
fea_import = pd.DataFrame({'variable': X_train.columns , 'importance': clf.feature_importances_})
fea_import = fea_import.sort_values('importance', ascending=False)
fea_import = fea_import.reset_index(drop=True)
fea_import.to_feather('data/feature_importance_rf_illegal.feather')

sns.barplot('importance', 'variable', data = fea_import)
plt.show()

