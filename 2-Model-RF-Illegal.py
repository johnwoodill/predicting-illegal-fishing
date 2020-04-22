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
from sklearn.metrics import confusion_matrix

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

# Get lag seascape t-1
dat['seascape_lag'] = dat.groupby(['date', 'mmsi'])['seascape_class'].shift(-1)

# Linear model
# Get data frame of variables and dummy seascapes
# moddat = dat[['illegal', 'year', 'fishing_hours', 'month_abbr', 'seascape_class', 'sst', 'sst_grad', 'chlor_a', 'lon1', 'lat1', 'depth_m', 'coast_dist_km', 'port_dist_km', 'eez', 'distance_to_eez_km']].dropna().reset_index(drop=True)

# Include measures of distance
# moddat = dat[['illegal', 'year', 'month_abbr',
#              'seascape_class', 'sst', 'chlor_a', 'lon1', 'lat1', 
#              'coast_dist_km', 'port_dist_km', 'eez', 'distance_to_eez_km']].dropna().reset_index(drop=True)

# Drop measures of distance
moddat = dat[['illegal', 'year', 'month_abbr',
             'seascape_class', 'sst', 'chlor_a', 'lon1', 'lat1', 'eez']].dropna().reset_index(drop=True)
 
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

# Cross-validate model
feffort = pd.DataFrame()
sdat = pd.DataFrame()
feadat = pd.DataFrame()

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
    y_train = y_train['illegal']
    y_test = y_test['illegal']
   
    #clf = LogisticRegression().fit(X_train, y_train)
    # Parameter tuning
    # n_estimators: 1600, min_samples_split: 2, min_samples_leaf:2, max_depth:40, bootstrap:True
    clf = RandomForestClassifier(n_estimators = 100, random_state=123).fit(X_train, y_train)
    # clf = RandomForestClassifier(n_estimators=1600, 
    #                              min_samples_split=2,
    #                              max_depth=40,
    #                              bootstrap=True).fit(X_train, y_train)
    # Get predicted probabilities for sensitivity analysis
    
    # Get predicted probabilities
    pred_proba = clf.predict_proba(X_test)
    proba = pred_proba[:, 1]

    # Precision-recall across thresholds
    # Precision = tp / ( tp + fp )
    # Recall = tn / ( tn + fp )
    precision, recall, thresholds = precision_recall_curve(y_test, proba )
    
    # Test predictions
    y_pred = clf.predict(X_test)
    
    # F1 score = 2 * precision * recall / precision + recall
    f1 = f1_score(y_test, y_pred)
    
    # calculate precision-recall AUC
    auc_m = auc(recall, precision)

    # Specificity = TN/(TN+FP)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    specificity = tn / (tn+fp)
    
    
    # Feature importance
    fea_import = pd.DataFrame({'variable': X_train.columns , 'importance': clf.feature_importances_, 'year': year})
    feadat = pd.concat([feadat, fea_import])
    
    # calculate average precision score
    ap = average_precision_score(y_test, proba)
    print('f1=%.3f auc=%.3f ap=%.3f' % (f1, auc_m, ap))
    ddat = pd.DataFrame({'year': year, 'prec': precision, 'recall': recall, 'f1': f1, 'auc': auc_m, 'ap':ap, 'spec': specificity, 'sens':sensitivity})
    sdat = pd.concat([sdat, ddat])





# Save precision-recall data
sdat = sdat.reset_index(drop=True)
sdat.to_feather('data/illegal_cross_val_dat.feather')

# Feature Importance
feadat = feadat.reset_index(drop=True)
feadat.to_feather('data/feature_importance_rf_illegal.feather')


# Feature importance
fea_import = pd.DataFrame({'variable': X_train.columns , 'importance': clf.feature_importances_})
fea_import = fea_import.sort_values('importance', ascending=False)
fea_import = fea_import.reset_index(drop=True)
fea_import.to_feather('data/feature_importance_rf_illegal.feather')

sns.barplot('importance', 'variable', data = fea_import)
plt.show()

