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
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
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
dat = dat[dat.geartype == 'drifting_longlines']

# If illegally operating inside EEZ (!= ARG)
dat.loc[:, 'illegal'] = np.where(((dat['eez'] == True) & (dat['flag'] != 'ARG') & (dat['flag'] != 'URY')), 1, 0)

# Convert true/false eez to 0/1
dat.loc[:, 'eez'] = dat.eez.astype('uint8')

# Calc distance from eez for illegal vessels
dat.loc[:, 'illegal_distance_to_eez_km'] = np.where((dat['illegal'] == 1), dat['distance_to_eez_km'], 0)

mdat = dat[dat['illegal_distance_to_eez_km'] != 99999999]
mdat

mdat.loc[:, 'year'] = pd.DatetimeIndex(mdat['date']).year

mdat.loc[:, 'month'] = pd.DatetimeIndex(mdat['date']).month

# Convert month number to name
mdat.loc[:, 'month_abbr'] = mdat.apply(lambda x: calendar.month_abbr[x['month']], 1)

# Linear model
#moddat = mdat[['illegal_distance_to_eez_km', 'year', 'month_abbr', 'seascape_class', 'sst', 'sst_grad', 'sst4', 'sst4_grad', 'chlor_a', 'lon1', 'lat1', 'depth_m', 'coast_dist_km', 'port_dist_km']].dropna().reset_index(drop=True)

moddat = mdat[['illegal_distance_to_eez_km', 'seascape_class', 'year', 'month_abbr', 'sst', 'sst4', 'chlor_a', 'lon1', 'lat1', 'coast_dist_km', 'port_dist_km']].dropna().reset_index(drop=True)

seascape_dummies = pd.get_dummies(moddat['seascape_class'], prefix='seascape').reset_index(drop=True)
month_dummies = pd.get_dummies(moddat['month_abbr']).reset_index(drop=True)

moddat = pd.concat([moddat, seascape_dummies, month_dummies], axis=1)
moddat.head()

# Get X, y
y = moddat[['year', 'illegal_distance_to_eez_km']].reset_index(drop=True)

moddat = moddat.drop(columns = ['month_abbr', 'seascape_class', 'illegal_distance_to_eez_km'])
moddat.head()

X = moddat
X.columns
X.head()
y.head()

# Cross-validate model rolling year
pred_score = []
roc_dat = pd.DataFrame()
for year in range(2013, 2017):
    
    X_train = X[X.year != year]
    y_train = y[y.year != year]

    X_test = X[X.year == year]
    y_test = y[y.year == year]

    print(f"Training Years: {X_train.year.unique()} - Test Year: {X_test.year.unique()}")

    X_train = X_train.drop(columns = ['year'])
    X_test = X_test.drop(columns = ['year'])

    y_train = y_train['illegal_distance_to_eez_km']
    y_test = y_test['illegal_distance_to_eez_km']
    
    #clf = LogisticRegression().fit(X_train, y_train)
    clf = RandomForestRegressor(n_estimators=100).fit(X_train, y_train)
    y_test_pred = clf.predict(X_test)
    np.mean(y_test_pred**2)

    test = pd.DataFrame({'true': y_test, 'pred': y_test_pred})
    test[test.true > 0]

    rsqr_train = clf.score(X_train, y_train)
    #rsqr_train
    
    rsqr_test = clf.score(X_test, y_test)
    #rsqr_test

    mse_train = mean_squared_error(y_train, clf.predict(X_train))
    mse_test = mean_squared_error(y_test, clf.predict(X_test))
    
    outdat = pd.DataFrame({'pred_year': year, 'train_mse': [mse_train], 'test_mse': [mse_test], 'rsqr_test': [rsqr_test], 'rsqr_train': [rsqr_train]})
    roc_dat = pd.concat([roc_dat, outdat])
        
    # score = score(y_test, y_test_pred)
    # score
    # pred_score.append(score)

    # probs = clf.predict_proba(X_test)
    # preds = probs[:,1]
    # fpr, tpr, threshold = metrics.roc_curve(y_test, preds)
    # roc_auc = metrics.auc(fpr, tpr)

    # outdat = pd.DataFrame({'pred_year': year, 'fpr': fpr, 'tpr': tpr, 'thresh': threshold})
    # roc_dat = pd.concat([roc_dat, outdat])

print(pred_score)
print(roc_dat)

# Cross-validate model by folded-year
pred_score = []
roc_dat = pd.DataFrame()
for year in range(2013, 2017):
    
    X_train = X[X.year != year]
    y_train = y[y.year != year]

    X_test = X[X.year == year]
    y_test = y[y.year == year]

    print(f"Training Years: {X_train.year.unique()} - Test Year: {X_test.year.unique()}")

    X_train = X_train.drop(columns = ['year'])
    X_test = X_test.drop(columns = ['year'])

    y_train = y_train['illegal_distance_to_eez_km']
    y_test = y_test['illegal_distance_to_eez_km']
    
    #clf = LogisticRegression().fit(X_train, y_train)
    clf = RandomForestRegressor(n_estimators=100).fit(X_train, y_train)
    y_test_pred = clf.predict(X_test)
    
    rsqr_train = clf.score(X_train, y_train)
    #rsqr_train
    
    rsqr_test = clf.score(X_test, y_test)
    #rsqr_test

    mse_train = mean_squared_error(y_train, clf.predict(X_train))
    mse_test = mean_squared_error(y_test, clf.predict(X_test))
    
    outdat = pd.DataFrame({'pred_year': year, 'train_mse': [mse_train], 'test_mse': [mse_test], 'rsqr_test': [rsqr_test], 'rsqr_train': [rsqr_train]})
    roc_dat = pd.concat([roc_dat, outdat])


roc_dat['test_mse'] - roc_dat['train_mse']

# method I: plt
import matplotlib.pyplot as plt
plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()


clf = RandomForestClassifier().fit(X_train, y_train)
clf.predict_proba(X_train)

fea_import = pd.DataFrame({'variable': X_train.columns , 'importance': clf.feature_importances_})
fea_import = fea_import.sort_values('importance', ascending=False)
#fea_import.to_feather('data/feature_importance_rf_illegal.feather')
sns.barplot('importance', 'variable', data = fea_import)
plt.show()

