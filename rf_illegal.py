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
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
import sklearn.metrics as metrics
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit, GridSearchCV
from collections import deque
import calendar

#Fully processed data
# dat.to_feather('data/full_gfw_10d_illegal_model_data_DAILY_2012-01-01_2016-12-31.feather')

# 8Day data
dat = pd.read_feather('~/Projects/Seascape-and-fishing-effort/data/full_gfw_10d_effort_model_data_8DAY_2012-01-01_2016-12-26.feather')

# Subset drifting longlines
#dat = dat[dat.geartype == 'drifting_longlines']
#dat = dat.sort_values('date')



# If illegally operating inside EEZ (!= ARG)
dat.loc[:, 'illegal'] = np.where(((dat['eez'] == True) & (dat['flag'] != 'ARG') & (dat['flag'] != 'URY')), 1, 0)

# Illegal if within 10km
dat.loc[:, 'illegal_10km'] = np.where(((dat['eez'] == True) & (dat['distance_to_eez_km'] <= 10) & (dat['flag'] != 'ARG')), 1, 0)

# Illegal if within 20km
dat.loc[:, 'illegal_20km'] = np.where(((dat['eez'] == True) & (dat['distance_to_eez_km'] <= 20) & (dat['flag'] != 'ARG')), 1, 0)

# Illegal if within 30km
dat.loc[:, 'illegal_30km'] = np.where(((dat['eez'] == True) & (dat['distance_to_eez_km'] <= 30) & (dat['flag'] != 'ARG')), 1, 0)

# Convert true/false eez to 0/1
dat.loc[:, 'eez'] = dat.eez.astype('uint8')
dat.loc[:, 'eez_10km'] = dat.eez_10km.astype('uint8')
dat.loc[:, 'eez_20km'] = dat.eez_20km.astype('uint8')
dat.loc[:, 'eez_30km'] = dat.eez_30km.astype('uint8')

sum(dat.illegal_10km)/len(dat)
sum(dat.illegal_20km)/len(dat)
sum(dat.illegal_30km)/len(dat)

# Calc distance from eez for illegal vessels
dat.loc[:, 'illegal_distance_to_eez_km'] = np.where((dat['illegal'] == 1), dat['distance_to_eez_km'], 99999999)

test = dat[dat['illegal_distance_to_eez_km'] != 99999999]
test

# Convert month number to name
dat.loc[:, 'month_abbr'] = dat.apply(lambda x: calendar.month_abbr[x['month']], 1)

# Linear model
# Get data frame of variables and dummy seascapes
#moddat = dat[['illegal', 'fishing_hours', 'flag', 'year', 'month', 'seascape_class', 'sst', 'sst_grad', 'sst4', 'sst4_grad', 'chlor_a', 'lon1', 'lat1', 'tonnage', 'depth_m', 'coast_dist_km', 'port_dist_km', 'eez', 'distance_to_eez_km']].dropna().reset_index(drop=True)

#moddat = dat[['illegal', 'year', 'month_abbr', 'seascape_class', 'sst', 'sst_grad', 'sst4', 'sst4_grad', 'chlor_a', 'lon1', 'lat1', 'tonnage', 'depth_m', 'coast_dist_km', 'port_dist_km', 'eez', 'distance_to_eez_km']].dropna().reset_index(drop=True)

moddat = dat[['illegal', 'year', 'month_abbr', 'seascape_class', 'sst', 'sst_grad', 'sst4', 'sst4_grad', 'chlor_a', 'lon1', 'lat1', 'depth_m', 'coast_dist_km', 'port_dist_km', 'eez', 'distance_to_eez_km']].dropna().reset_index(drop=True)

'sst_grad', 'sst4', 'sst4_grad', 'chlor_a', 'lon1', 'lat1', 'depth_m', 'coast_dist_km', 'port_dist_km', 'eez', 'distance_to_eez_km'

moddat = dat[['illegal', 'year', 'sst', 'lat1', 'lon1']].dropna().reset_index(drop=True)

moddat.to_feather('data/test_illegal.feather')


seascape_dummies = pd.get_dummies(moddat['seascape_class'], prefix='seascape').reset_index(drop=True)
month_dummies = pd.get_dummies(moddat['month_abbr']).reset_index(drop=True)

#flag_dummies = pd.get_dummies(moddat['flag']).reset_index(drop=True)

# Combine dummy variables
#moddat = pd.concat([moddat, seascape_dummies, month_dummies, flag_dummies], axis=1)

moddat = pd.concat([moddat, month_dummies], axis=1)

# Get X, y
y = moddat[['year', 'illegal']].reset_index(drop=True)

#moddat = moddat.drop(columns = ['month', 'illegal', 'seascape_class', 'flag'])

moddat = moddat.drop(columns = ['month_abbr', 'illegal', 'seascape_class'])

moddat = moddat.drop(columns = ['month_abbr', 'illegal'])

moddat = moddat.drop(columns = 'illegal')

X = moddat
X.columns
X.head()
y.head()

# Cross-validate model
pred_score = []
roc_dat = pd.DataFrame()
for year in range(2013, 2017):
    
    X_train = X[X.year < year]
    y_train = y[y.year < year]

    X_test = X[X.year == year]
    y_test = y[y.year == year]

    print(f"Training Years: {X_train.year.unique()} - Test Year: {X_test.year.unique()}")

    X_train = X_train.drop(columns = ['year'])
    X_test = X_test.drop(columns = ['year'])

    y_train = y_train['illegal']
    y_test = y_test['illegal']
    
    #clf = LogisticRegression().fit(X_train, y_train)
    clf = RandomForestClassifier(n_estimators=100).fit(X_train, y_train)
    y_test_pred = clf.predict(X_test)
        
    score = accuracy_score(y_test, y_test_pred)
    score
    pred_score.append(score)

    probs = clf.predict_proba(X_test)
    preds = probs[:,1]
    fpr, tpr, threshold = metrics.roc_curve(y_test, preds)
    roc_auc = metrics.auc(fpr, tpr)

    outdat = pd.DataFrame({'pred_year': year, 'fpr': fpr, 'tpr': tpr, 'thresh': threshold})
    roc_dat = pd.concat([roc_dat, outdat])

print(pred_score)

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


clf.predict_proba(X_train)[:][1]



cross_validate(clf, X_train, y_train)
