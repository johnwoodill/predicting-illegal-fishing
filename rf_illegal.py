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

# DAILY data
#dat = pd.read_feather('~/Projects/predicting-illegal-fishing/data/full_gfw_10d_illegal_model_data_DAILY_2012-01-01_2016-12-31.feather')

# Subset drifting longlines
#dat = dat[dat.geartype == 'drifting_longlines']

# Keep only vessel with fishing hours
dat = dat[dat['fishing_hours'] > 0 ]

#dat = dat.sort_values('date')

# Keep only Chinese vessels
dat = dat[(dat['flag'] == 'CHN') | (dat['flag'] == 'ARG')]

# If illegally operating inside EEZ (!= ARG)
dat.loc[:, 'illegal'] = np.where(((dat['eez'] == True) & (dat['flag'] != 'ARG') & (dat['fishing_hours'] > 0)), 1, 0)

# Illegal if within 10km
dat.loc[:, 'illegal_50km'] = np.where(((dat['eez'] == True) & (dat['distance_to_eez_km'] <= 50) & (dat['flag'] != 'ARG')), 1, 0)

# Illegal if within 20km
dat.loc[:, 'illegal_150km'] = np.where(((dat['eez'] == True) & (dat['distance_to_eez_km'] <= 150) & (dat['flag'] != 'ARG')), 1, 0)

# Convert true/false eez to 0/1
dat.loc[:, 'illegal'] = dat.illegal.astype('uint8')
dat.loc[:, 'illegal_50km'] = dat.illegal_50km.astype('uint8')
dat.loc[:, 'illegal_150km'] = dat.illegal_150km.astype('uint8')

sum(dat.illegal)/len(dat)
sum(dat.illegal_50km)/len(dat)
sum(dat.illegal_150km)/len(dat)

# Calc distance from eez for illegal vessels
dat.loc[:, 'illegal_distance_to_eez_km'] = np.where((dat['illegal'] == 1), dat['distance_to_eez_km'], 99999999)

dat.loc[:, 'illegal_distance_to_eez_50km'] = np.where((dat['illegal_50km'] == 1), dat['distance_to_eez_km'], 99999999)

dat.loc[:, 'illegal_distance_to_eez_150km'] = np.where((dat['illegal_150km'] == 1), dat['distance_to_eez_km'], 99999999)

dat.loc[:, 'year'] = pd.DatetimeIndex(dat['date']).year
dat.loc[:, 'month'] = pd.DatetimeIndex(dat['date']).month

# Convert month number to name
dat.loc[:, 'month_abbr'] = dat.apply(lambda x: calendar.month_abbr[x['month']], 1)

# Linear model
# Get data frame of variables and dummy seascapes
moddat = dat[['illegal', 'year', 'month_abbr', 'seascape_class', 'sst', 'sst_grad', 'sst4', 'sst4_grad', 'chlor_a', 'lon1', 'lat1', 'depth_m', 'coast_dist_km', 'port_dist_km', 'eez', 'distance_to_eez_km']].dropna().reset_index(drop=True)

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
pred_score = []
roc_dat = pd.DataFrame()
sdat = pd.DataFrame()
for year in range(2013, 2017):
    
    X_train = X[X.year != year]
    y_train = y[y.year != year]

    X_test = X[X.year == year]
    y_test = y[y.year == year]

    print(f"Training Years: {X_train.year.unique()} - Test Year: {X_test.year.unique()}")

    X_train = X_train.drop(columns = ['year'])
    X_test = X_test.drop(columns = ['year'])

    y_train = y_train['illegal']
    y_test = y_test['illegal']
    
    #clf = LogisticRegression().fit(X_train, y_train)
    clf = RandomForestClassifier(n_estimators=100).fit(X_train, y_train)

    # Get out-of-sample predictions
    ydat = pd.DataFrame({'illegal': y_test, 'pred_illegal': clf.predict(X_test)})

    # Get true pos
    ydat1 = ydat[(ydat.illegal == 1)]
    true = sum(ydat1.pred_illegal) / sum(ydat1.illegal)
    
    # Get false pos
    ydat2 = ydat[(ydat.pred_illegal == 1)]
    false_true = 0 if sum(ydat2.pred_illegal) == 0 else sum(ydat2.illegal) / sum(ydat2.pred_illegal)

    perc_true = true
    perc_false_true = sum(ydat2.pred_illegal - ydat2.illegal)/len(ydat2)
         
    main_perc_true = perc_true
        
    print(f"% True pos: {round(perc_true, 2)}   -   % False pos: {round(perc_false_true, 2)}")

    # ROC Curve
    for i in np.linspace(0, 1, 11):
        threshold = i
        predicted_proba = clf.predict_proba(X_test)
        predicted = (predicted_proba [:, 1] >= threshold).astype('int')
        ydat = pd.DataFrame({'illegal': y_test, 'pred_illegal': predicted})
   
        ydat1 = ydat[(ydat.illegal == 1)]
        true = sum(ydat1.pred_illegal) / sum(ydat1.illegal)
        
        ydat2 = ydat[(ydat.pred_illegal == 1)]
        false_true = 0 if sum(ydat2.pred_illegal) == 0 else sum(ydat2.illegal) / sum(ydat2.pred_illegal)

        perc_true = true
        perc_false_true = sum(ydat2.pred_illegal - ydat2.illegal)/len(ydat2)
            
        print(f"% True pos: {round(perc_true, 2)}   -   % False pos: {round(perc_false_true, 2)}")
        ldat = pd.DataFrame({'year': [year], 'threshold': [i], 'tpr': [perc_true], 'fpr': [perc_false_true], 'Acccuracy': f"{year} - {round(main_perc_true, 2)}%"})
        sdat = pd.concat([sdat, ldat])

\
print(sdat)

sns.lineplot('fpr', 'tpr', hue = 'Acccuracy', data = sdat)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.title('Predicting Chinese Fishing Vessels Illegal Activity')

plt.savefig('figures/roc_curve.pdf')
plt.show()

# Feature importance
fea_import = pd.DataFrame({'variable': X_train.columns , 'importance': clf.feature_importances_})
fea_import = fea_import.sort_values('importance', ascending=False)
#fea_import.to_feather('data/feature_importance_rf_illegal.feather')
sns.barplot('importance', 'variable', data = fea_import)
plt.show()

