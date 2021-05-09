import pandas as pd
import numpy as np
import calendar
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (average_precision_score, auc,
                             precision_recall_curve, f1_score,
                             confusion_matrix, balanced_accuracy_score)


# Load full 8Day data from data-step
dat = pd.read_feather((f"data/full_gfw_10d_effort_model_data_8DAY_"
                       f"2012-01-01_2016-12-26.feather"))

# If operating inside EEZ, with positive fishing hours, and != ARG
dat = dat.assign(illegal=np.where(((dat['eez'] == True) &
                                   (dat['fishing_hours'] > 0) &
                                   (dat['flag'] != 'ARG')), 1, 0))

# Buffer by 2km (sensitivity analysis)
dat = dat.assign(illegal_2km=np.where(((dat['illegal'] == 1) &
                                       (dat['distance_to_eez_km'] >= 2)),
                                      1, 0))

# Check proportion of illegal activity
sum(dat.illegal)/len(dat)
sum(dat.illegal_2km)/len(dat)

# Get year month
dat = dat.assign(year=pd.DatetimeIndex(dat['date']).year,
                 month=pd.DatetimeIndex(dat['date']).month)

# Convert month number to name using calendar lib
dat = dat.assign(month_abbr=dat.apply(
    lambda x: calendar.month_abbr[x['month']], 1))

# Keep variables (baseline model)
moddat = dat[['illegal', 'year', 'month_abbr', 'seascape_class',
              'sst', 'chlor_a', 'lon1', 'lat1', 'eez']]

# Dummy variables for seascape and dummies
seascape_dummies = pd.get_dummies(moddat['seascape_class'], prefix='seascape')
month_dummies = pd.get_dummies(moddat['month_abbr'])

# Concat dummy variables for model
moddat = pd.concat([moddat, seascape_dummies, month_dummies], axis=1)

# Drop na
moddat = moddat.dropna()

# Get y, X
y = moddat[['year', 'illegal']]

# Drop cat. variables and dep variable
X = moddat.drop(columns=['month_abbr', 'illegal', 'seascape_class'])

# Cross-validate model by year
sdat = pd.DataFrame()        # Dataframe for binding results
feadat = pd.DataFrame()      # Dataframe for binding feature impt.

for year in range(2012, 2017):
    # Get training data
    X_train, y_train = X[X['year'] != year], y[y['year'] != year]

    # Get test data
    X_test, y_test = X[X['year'] == year], y[y['year'] == year]

    print(f"Training Years: {X_train['year'].unique()} - "
          f"Test Year: {X_test['year'].unique()}")

    # Drop year
    X_train = X_train.drop(columns=['year'])
    X_test = X_test.drop(columns=['year'])

    # Set binary dep variable
    y_train = y_train['illegal']
    y_test = y_test['illegal']

    # Base RF Model (faster)
    clf = RandomForestClassifier(random_state=123).fit(X_train, y_train)

    # Hyper-parameter tuned RF model
    # clf = RandomForestClassifier(n_estimators=1600,
    #                              min_samples_split=2,
    #                              max_depth=40,
    #                              bootstrap=True).fit(X_train, y_train)

    # Get test predictions
    y_pred = clf.predict(X_test)

    # Get predicted probabilities
    pred_proba = clf.predict_proba(X_test)[:, 1]

    # Calc Precision-recall across thresholds
    # Precision = tp / ( tp + fp )
    # Recall = tn / ( tn + fp )
    precision, recall, thresholds = precision_recall_curve(y_test, pred_proba)

    # F1 score = 2 * precision * recall / precision + recall
    f1 = f1_score(y_test, y_pred)

    # calculate precision-recall AUC
    auc_m = auc(recall, precision)

    # Calculate average precision score
    ap_m = average_precision_score(y_test, pred_proba)

    # Calc specificity = tn / (tn + fp) 
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    specificity = tn / (tn + fp)

    # Balanced accuracy score metric
    bas = balanced_accuracy_score(y_test, y_pred)
    bas_t = balanced_accuracy_score(y_test, y_pred, adjusted=True)

    print('f1=%.3f   auc=%.3f   ap=%.3f' % (f1, auc_m, ap_m))

    # Dataframe with model validation metrics
    ddat = pd.DataFrame({'year': year,
                         'prec': precision,
                         'recall': recall,
                         'f1': f1,
                         'auc': auc_m,
                         'ap': ap_m,
                         'spec': specificity,
                         'bas': bas,
                         'bas_t': bas_t})

    # Bind with all results
    sdat = pd.concat([sdat, ddat])

    # Get Feature importance
    fea_import = pd.DataFrame({'year': year,
                               'variable': X_train.columns,
                               'importance': clf.feature_importances_})

    # Bind feature importance
    feadat = pd.concat([feadat, fea_import])


# Results
# Training Years: [2016 2013 2015 2014] - Test Year: [2012]
# f1=0.687 auc=0.835 ap=0.834

# Training Years: [2016 2015 2012 2014] - Test Year: [2013]
# f1=0.865 auc=0.903 ap=0.900

# Training Years: [2016 2013 2015 2012] - Test Year: [2014]
# f1=0.736 auc=0.916 ap=0.913

# Training Years: [2016 2013 2012 2014] - Test Year: [2015]
# f1=0.872 auc=0.936 ap=0.934

# Training Years: [2013 2015 2012 2014] - Test Year: [2016]
# f1=0.830 auc=0.896 ap=0.893


# Save precision-recall data
sdat = sdat.reset_index(drop=True)
sdat.to_feather('data/illegal_cross_val_dat.feather')

# Save Feature importance
feadat = feadat.sort_values('importance', ascending=False)
feadat = feadat.reset_index(drop=True)
feadat.to_feather('data/feature_importance_rf_illegal.feather')