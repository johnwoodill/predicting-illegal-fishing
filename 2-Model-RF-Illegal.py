import pandas as pd
import numpy as np
from sklearn.metrics import average_precision_score, auc, precision_recall_curve, f1_score
from sklearn.metrics import confusion_matrix, balanced_accuracy_score
from sklearn.ensemble import RandomForestClassifier
import calendar

# Load full 8Day data from data-step
dat = pd.read_feather('data/full_gfw_10d_effort_model_data_8DAY_2012-01-01_2016-12-26.feather')

# If illegally operating inside EEZ (!= ARG)
dat = dat.assign(illegal = np.where(((dat['eez'] == True) &
                                     (dat['fishing_hours'] > 0)
                                     & (dat['flag'] != 'ARG') ), 1, 0))

# Buffer by 2km (sensitivity analysis)
dat = dat.assign(illegal_2km = np.where(((dat['illegal'] == True) &
                                        (dat['distance_to_eez_km'] >= 2) ), 1, 0))

# Check proportion of illegal activity
sum(dat.illegal)/len(dat)
sum(dat.illegal_2km)/len(dat)

# Get year month
dat = dat.assign(year = pd.DatetimeIndex(dat['date']).year,
                 month = pd.DatetimeIndex(dat['date']).month)

# Convert month number to name
dat = dat.assign(month_abbr = dat.apply(
    lambda x: calendar.month_abbr[x['month']], 1))

# Drop measures of distance
moddat = dat[['illegal', 'year', 'month_abbr',
             'seascape_class', 'sst', 'chlor_a', 'lon1', 'lat1', 'eez']].dropna()

# Dummy variables for seascape and dummies
seascape_dummies = pd.get_dummies(moddat['seascape_class'], prefix='seascape')
month_dummies = pd.get_dummies(moddat['month_abbr'])

# Concat dummy variables for model
moddat = pd.concat([moddat, seascape_dummies, month_dummies], axis=1)

# Drop na
moddat = moddat.dropna()

# Get X, y
y = moddat[['year', 'illegal']]

# Drop dummy variables and prediction
moddat = moddat.drop(columns = ['month_abbr', 'illegal', 'seascape_class'])

# X data for model
X = moddat

# Cross-validate model
sdat = pd.DataFrame()     # Dataframe for binding results
feadat = pd.DataFrame()   # Dataframe for bind feature impt.
for year in range(2012, 2017):
    # Get training data
    X_train, y_train = X[X.year != year], y[y.year != year]
    
    # Get test data
    X_test, y_test = X[X.year == year], y[y.year == year]
    
    print(f"Training Years: {X_train.year.unique()} - Test Year: {X_test.year.unique()}")

    # Drop year
    X_train = X_train.drop(columns = ['year'])
    X_test = X_test.drop(columns = ['year'])
    
    # Set binary dep variable 
    y_train = y_train['illegal']
    y_test = y_test['illegal']
   
   # Base RF Model (faster)
    clf = RandomForestClassifier(n_estimators = 100, random_state=123).fit(X_train, y_train)

    # Hyper-parameter tuned RF model
    # clf = RandomForestClassifier(n_estimators=1600, 
    #                              min_samples_split=2,
    #                              max_depth=40,
    #                              bootstrap=True).fit(X_train, y_train)
    
    # Get predicted probabilities for sensitivity analysis
    pred_proba = clf.predict_proba(X_test)[:, 1]
    
    # Calc Precision-recall across thresholds
    # Precision = tp / ( tp + fp )
    # Recall = tn / ( tn + fp )
    precision, recall, thresholds = precision_recall_curve(y_test, pred_proba)
    
    # Get test predictions
    y_pred = clf.predict(X_test)
    
    # F1 score = 2 * precision * recall / precision + recall
    f1 = f1_score(y_test, y_pred)
    
    # calculate precision-recall AUC
    auc_m = auc(recall, precision)

    # Calc specificity = TN/(TN+FP)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    specificity = tn / (tn+fp)
        
    # Get Feature importance
    fea_import = pd.DataFrame({'variable': X_train.columns , 'importance': clf.feature_importances_, 'year': year})
    feadat = pd.concat([feadat, fea_import])
    
    # Balanced accuracy score metric
    bas = balanced_accuracy_score(y_test, y_pred)
    bas_t = balanced_accuracy_score(y_test, y_pred, adjusted=True)

    # Calculate average precision score
    ap = average_precision_score(y_test, pred_proba)
    
    print('f1=%.3f auc=%.3f ap=%.3f' % (f1, auc_m, ap))
    
    # Dataframe with model validation metrics
    ddat = pd.DataFrame({'year': year, 
                         'prec': precision, 
                         'recall': recall, 
                         'f1': f1, 
                         'auc': auc_m, 
                         'ap':ap, 
                         'spec': specificity, 
                         'bas': bas, 
                         'bas_t': bas_t})
    
    # Bind with all results
    sdat = pd.concat([sdat, ddat])


# Save precision-recall data
sdat = sdat.reset_index(drop=True)
sdat.to_feather('data/illegal_cross_val_dat.feather')

# Save Feature importance
fea_import = pd.DataFrame({'variable': X_train.columns , 'importance': clf.feature_importances_})
fea_import = fea_import.sort_values('importance', ascending=False)
fea_import = fea_import.reset_index(drop=True)
fea_import.to_feather('data/feature_importance_rf_illegal.feather')


