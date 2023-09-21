import matplotlib.pyplot as plt
import joblib
import pandas as pd
import os
from pathlib import Path
import numpy as np
from sklearn import preprocessing
from sklearn import svm
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score, RocCurveDisplay
from sklearn.model_selection import GridSearchCV
import seaborn as sns
sns.reset_orig()
plt.rcParams['font.family'] = 'serif'
plt.rcParams['figure.figsize']=(20,12)

verbose = 1
data_folder = 'Demo Data'
performance_folder = 'Demo Performance'
random_state = 42

param_grid = {
    'C': [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3],
    'gamma': [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3, 'auto', 'scale']
}

summary_df = pd.DataFrame()
if verbose >= 1:
    print('Reading Data')
train_valid_df = pd.read_excel(os.path.join(data_folder, 'train_valid_user.xlsx'))
test_valid_df = pd.read_excel(os.path.join(data_folder, 'test_valid_user.xlsx'))
train_imposter_df = pd.read_excel(os.path.join(data_folder, 'train_imposter.xlsx'))
test_imposter_df = pd.read_excel(os.path.join(data_folder, 'test_imposter.xlsx'))
selected_feature_columns = train_valid_df.columns[train_valid_df.columns.get_loc(0):
                                                  train_valid_df.columns.get_loc('Class')].tolist()

X_train_valid = train_valid_df[selected_feature_columns].values
y_train_valid = train_valid_df['Class'].values
X_test_valid = test_valid_df[selected_feature_columns].values
y_test_valid = test_valid_df['Class'].values
X_train_imposter = train_imposter_df[selected_feature_columns].values
y_train_imposter = train_imposter_df['Class'].values
X_test_imposter = test_imposter_df[selected_feature_columns].values
y_test_imposter = test_imposter_df['Class'].values

# Normalization
scaler = preprocessing.StandardScaler()
X_train_valid_n = scaler.fit_transform(X_train_valid)
X_test_valid_n = scaler.transform(X_test_valid)
X_train_imposter_n = scaler.transform(X_train_imposter)
X_test_imposter_n = scaler.transform(X_test_imposter)

# Eigenheart
pca = PCA(n_components=3, random_state=random_state)
X_train_valid_p = pca.fit_transform(X_train_valid_n)
X_test_valid_p = pca.transform(X_test_valid_n)
X_train_imposter_p = pca.transform(X_train_imposter_n)
X_test_imposter_p = pca.transform(X_test_imposter_n)

X_train = np.concatenate((X_train_valid_p, X_train_imposter_p), axis=0)
y_train = np.concatenate((y_train_valid, y_train_imposter), axis=0)
X_test = np.concatenate((X_test_valid_p, X_test_imposter_p), axis=0)
y_test = np.concatenate((y_test_valid, y_test_imposter), axis=0)

# Hyperparameter Optimization
if verbose >= 1:
    print('Hyperparameter Optimizing')

svc = svm.SVC(kernel='rbf', probability=True, random_state=random_state)
grid = GridSearchCV(estimator=svc, param_grid=param_grid, cv=5, verbose=verbose)
grid.fit(X_train, y_train)
best_params = grid.best_params_
best_clf = grid.best_estimator_
best_clf.fit(X_train, y_train)

cv_result = grid.cv_results_
best_index = np.where(cv_result['rank_test_score'] == 1)
mean_validate_score = cv_result['mean_test_score'][best_index][0]
std_validate_score = cv_result['std_test_score'][best_index][0]
if verbose > 1:
    print(f'Best parameters found: {grid.best_params_}')

# Retraining
if verbose >= 1:
    print(f'Retrain with all train data with best parameters')
best_clf.fit(X_train, y_train)
y_train_pred = best_clf.predict(X_train)
train_acc = accuracy_score(y_train, y_train_pred)

# Testing
if verbose >= 1:
    print('Testing')
y_test_pred = best_clf.predict(X_test)
y_test_pred_prob = best_clf.predict_proba(X_test)
test_acc = accuracy_score(y_test, y_test_pred)

confusion_matrix_df = pd.DataFrame(confusion_matrix(y_test, y_test_pred))
confusion_matrix_df.to_excel(os.path.join(performance_folder, f'confusion_matrix.xlsx'))

test_fp = confusion_matrix_df[0][1]
test_tn = confusion_matrix_df[0][0]
test_fpr = test_fp / (test_fp + test_tn)

test_fn = confusion_matrix_df[1][0]
test_tp = confusion_matrix_df[1][1]
test_fnr = test_fn / (test_fn + test_tp) 

test_auc_roc = roc_auc_score(y_test, y_test_pred_prob[:, 1])

if verbose >= 1:
    print(f'train accuracy: {train_acc:.4f}, validation accuracy: {mean_validate_score:.4f}, test accuracy: {test_acc:.4f}')
    
summary_df = pd.concat([
    summary_df,
    pd.DataFrame.from_records([{
        'Train ACC': train_acc,
        'Mean Validation ACC': mean_validate_score, 
        'Std Validation ACC': std_validate_score, 
        'Test ACC': test_acc, 
        'Test AUC-ROC': test_auc_roc, 
        'Test FPR': test_fpr,
        'Test FNR': test_fnr
    }])
])
joblib.dump(best_clf, os.path.join(f'trained_model.sav'))
best_params_df = pd.DataFrame([grid.best_params_])
best_params_df.to_excel(os.path.join(performance_folder, f'best_params.xlsx'), index=False)

summary_df.to_excel(os.path.join(performance_folder, 'summary.xlsx'), index=False)