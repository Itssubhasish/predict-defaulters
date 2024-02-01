# !python.exe -m pip install --upgrade pip 
# !pip install sweetviz 

# !pip install pyxlsb                                              ### As the dataset format is .xlsb (excel binary file) we will require pyxlsb package to read the file
# !pip install h2o                                                 ### AutoL library

import pandas as pd, sweetviz as sv, numpy as np, h2o, warnings, matplotlib.pyplot as plt, seaborn as sns
from pyxlsb import open_workbook
from scipy import stats
from h2o.automl import H2OAutoML
from sklearn.metrics import confusion_matrix, precision_score, recall_score, accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.feature_selection import SelectKBest, chi2

pd.set_option('display.max_columns', None)
warnings.filterwarnings('always')
warnings.filterwarnings(action='ignore')



def calculate_iv_woe(data, target_col, feature_col):
    crosstab = pd.crosstab(data[feature_col], data[target_col])
    # Add a small epsilon value to avoid divide by zero
    epsilon = 1e-9
    crosstab['WOE'] = np.log(((crosstab[1] + epsilon) / (crosstab[1].sum() + epsilon)) / ((crosstab[0] + epsilon) / (crosstab[0].sum() + epsilon)))
    crosstab['IV'] = crosstab['WOE'] * (stats.entropy((crosstab[1] + epsilon) / (crosstab[1].sum() + epsilon), (crosstab[0] + epsilon) / (crosstab[0].sum() + epsilon)))
    return crosstab['IV'].sum(), crosstab


def create_metrics(y_test, y_pred):
    conf_matrix = confusion_matrix(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average=None)
    recall = recall_score(y_test, y_pred, average=None)
    accuracy = accuracy_score(y_test, y_pred)
    specificity = conf_matrix[0, 0] / (conf_matrix[0, 0] + conf_matrix[0, 1])

    tp = conf_matrix[1, 1]
    tn = conf_matrix[0, 0]
    fp = conf_matrix[0, 1]
    fn = conf_matrix[1, 0]

    # Calculate row and column sums
    row_sums = [tp + fn, fp + tn]
    col_sums = [tp + fp, fn + tn]
    total = tp + tn + fp + fn

    # Create a DataFrame with a MultiIndex
    index = pd.MultiIndex.from_tuples([('Actual', 'Positive'), ('Actual', 'Negative'), ('Total', '')])
    columns = pd.MultiIndex.from_tuples([('Predicted', 'Positive'), ('Predicted', 'Negative'), ('Total', '')])

    data = [
        [tp, fn, row_sums[0]],
        [fp, tn, row_sums[1]],
        [col_sums[0], col_sums[1], total]
    ]

    confusion_matrix_df = pd.DataFrame(data, index=index, columns=columns)



    data = {
        'Class': [f'Class {i}' for i in range(conf_matrix.shape[0])],
        'Precision': precision,
        'Recall': recall,
        'Correctness': [accuracy] * len(precision),
        'Specificity': [specificity] + [None] * (conf_matrix.shape[0] - 1),
    }

    metrics_df = pd.DataFrame(data)
    display(confusion_matrix_df)
    display(metrics_df)


param_grid = {
    'penalty': ['l1', 'l2', 'elasticnet', None],
    # 'dual': [True, False],
    # 'tol': [1e-4, 1e-3, 1e-2],
    # 'C': [0.1, 1.0, 10.0],
    # 'fit_intercept': [True, False],
    # 'intercept_scaling': [1.0, 2.0],
    'class_weight': ['balanced'],
    # 'solver': ['lbfgs', 'newton-cg', 'sag', 'saga'],
    'max_iter': np.arange(500, 1200, 200),
    # 'multi_class': ['auto', 'ovr', 'multinomial'],
    # 'verbose': [0, 1, 2],
    # 'warm_start': [True, False],
    'n_jobs': [1, -1]
    # 'l1_ratio': [0.0, 0.5, 1.0]
}

# Define the parameter grid for the grid search
param_grid_GBM = {
    'loss': ['deviance', 'exponential'],
    'learning_rate': [0.1, 0.01, 0.001],
    'n_estimators': [100, 200, 300],
    'subsample': [0.8, 1.0],
    'criterion': ['friedman_mse', 'squared_error'],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2],
    'max_depth': [3, 4, 5, 6, 7],
    'min_impurity_decrease': [0.0, 0.1],
    'init': [None, 'zero'],
    'random_state': [None, 42],
    'max_features': [None, 'sqrt', 'log2']
    # 'verbose': [0, 1],
    # 'max_leaf_nodes': [None, 10, 20],
    # 'warm_start': [True, False],
    # 'validation_fraction': [0.1, 0.2],
    # 'n_iter_no_change': [None, 5],
    # 'tol': [1e-4, 1e-5],
    # 'ccp_alpha': [0.0, 0.1],
}