# This script is exported from a jupyter notebook environment
# imports

from __future__ import print_function
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn.feature_selection as skfs
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import RandomForestClassifier
import sklearn.preprocessing as skpr
from sklearn.feature_selection import f_classif
from genetic_selection import GeneticSelectionCV

# display options

np.set_printoptions(threshold=sys.maxsize)
pd.set_option('display.max_rows', 50000)
pd.set_option('display.max_columns', 500)
pd.set_option('display.float_format', lambda x: '%.6f' % x)

# read csv files

X = pd.read_csv('/***/***/***/***/***/***.csv', index_col=0, low_memory=False)
y = pd.read_csv('/***/***/***/***/***/***.csv', index_col=0, low_memory=False)
df_whole = pd.read_csv('/***/***/***/***/***/***.csv', index_col=0, low_memory=False)
X = X.drop(["SrcAddr", "DstAddr"], axis=1)
df_whole = df_whole.drop(["SrcAddr", "DstAddr"], axis=1)

# Feature selection with filter method starts
# 1.Features with low varience (below 0.005) are eliminated

sel = skfs.VarianceThreshold(threshold=(.995 * (1 - .995)))
sel.fit(df_whole)
df_wo_low_variance = df_whole[df_whole.columns[sel.get_support(indices=True)]]
df_wo_low_variance.to_csv(path_or_buf="/***/***/***/***/***/***.csv")

# 2a. Filter method feature selection with correlation > 0.3

target = y["Class"]
df_wo_low_variance = df_wo_low_variance.astype(dtype="float64")
importances = df_wo_low_variance.apply(lambda x: x.corr(target))
corr_importances = importances[abs(importances) > 0.3]
indices = np.argsort(corr_importances)
print(corr_importances[:])
corr_importances.index

# Highly correlated features illustrated 

features_corr = corr_importances.index
plt.title("features_corr")
plt.barh(range(len(indices)), importances[indices], color="g", align="center")
plt.yticks(range(len(indices)), [features_corr[i] for i in indices])
plt.xlabel("Relative Importance")
plt.show()

# 2b. Filter method feature selection with ensemble learning extra trees

model = ExtraTreesClassifier(n_estimators=10000, n_jobs=-1)
model.fit(df_wo_low_variance, y)
print(model.feature_importances_)

df_wo_low_variance_columns = df_wo_low_variance.columns
df_model_feature_importances = pd.DataFrame(model.feature_importances_, index = df_wo_low_variance_columns, columns=["importance"])
df_decision_forest_selected_columns = df_model_feature_importances[df_model_feature_importances["importance"].astype(float) > 0.005]
print(df_decision_forest_selected_columns.index)

# 2c. Filter method feature selection with ANOVA f-value method

fvalue_selector = skfs.SelectKBest(score_func=f_classif, k=25)
fvalue_selector.fit(df_wo_low_variance, target)
print(fvalue_selector.scores_)

df_wo_low_variance_columns = df_wo_low_variance.columns
df_model_feature_importances = pd.DataFrame(fvalue_selector.scores_,  index=df_wo_low_variance_columns, columns=["score"])
df_f_value_selected_columns = df_model_feature_importances[df_model_feature_importances ["score"].astype(float) > 10000]
print(df_f_value_selected_columns.index)

# 2d. Filter method feature selection with chi2 method

min_max_scaler = preprocessing.MinMaxScaler(feature_range=(0, 100))
np_wo_low_variance_minmax = min_max_scaler.fit_transform(df_wo_low_variance)
df_wo_low_variance_minmax = pd.DataFrame(np_wo_low_variance_minmax, columns=df_wo_low_variance.columns)

chi2_selector = skfs.SelectKBest(skfs.chi2, k=25)
chi2_selector.fit(df_wo_low_variance_minmax, target)
print(chi2_selector.scores_)

df_wo_low_variance_columns = df_wo_low_variance.columns
df_model_feature_scores = pd.DataFrame(chi2_selector.scores_, index=df_wo_low_variance_columns, columns=["score"])
df_chi2_selector_selected_columns = df_model_feature_scores[df_model_feature_importances["score"].astype(float) > 10000]
print(df_chi2_selector_selected_columns.index)

# Results from four different filter methods

print(features_corr)
print(type(features_corr))
print("******************************************************************")
# print(df_decision_forest_selected_columns)
df_decision_forest_selected_columns_list = df_decision_forest_selected_columns.index.values.tolist()
print(df_decision_forest_selected_columns_list)
print(type(df_decision_forest_selected_columns_list))
print("******************************************************************")
df_f_value_selected_columns_list = df_f_value_selected_columns.index.values.tolist()
print(df_f_value_selected_columns_list)
print(type(df_f_value_selected_columns_list))
print("******************************************************************")
df_chi2_selector_selected_columns_list = df_chi2_selector_selected_columns.index.values.tolist()
print(df_chi2_selector_selected_columns_list)
print(type(df_chi2_selector_selected_columns_list))

# Combination of four filter method feature selection result sets

combined_selected_features = set(features_corr + df_decision_forest_selected_columns_list + df_f_value_selected_columns_list + df_chi2_selector_selected_columns_list)
print(combined_selected_features)

# 3. Elimination of highly inter-correlated (higher than 0.75) features 
# Filter method final feature set is ready.

X = df_wo_low_variance[combined_selected_features]

def correlation(dataset, threshold):
    col_corr = set()
    corr_matrix = dataset.corr()
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if abs(corr_matrix.iloc[i, j]) > threshold:
                colName = corr_matrix.columns[i]
                col_corr.add(colName)
    return col_corr
col = correlation(X, 0.75)

print('Correlated columns:',col)
print(len(col))
filter_method_selected_features = combined_selected_features - col
print("********************************************************************")
print('Remaining Columns:', filter_method_selected_features)
print(len(filter_method_selected_features))

df_selected_features_filter = df_wo_low_variance[filter_method_selected_features]
df_selected_features_filter.to_csv(path_or_buf="/***/***/***/***/***/df_selected_features_filter.csv", index=False)
df_selected_features_filter = df_selected_features_filter.to_numpy()

scaler = skpr.MinMaxScaler(feature_range=(0,1), copy=False)
scaler.fit(df_selected_features_filter)
scaler.transform(df_selected_features_filter)
df_selected_features_filter = pd.DataFrame(df_selected_features_filter, columns=filter_method_selected_features)

# 4. Final feature selection with the wrapper method of genetic algorithm

X = df_selected_features_filter.copy()

estimator = RandomForestClassifier(n_estimators=1000, n_jobs=1)
selector = GeneticSelectionCV(estimator,
                                  cv=5,
                                  verbose=1,
                                  scoring="accuracy",
                                  max_features=18,
                                  n_population=300,
                                  crossover_proba=0.5,
                                  mutation_proba=0.2,
                                  n_generations=50,
                                  crossover_independent_proba=0.1,
                                  mutation_independent_proba=0.05,
                                  tournament_size=3,
                                  n_gen_no_change=10,
                                  caching=True,
                                  n_jobs=-1)
selector = selector.fit(X, y.values.ravel())

print(selector.support_)
print(X.columns)

X.drop(X.columns[np.where(selector.support_ == False)[0]], axis=1, inplace=True)

dset = pd.DataFrame()
dset['attr'] = X.columns
dset['importance'] = selector.estimator_.feature_importances_
dset = dset.sort_values(by='importance', ascending=False)

plt.figure(figsize=(16, 14))
plt.barh(y=dset['attr'], width=dset['importance'], color='#1976D2')
plt.title('sklearn.GeneticSelectionCV - Feature Importances', fontsize=20, fontweight='bold', pad=20)
plt.xlabel('Importance', fontsize=14, labelpad=20)
plt.show()

sklearn_GeneticSelector_selected_columns = X.columns
print(sklearn_GeneticSelector_selected_columns, len(sklearn_GeneticSelector_selected_columns))


# Note that * symbols were placed where required to mask personal and institutional information.
