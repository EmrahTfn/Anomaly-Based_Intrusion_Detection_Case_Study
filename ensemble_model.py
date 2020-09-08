# This script is exported from a jupyter notebook environment
# imports

from __future__ import print_function
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import sklearn.metrics as metrics
from sklearn.metrics import precision_score, recall_score, roc_auc_score, roc_curve, SCORERS
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, f1_score

# display options

np.set_printoptions(threshold=sys.maxsize)
pd.set_option('display.max_rows', 50000)
pd.set_option('display.max_columns', 500)
pd.set_option('display.float_format', lambda x: '%.6f' % x)
np.set_printoptions(suppress=True)

# numpy arrays are loaded

X = np.load("/***/***/***/***/***/***.npy")
y = np.load("/***/***/***/***/***/***.npy")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25)

# Ensemble ML model with Bagging Classifier using four different base classifiers
# For all classifiers, hyperparameters are obtained beforehand using hyperopt hyper-parameter optimization library
# 1a. SVM base classifier

model_svc = BaggingClassifier(base_estimator=SVC(C=474.0, kernel='rbf', gamma=1.6480790131462046, degree=4.250804641943586), bootstrap='True', bootstrap_features='False', max_features=0.876210911922131, max_samples=0.751498038137023, n_jobs=-1, warm_start='False')

# training the model and prediction

model_svc.fit(X_train, y_train.ravel())
rfc_predict = model_svc.predict(X_test)
# printing the performance metric results

print("=== Results for Support Vector Machine (SVM) Base Learner ===")
print("=== Confusion Matrix ===")
np = confusion_matrix(y_test, rfc_predict)
df_cm = pd.DataFrame(np, index=["0", "1"], columns=["0", "1"])
print(df_cm)
print('\n')
print("=== F1 Score ===")
print(f1_score(y_test, rfc_predict))
print('\n')
print("=== ROC AUC Score ===")
roc_auc = roc_auc_score(y_test, rfc_predict)
print(roc_auc)

# Illustration of ROC AUC score

probs = model_svc.predict_proba(X_test)
preds = probs[:,1]
fpr, tpr, threshold = metrics.roc_curve(y_test, preds)
plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()


# 1b. KNN base classifier

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25)

model_knn = BaggingClassifier(base_estimator=KNeighborsClassifier(algorithm='auto', leaf_size=26.0, n_neighbors=3, p=1.0, weights='distance'), bootstrap='False', bootstrap_features='False', n_jobs=-1, max_features=0.954309914114886, max_samples=0.6054250829008887, warm_start='True')

# training the model and prediction

model_knn.fit(X_train, y_train.ravel())
rfc_predict = model_knn.predict(X_test)

# printing the performance metric results

print("=== Results for KNN Base Learner ===")
print("=== Confusion Matrix ===")
np = confusion_matrix(y_test, rfc_predict)
df_cm = pd.DataFrame(np, index=["0", "1"], columns=["0","1"])
print(df_cm)
print('\n')
print("=== F1 Score ===")
print(f1_score(y_test, rfc_predict))
print('\n')
print("=== ROC AUC Score ===")
roc_auc = roc_auc_score(y_test, rfc_predict)
print(roc_auc)

# Illustration of ROC AUC score

probs = model_knn.predict_proba(X_test)
preds = probs[:,1]
fpr, tpr, threshold = metrics.roc_curve(y_test, preds)
plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()

# 1c. Naive Bayes base classifier

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25)

model_nb = BaggingClassifier(base_estimator=GaussianNB(var_smoothing = 3.1522149602830965e-05), bootstrap='False', bootstrap_features='True', warm_start='False',
max_features=0.795410586443666, max_samples=0.4281704573756816, n_jobs=-1)

# training the model and prediction

model_nb.fit(X_train, y_train.ravel())
rfc_predict = model_nb.predict(X_test)

# printing the performance metric results

print("=== Results for Naive Bayes Base Learner ===")
print("=== Confusion Matrix ===")
np = confusion_matrix(y_test, rfc_predict)
df_cm = pd.DataFrame(np, index=["0", "1"], columns=["0","1"])
print(df_cm)
print('\n')
print("=== F1 Score ===")
print(f1_score(y_test, rfc_predict))
print('\n')
print("=== ROC AUC Score ===")
roc_auc = roc_auc_score(y_test, rfc_predict)
print(roc_auc)

# Illustration of ROC AUC score

probs = model_nb.predict_proba(X_test)
preds = probs[:,1]
fpr, tpr, threshold = metrics.roc_curve(y_test, preds)
plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()

# 1d. Logistic Regression Base Classifier

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25)

model_lr = BaggingClassifier(base_estimator=LogisticRegression(C=190.0, max_iter=200.0, multi_class='auto', n_jobs=-1, penalty='none', solver='lbfgs', tol=0.06735330298902958, warm_start='False'), bootstrap='False', bootstrap_features='False', warm_start='False', max_features=0.8906458640258335, max_samples=0.06929383629914337, n_jobs=-1)

# training the model and prediction

model_lr.fit(X_train, y_train.ravel())
rfc_predict = model_lr.predict(X_test)

# printing the performance metric results

print("=== Results for Naive Bayes Base Learner ===")
print("=== Confusion Matrix ===")
np = confusion_matrix(y_test, rfc_predict)
df_cm = pd.DataFrame(np, index=["0", "1"], columns=["0","1"])
print(df_cm)
print('\n')
print("=== F1 Score ===")
print(f1_score(y_test, rfc_predict))
print('\n')
print("=== ROC AUC Score ===")
roc_auc = roc_auc_score(y_test, rfc_predict)
print(roc_auc)

# Illustration of ROC AUC score

probs = model_lr.predict_proba(X_test)
preds = probs[:, 1]
fpr, tpr, threshold = metrics.roc_curve(y_test, preds)
print(roc_auc)
plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()

# Note that * symbols were placed where required to mask personal and institutional information.
