# This script is exported from a jupyter notebook environment
# imports

from __future__ import print_function
import sys
import random
import pandas as pd
import numpy as np
from sklearn import preprocessing as skpr

# display options

np.set_printoptions(threshold=sys.maxsize)
pd.set_option('display.max_rows', 50000)
pd.set_option('display.max_columns', 500)
pd.set_option('display.float_format', lambda x: '%.6f' % x)
np.set_printoptions(suppress=True)

# load .csv files and type conversion

X = pd.read_csv('/***/***/***/***/***/***.csv', low_memory=False, index_col=0)
y = pd.read_csv('/***/***/***/***/***/***.csv', low_memory=False, index_col=0)

X_np = np.zeros([100000, 1024], dtype=float)
y_np = np.zeros([100000, 1], dtype=int)

X_list = X_np.tolist()
y_list = y_np.tolist()

print(type(X_np), type(y_np))
print(X_np.shape, y_np.shape)
X.drop(["SrcAddr", "DstAddr"], axis=1, inplace=True)

X = X.to_numpy(dtype=float)
y = y.to_numpy(dtype=float)

# Scaling between 0-1 with MinMaxScaler

scaler = skpr.MinMaxScaler(feature_range=(0,1), copy=False)
scaler.fit(X)
scaler.transform(X)

# Dataframe row vector with 1x139 size converted into 1x1024 vector
# by replicating features until 973rd element (7x139). The rest (until 1024) were given
# random numbers to avoid bias.

sample_size = len(X_list)
feature_size = len(X_list[0])

sample_count = 0
for sample in range(sample_size):
    feature_count = 0
    for feature in range(feature_size):
        if feature_count < 973:
            X_list[sample_count][feature_count] = X[sample_count, (feature_count%139)]
        else:
            feature = float(random.uniform(0, 1))
        feature_count = feature_count + 1
    sample_count = sample_count + 1
print(sample_count, feature_count)

X_np = np.asarray(X_list, dtype=float)

# Dataframe row vector with 1x1024 size converted into 32x32 matrix

X_np = np.reshape(X_np, (100000, 32, 32))

np.save('/***/***/***/***/***/***.npy', X_np)
np.save('/***/***/***/***/***/***.npy', y)


# Note that * symbols were placed where required to mask personal and institutional information.
