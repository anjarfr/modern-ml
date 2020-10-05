# Processing
import pandas as pd
import numpy as np
from sklearn import model_selection
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA, TruncatedSVD, FastICA
from sklearn.model_selection import KFold
from sklearn.random_projection import GaussianRandomProjection, SparseRandomProjection
from sklearn.preprocessing import normalize
from sklearn.metrics import roc_auc_score
import gc

# Plotting
import matplotlib.pyplot as plt

# Categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

# Models
import xgboost as xgb
from catboost import CatBoostRegressor

train = pd.read_csv('challenge2_train.csv')
test = pd.read_csv('challenge2_test.csv')

train.info()
test.info()


X_train = train.drop(["id", "target"], axis=1)
Y_train = train["target"].values

# Define which columns include categorical data
cat_cols = ['f4', 'f8', 'f17', 'f21']

# Columns containing hex values
hex_cols = ['f12']


encoded_X = None
for i in cat_cols:
    label_encoder = LabelEncoder()
    feature = label_encoder.fit_transform(X_train[i].values.astype(str))
    feature = feature.reshape(X_train.shape[0], 1)
    one_hot_encoder = OneHotEncoder(sparse=False, categories='auto')
    feature = one_hot_encoder.fit_transform(feature)
    print(feature)
    X_train[i] = feature
    

print(X_train)


# read in data
dtrain = xgb.DMatrix('challenge2_train.csv?format=csv&label_column=1')
dtest = xgb.DMatrix('challenge2_test.csv?format=csv')

# specify parameters via map
param = {'max_depth': 5, 'eta': 0.001, 'objective': 'binary:logistic'}
num_round = 2
bst = xgb.train(param, dtrain, num_round)
# make prediction
preds = bst.predict(dtest)

accuracy = roc_auc_score(Y_train.astype(int), preds[1:])
print(accuracy)
