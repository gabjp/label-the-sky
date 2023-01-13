from label_the_sky.training.trainer import Trainer, set_random_seeds
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import  LogisticRegression
import sys
import pandas as pd
import numpy as np

CLASS_MAP = {0:2,1:1,2:0} # 0 - Galaxy, 1 - Star, 2 - Quasar

_morph = ['FWHM_n', 'A', 'B', 'KRON_RADIUS']
_feat = ['u_iso',
             'J0378_iso',
             'J0395_iso',
             'J0410_iso',
             'J0430_iso',
             'g_iso',
             'J0515_iso',
             'r_iso',
             'J0660_iso',
             'i_iso',
             'J0861_iso',
             'z_iso']

set_random_seeds()

dataset = sys.argv[1]

dataset_csv = pd.read_csv(dataset + ".csv")

#Fixing csv data target labels (This is pretty ugly, find a better fix later)
dataset_csv["target"] = dataset_csv["target"].apply(lambda c: CLASS_MAP[c])

train_csv = dataset_csv[(dataset_csv.split=="train")]
val_csv = dataset_csv[dataset_csv.split=="val"]
test_csv = dataset_csv[dataset_csv.split=="test"]

# Load tabular data
X_train_csv, y_train_csv = (train_csv[_morph+_feat], train_csv["target"])
X_val_csv, y_val_csv = (val_csv[_morph+_feat], val_csv["target"])
X_test_csv, y_test_csv = (test_csv[_morph+_feat], test_csv["target"])

#Load 12ch image data
X_train_12ch, y_train_12ch = (np.load(f"../data/{dataset}_12_X_train.npy"), np.load(f"../data/{dataset}_12_y_train.npy"))
X_val_12ch, y_val_12ch = (np.load(f"../data/{dataset}_12_X_val.npy"), np.load(f"../data/{dataset}_12_y_val.npy"))
X_test_12ch, y_test_12ch = (np.load(f"../data/{dataset}_12_X_test.npy"), np.load(f"../data/{dataset}_12_y_test.npy"))

print("Finished loading data")
print("Generating data to train the meta-model")

skf = StratifiedKFold(n_splits=5, shuffle=False, random_state=2)
split = skf.split(X_train_csv, y_train_csv)

print("Generating RF data")

RF_pred = []
for i, (train_index, test_index) in enumerate(split):
    print(f"Starting fold {i}")
    rf = RandomForestClassifier(random_state=2, n_estimators=100, bootstrap=False)
    rf.fit(X_train_csv.iloc[train_index], y = y_train_csv.iloc[train_index])
    RF_pred.append((rf.predict_proba(X_train_csv.iloc[test_index]), y_train_csv.iloc[test_index]))

print(RF_pred)

print("Generating 12ch CNN data")
for i, (train_index, test_index) in enumerate(split):
    continue