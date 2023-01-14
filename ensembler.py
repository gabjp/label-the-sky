from label_the_sky.training.trainer import Trainer, set_random_seeds
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import  LogisticRegression
import sys
import pandas as pd
import numpy as np
import os
from scipy.stats import loguniform

base_dir = os.environ['HOME']
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

def gen():

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
    split = list(skf.split(X_train_csv, y_train_csv))

    print("Generating RF data")

    RF_pred = np.array([]).reshape(0,3)
    RF_target = np.array([]).reshape(0,1)
    for i, (train_index, test_index) in enumerate(split):
        print(f"Starting fold {i}")
        rf = RandomForestClassifier(random_state=2, n_estimators=100, bootstrap=False)
        rf.fit(X_train_csv.iloc[train_index], y = y_train_csv.iloc[train_index])
        RF_pred = np.concatenate((RF_pred,rf.predict_proba(X_train_csv.iloc[test_index])), axis=0) 
        RF_target = np.concatenate((RF_target,np.array([y_train_csv.iloc[test_index].values]).T), axis = 0)


    print("Generating 12ch CNN data")

    weight_file = os.path.join(base_dir, 'trained_models', '0601_vgg_12_unl_w99.h5')
    CNN12_pred = np.array([]).reshape(0,3)
    CNN12_target = np.array([]).reshape(0,3)
    for i, (train_index, test_index) in enumerate(split):
        print(f"Starting fold {i}")
        trainer = Trainer(
            backbone="vgg",
            n_channels=12,
            output_type='class',
            base_dir=base_dir,
            weights=weight_file,
            model_name=f'0300_vgg_12_unl_w99_clf_ft1_full',
            l2 = 0.0007 
        )
        trainer.train(X_train_12ch[train_index,:,:,:], y_train_12ch[train_index,:], X_val_12ch, y_val_12ch, mode="finetune", epochs=100, runs=1)
        CNN12_pred = np.concatenate((CNN12_pred,trainer.predict(X_train_12ch[test_index,:,:,:])), axis=0)
        CNN12_target = np.concatenate((CNN12_target,y_train_12ch[test_index,:]), axis=0)

    meta_features = np.concatenate((CNN12_pred,RF_pred), axis=1)
    meta_target = RF_target

    print("Saving meta-model trainig set")
    np.save("../data/meta_features.npy",meta_features)
    np.save("../data/meta_target.npy",meta_target)

def rgs():
    space = dict()
    space['solver'] = ['newton-cg', 'lbfgs', 'liblinear']
    space['penalty'] = ['none', 'l1', 'l2', 'elasticnet']
    space['C'] = loguniform(1e-5, 100)


    X = np.load("../data/meta_features.npy")
    y = np.load("../data/meta_target.npy")
    lr = LogisticRegression()
    search = RandomizedSearchCV(estimator=lr, param_distributions=space, n_iter=100, cv=3, verbose=1, random_state=2, n_jobs=-1, scoring="accuracy")
    result = search.fit(X,y)
    
    print('Best Score: %s' % result.best_score_)
    print('Best Hyperparameters: %s' % result.best_params_)


def eval():
    pass

if __name__=="__main__":
    if "g" in sys.argv[2]:
        gen()
    if "s" in sys.argv[2]:
        rgs()
    if "e" in sys.argv[2]:
        eval()
