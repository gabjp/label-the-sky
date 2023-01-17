from label_the_sky.training.trainer import Trainer, set_random_seeds
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import  LogisticRegression
import sys
import pandas as pd
import numpy as np
import os
from scipy.stats import loguniform
from sklearn.metrics import classification_report
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

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

#Load 3ch image data
X_train_3ch, y_train_3ch = (np.load(f"../data/{dataset}_3_X_train.npy"), np.load(f"../data/{dataset}_3_y_train.npy"))
X_val_3ch, y_val_3ch = (np.load(f"../data/{dataset}_3_X_val.npy"), np.load(f"../data/{dataset}_3_y_val.npy"))
X_test_3ch, y_test_3ch = (np.load(f"../data/{dataset}_3_X_test.npy"), np.load(f"../data/{dataset}_3_y_test.npy"))

print("Finished loading data")

set_random_seeds()

def gen():
    print("Generating data to train the meta-model")

    skf = StratifiedKFold(n_splits=5, shuffle=False, random_state=2)
    split = list(skf.split(X_train_csv, y_train_csv))

    print("Generating SVM data")

    SVM_pred = np.array([]).reshape(0,3)
    SVM_target = np.array([]).reshape(0,1)
    for i, (train_index, test_index) in enumerate(split):
        print(f"Starting fold {i}")
        ss = StandardScaler()
        ss.fit(X_train_csv.iloc[train_index])
        svm_train = ss.transform(X_train_csv.iloc[train_index])
        svm_eval = ss.transform(X_train_csv.iloc[test_index])
        svm = SVC(decision_function_shape="ovo", kernel="rbf", C = 100, random_state=2, probability=True)
        svm.fit(svm_train, y = y_train_csv.iloc[train_index])
        SVM_pred = np.concatenate((SVM_pred,svm.predict_proba(svm_eval)), axis=0) 
        SVM_target = np.concatenate((SVM_target,np.array([y_train_csv.iloc[test_index].values]).T), axis = 0)

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

    meta_features = np.concatenate((CNN12_pred,RF_pred,SVM_pred), axis=1)
    meta_target = RF_target

    print("Saving meta-model trainig set")
    np.save("../data/meta_features.npy",meta_features)
    np.save("../data/meta_target.npy",meta_target)

def rgs():
    space = dict()
    space['gamma'] = ['scale', 'auto', 0.1,1,10]
    space['kernel'] = ['rbf', 'poly']
    space['C'] = loguniform(1e-5, 100)
    space['decision_function_shape'] = ['ovr', 'ovo']

    ss=StandardScaler()
    ss.fit(X_train_csv)
    transformed = ss.transform(X_train_csv)

    svc = SVC(random_state=2)
    search = RandomizedSearchCV(estimator=svc, param_distributions=space, n_iter=30, cv=3, verbose=1, random_state=2, n_jobs=-1, scoring="accuracy")
    result = search.fit(transformed,y_train_csv)

    print('Best Score: %s' % result.best_score_)
    print('Best Hyperparameters: %s' % result.best_params_)


def eval():
    #Load Meta-model data
    X_train_meta = np.load("../data/meta_features.npy")
    y_train_meta = np.load("../data/meta_target.npy").ravel()

    print("Starting RF evaluation")
    rf = RandomForestClassifier(random_state=2, n_estimators=100, bootstrap=False)
    rf.fit(X_train_csv, y=y_train_csv)

    RF_pred_val = rf.predict(X_val_csv)
    RF_pred_test = rf.predict(X_test_csv)
    print("RF performance on validation set")
    print(classification_report(y_val_csv, RF_pred_val, digits=6))
    print("RF performance on test set")
    print(classification_report(y_test_csv, RF_pred_test, digits=6))
    RF_proba_val = rf.predict_proba(X_val_csv)
    RF_proba_test = rf.predict_proba(X_test_csv)


    print("Starting 12ch CNN evaluation")
    weight_file = os.path.join(base_dir, 'trained_models', "0601_vgg_12_unl_w99_clf_ft1_full.h5")
    trainer = Trainer(
        backbone="vgg",
        n_channels=12,
        output_type='class',
        base_dir=base_dir,
        weights=weight_file,
        model_name=f'0301_vgg_12_unl_w99_clf_ft1_full',
        l2 = 0.0007 
    )
    print("CNN performane on validation set")
    trainer.evaluate(X_val_12ch, y_val_12ch)
    print("CNN performance on test set")
    trainer.evaluate(X_test_12ch, y_test_12ch)
    CNN12_proba_val = trainer.predict(X_val_12ch)
    CNN12_proba_test = trainer.predict(X_test_12ch)


    print("Starting SVM evaluation")

    ss = StandardScaler()
    ss.fit(X_train_csv)
    t_X_train_csv = ss.transform(X_train_csv)
    t_X_val_csv = ss.transform(X_val_csv)
    t_X_test_csv = ss.transform(X_test_csv)
    svm = SVC(decision_function_shape="ovo", kernel="rbf", C = 100, random_state=2, probability=True)
    svm.fit(t_X_train_csv, y=y_train_csv)

    SVM_pred_val = svm.predict(t_X_val_csv)
    SVM_pred_test = svm.predict(t_X_test_csv)
    print("SVM performance on validation set")
    print(classification_report(y_val_csv, SVM_pred_val, digits=6))
    print("SVM performance on test set")
    print(classification_report(y_test_csv, SVM_pred_test, digits=6))
    SVM_proba_val = svm.predict_proba(t_X_val_csv)
    SVM_proba_test = svm.predict_proba(t_X_test_csv)

    print("Starting LR evaluation")
    lr = LogisticRegression(C=0.568, penalty='l2', solver='lbfgs')
    lr.fit(X_train_meta, y=y_train_meta)

    X_val_meta = np.concatenate((CNN12_proba_val, RF_proba_val, SVM_proba_val), axis=1)
    X_test_meta = np.concatenate((CNN12_proba_test, RF_proba_test, SVM_proba_test), axis=1)
    y_val_meta = y_val_csv.values
    y_test_meta = y_test_csv.values

    LR_pred_val = lr.predict(X_val_meta)
    LR_pred_test = lr.predict(X_test_meta)

    print("LR performance on validation set")
    print(classification_report(y_val_meta, LR_pred_val, digits=6))
    print("LR performance on test set")
    print(classification_report(y_test_meta, LR_pred_test, digits=6))



if __name__=="__main__":
    if "g" in sys.argv[2]:
        gen()
    if "s" in sys.argv[2]:
        rgs()
    if "e" in sys.argv[2]:
        eval()
