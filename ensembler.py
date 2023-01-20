from label_the_sky.training.trainer import Trainer, set_random_seeds
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV, cross_validate
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
from sklearn.metrics import confusion_matrix, make_scorer, accuracy_score, precision_score, recall_score, f1_score
from scipy.stats import wilcoxon
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.optimizers import Adam
from sklearn.utils.class_weight import compute_class_weight

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
             'z_iso',
             'w1mpro',
             'w2mpro']

print("Starting", flush=True)

dataset = sys.argv[1]

dataset_csv = pd.read_csv(dataset + ".csv").fillna(99)

#Fixing csv data target labels (This is pretty ugly, find a better fix later)
dataset_csv["target"] = dataset_csv["target"].apply(lambda c: CLASS_MAP[c])

train_csv = dataset_csv[(dataset_csv.split=="train")]
val_csv = dataset_csv[dataset_csv.split=="val"]
test_csv = dataset_csv[dataset_csv.split=="test"]

with_wise_index_val = val_csv.w1mpro != 99
no_wise_index_val = val_csv.w1mpro == 99

print("Finished loading csv", flush=True)

# Load tabular data
X_train_csv, y_train_csv = (train_csv[_morph+_feat], train_csv["target"])
X_val_csv, y_val_csv = (val_csv[_morph+_feat], val_csv["target"])
X_test_csv, y_test_csv = (test_csv[_morph+_feat], test_csv["target"])

print("Finished loading tabular data", flush=True)

#Load 12ch image data
X_train_12ch, y_train_12ch = (np.load(f"../data/{dataset}_12_X_train.npy"), np.load(f"../data/{dataset}_12_y_train.npy"))
X_val_12ch, y_val_12ch = (np.load(f"../data/{dataset}_12_X_val.npy"), np.load(f"../data/{dataset}_12_y_val.npy"))
X_test_12ch, y_test_12ch = (np.load(f"../data/{dataset}_12_X_test.npy"), np.load(f"../data/{dataset}_12_y_test.npy"))

print("Finished loading image data", flush=True)

#Load 3ch image data
#X_train_3ch, y_train_3ch = (np.load(f"../data/{dataset}_3_X_train.npy"), np.load(f"../data/{dataset}_3_y_train.npy"))
#X_val_3ch, y_val_3ch = (np.load(f"../data/{dataset}_3_X_val.npy"), np.load(f"../data/{dataset}_3_y_val.npy"))
#X_test_3ch, y_test_3ch = (np.load(f"../data/{dataset}_3_X_test.npy"), np.load(f"../data/{dataset}_3_y_test.npy"))

print("Finished loading data", flush=True)

set_random_seeds()

def gen():
    print("Generating data to train the meta-model", flush=True)

    skf = StratifiedKFold(n_splits=5, shuffle=False, random_state=2)
    split = list(skf.split(X_train_csv, y_train_csv))

    print("Generating RF data", flush=True)

    RF_pred = np.array([]).reshape(0,3)
    RF_target = np.array([]).reshape(0,1)
    for i, (train_index, test_index) in enumerate(split):
        print(f"Starting fold {i}", flush=True)
        rf = RandomForestClassifier(random_state=2, n_estimators=100, bootstrap=False)
        rf.fit(X_train_csv.iloc[train_index], y = y_train_csv.iloc[train_index])
        RF_pred = np.concatenate((RF_pred,rf.predict_proba(X_train_csv.iloc[test_index])), axis=0) 
        RF_target = np.concatenate((RF_target,np.array([y_train_csv.iloc[test_index].values]).T), axis = 0)


    print("Generating 12ch CNN data", flush=True)

    weight_file = os.path.join(base_dir, 'trained_models', '0601_vgg_12_unl_w99.h5')
    CNN12_pred = np.array([]).reshape(0,3)
    CNN12_target = np.array([]).reshape(0,3)
    for i, (train_index, test_index) in enumerate(split):
        print(f"Starting fold {i}", flush=True)
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

    print("Saving meta-model trainig set", flush=True)
    np.save("../data/meta_features.npy",meta_features)
    np.save("../data/meta_target.npy",meta_target)

def wil():

    scoring = {
    'Precision_QSO': make_scorer(precision_score, average=None, labels=[2]),
    'Recall_QSO': make_scorer(recall_score, average=None, labels=[2]),
    'f1_QSO': make_scorer(f1_score, average=None, labels=[2]),
    'Precision_STAR': make_scorer(precision_score, average=None, labels=[1]),
    'Recall_STAR': make_scorer(recall_score, average=None, labels=[1]),
    'f1_STAR': make_scorer(f1_score, average=None, labels=[1]),
    'Precision_GAL': make_scorer(precision_score, average=None, labels=[0]),
    'Recall_GAL': make_scorer(recall_score, average=None, labels=[0]),
    'f1_GAL': make_scorer(f1_score, average=None, labels=[0]),
    'f1_macro': make_scorer(f1_score, average='macro')
}

    X_train_meta = np.load("../data/meta_features.npy")
    y_train_meta = np.load("../data/meta_target.npy").ravel()

    skf = StratifiedKFold(n_splits=5, shuffle=False, random_state=2)
    rf = RandomForestClassifier(random_state=2, n_estimators=100, bootstrap=False)

    resultsrf = cross_validate(estimator=rf, X=X_train_csv, y=y_train_csv, cv=skf, scoring=scoring, return_train_score=False)
    print("5-fold RF:")
    for key in resultsrf:
        print(f"{key}: {np.round(resultsrf[key]*100, 2)}")

    skf = StratifiedKFold(n_splits=5, shuffle=False, random_state=2)
    lr = LogisticRegression(C=0.685, penalty='l1', solver='saga')

    resultslr = cross_validate(estimator=lr, X=X_train_meta, y=y_train_meta, cv=skf, scoring=scoring, return_train_score=False)
    print("5-fold LR")
    for key in resultslr:
        print(f"{key}: {np.round(resultslr[key]*100, 2)}")

    for key in resultslr:
        w, p = wilcoxon(np.round(resultslr[key]*100, 2), np.round(resultsrf[key]*100, 2), mode='exact', alternative='greater')
        print(f"{key}: W: {w} p: {p}")

    


def eval():
    #Load Meta-model data
    X_train_meta = np.load("../data/meta_features.npy")
    y_train_meta = np.load("../data/meta_target.npy").ravel()
    y_train_meta = keras.utils.to_categorical(y_train_meta, num_classes=3)

    print("Starting RF evaluation", flush=True)
    rf = RandomForestClassifier(random_state=2, n_estimators=100, bootstrap=False)
    rf.fit(X_train_csv, y=y_train_csv)

    RF_pred_val = rf.predict(X_val_csv)
    RF_pred_test = rf.predict(X_test_csv)
    print("RF performance on validation set", flush=True)
    print(classification_report(y_val_csv, RF_pred_val, digits=6))

    print("RF performance on validation (with_wise) set")
    print(classification_report(y_val_csv[with_wise_index_val], RF_pred_val[with_wise_index_val], digits=6))

    print("RF performance on validation (no_wise) set")
    print(classification_report(y_val_csv[no_wise_index_val], RF_pred_val[no_wise_index_val], digits=6))

    print("RF performance on test set", flush=True)
    print(classification_report(y_test_csv, RF_pred_test, digits=6))
    RF_proba_val = rf.predict_proba(X_val_csv)
    RF_proba_test = rf.predict_proba(X_test_csv)


    print("Starting 12ch CNN evaluation", flush=True)
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
    print("CNN performane on validation set", flush=True)
    trainer.evaluate(X_val_12ch, y_val_12ch)

    print("CNN performane on validation (with_wise) set", flush=True)
    trainer.evaluate(X_val_12ch[with_wise_index_val], y_val_12ch[with_wise_index_val])

    print("CNN performane on validation (no_wise) set ", flush=True)
    trainer.evaluate(X_val_12ch[no_wise_index_val], y_val_12ch[no_wise_index_val])

    print("CNN performance on test set")
    trainer.evaluate(X_test_12ch, y_test_12ch)
    CNN12_proba_val = trainer.predict(X_val_12ch)
    CNN12_proba_test = trainer.predict(X_test_12ch)

    # print("Starting LR evaluation", flush=True)
    # lr = LogisticRegression(C=0.685, penalty='l1', solver='saga')
    # lr.fit(X_train_meta, y=y_train_meta)

    X_val_meta = np.concatenate((CNN12_proba_val, RF_proba_val), axis=1)
    X_test_meta = np.concatenate((CNN12_proba_test, RF_proba_test), axis=1)
    # y_val_meta = y_val_csv.values
    # y_test_meta = y_test_csv.values
    y_val_meta = y_val_12ch
    y_test_meta = y_test_12ch
    

    # LR_pred_val = lr.predict(X_val_meta)
    # LR_pred_test = lr.predict(X_test_meta)

    print("Starting meta-model evaluation", flush=True)
    meta = keras.models.Sequential()
    meta.add(keras.layers.Input(shape=(6,)))
    meta.add(keras.layers.Dense(300, activation = "relu"))
    meta.add(keras.layers.Dense(100, activation = "relu"))
    meta.add(keras.layers.Dense(3, activation = "softmax"))

    meta.compile(loss = "categorical_crossentropy", optimizer = Adam(lr=1e-3), metrics = ["accuracy"])
    meta.fit(X_train_meta, y_train_meta,validation_data = (X_val_meta, y_val_meta), batch_size =32, verbose =2, epochs=20, 
            class_weight=compute_class_weight(class_weight='balanced', classes=[0,1,2], y=np.argmax(y_train_meta, axis=1)))

    predict_y_val = np.argmax(meta.predict(X_val_meta), axis=1)
    predict_y_test = np.argmax(meta.predict(X_test_meta), axis=1)
    y_val_meta = np.argmax(y_val_meta, axis=1)
    y_test_meta = np.argmax(y_test_meta, axis=1)

    print("Meta-model performance on validation set", flush=True)
    print(classification_report(y_val_meta, predict_y_val, digits=6))

    print("Meta-model performance on validation (with_wise) set")
    print(classification_report(y_val_csv[with_wise_index_val], predict_y_val[with_wise_index_val], digits=6))

    print("Meta-model performance on validation (no_wise) set")
    print(classification_report(y_val_csv[no_wise_index_val], predict_y_val[no_wise_index_val], digits=6))

    print("Meta-model performance on test set", flush=True)
    print(classification_report(y_test_meta, predict_y_test, digits=6)) 




if __name__=="__main__":
    if "g" in sys.argv[2]:
        gen()
    if "w" in sys.argv[2]:
        wil()
    if "e" in sys.argv[2]:
        eval()
