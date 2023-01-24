from sklearn.metrics import classification_report
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler

CLASSES = ["GALAXY", "STAR", "QSO"]
BASE_DIR = os.environ['HOME']
CHECKPOINT_PATH = os.path.join(BASE_DIR,"/trained_models/meta-model_checkpoint.h5")

def print_classification_report(true,pred):
    print(classification_report(true, pred, digits=6, target_names=CLASSES))

def print_wise_report(y_val, pred_val, with_wise_index_val, no_wise_index_val, y_test, pred_test, with_wise_index_test, no_wise_index_test, m_name="", trainer=None):
    """
    If trainer != None, pred_(val/test) should receive X (feature values). The trainer will generate the predictions.
    Else, val/test should receive y (predicted target).
    """
    
    if trainer != None:
        report  = trainer.evaluate
    else:
        report = print_classification_report

    print(f"{m_name} performance on validation set", flush=True)
    report(y_val, pred_val)

    print(f"{m_name} performance on validation (with_wise) set")
    report(y_val[with_wise_index_val], pred_val[with_wise_index_val])

    print(f"{m_name} performance on validation (no_wise) set")
    report(y_val[no_wise_index_val], pred_val[no_wise_index_val])

    print(f"{m_name} performance on test set", flush=True)
    report(y_test, pred_test)

    print(f"{m_name} performance on test (with_wise) set", flush=True)
    report(y_test[with_wise_index_test], pred_test[with_wise_index_test])

    print(f"{m_name} performance on test (no_wise) set", flush=True)
    report(y_test[no_wise_index_test], pred_test[no_wise_index_test])

class MetaTrainer:
    def __init__(self):
        self.model = keras.models.Sequential()

        self.model.add(keras.layers.Input(shape=(6,)))
        self.model.add(keras.layers.Dense(300, activation = "relu"))
        self.model.add(keras.layers.Dense(100, activation = "relu"))
        self.model.add(keras.layers.Dense(3, activation = "softmax"))
        self.model.compile(loss = "categorical_crossentropy", optimizer = Adam(lr=1e-3), metrics = ["accuracy"])
    
    def fit(self, X_train, y_train, X_val, y_val):
        self.ss = StandardScaler()
        self.ss.fit(X_train)

        t_X_train = self.ss.transform(X_train)
        t_X_val = self.ss.transform(X_val)

        self.model.fit(t_X_train, y_train,validation_data = (t_X_val, y_val), batch_size = 32, verbose = 2, epochs = 30, 
            class_weight=compute_class_weight(class_weight='balanced', classes=[0,1,2], y = np.argmax(y_train, axis=1)),
            callbacks = [tf.keras.callbacks.ModelCheckpoint(
                        filepath=CHECKPOINT_PATH,
                        save_weights_only=True,
                        monitor='val_loss',
                        mode='min',
                        save_best_only=True)])

        self.model.load_weights(CHECKPOINT_PATH)

    def predict_proba(self, X):
        t_X = self.ss.transform(X)
        return self.model.predict(t_X)
