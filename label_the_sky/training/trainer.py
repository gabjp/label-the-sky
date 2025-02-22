import efficientnet
from efficientnet.tfkeras import EfficientNetB0
import json
from keras_applications import vgg16, resnext, resnet50
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # info and warning messages are not printed

import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
import shutil
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.constraints import max_norm
from tensorflow.keras.layers import Input, Dense, Dropout, GlobalAveragePooling2D, LeakyReLU
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K


from label_the_sky.training.callbacks import TimeHistory


BACKBONES = ['efficientnet', 'resnext', 'vgg', None, 'resnet']
BATCH_SIZE = 32
BROAD_BANDS = [0, 5, 7, 9, 11]
CLASS_NAMES = ['GALAXY', 'STAR', 'QSO']
MAG_MAX = 35.
N_CHANNELS = [3, 5, 12]
N_CLASSES = 3
OUTPUT_TYPES = ['class', 'magnitudes', 'mockedmagnitudes']
SPLITS = ['train', 'val', 'test']

BACKBONE_FN = {
    'efficientnet': EfficientNetB0,
    'resnext': resnext.ResNeXt50,
    'vgg': vgg16.VGG16, 
    'resnet': resnet50.ResNet50
}

PREPROCESSING_FN = {
    'efficientnet': efficientnet.model.preprocess_input,
    'resnext': resnext.preprocess_input,
    'vgg': vgg16.preprocess_input, 
    'resnet': resnet50.preprocess_input
}


def compute_metrics(y_pred, y_true, target='class', onehot=True, print_cm=True):
    if target not in OUTPUT_TYPES:
        raise ValueError('target should be one of %s, but %s was given' % (
            OUTPUT_TYPES, target))

    if target == 'class':
        if onehot:
            y_pred_arg = np.argmax(y_pred, axis=1)
            y_true_arg = np.argmax(y_true, axis=1)
        else:
            y_pred_arg = np.copy(y_pred)
            y_true_arg = np.copy(y_true)
        print(y_true.shape)
        print(classification_report(
            y_true_arg, y_pred_arg, target_names=CLASS_NAMES, digits=4))

        if print_cm:
            cm = confusion_matrix(y_true_arg, y_pred_arg)
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            cm = np.round(cm, 2)
            print('confusion matrix')
            print(cm)
    elif target in ['magnitudes', 'mockedmagnitudes']:
        err_abs = np.absolute(y_true-y_pred)
        df = pd.DataFrame(err_abs)
        print(df.describe().to_string())
        df = pd.DataFrame(err_abs*MAG_MAX)
        print(df.describe().to_string())
        err_abs = MAG_MAX*err_abs
        print('MAE:', np.mean(err_abs))
        print('MAPE:', np.mean(err_abs/(MAG_MAX*y_true))*100)


def relu_saturated(x):
    return tf.keras.backend.relu(x, max_value=1.)


def serialize(history):
    d = {}
    for k in history.keys():
        d[k] = [float(item) for item in history[k]]
    return d


def set_random_seeds():
    os.environ['PYTHONHASHSEED'] = '0'
    np.random.seed(42)
    tf.random.set_seed(420)
    # session_conf = tf.ConfigProto(
    #     intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
    # sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
    # K.set_session(sess)


def multify_weights(kernel, out_channels):
  mean_1d = np.mean(kernel, axis=-2).reshape(kernel[:,:,-1:,:].shape)
  tiled = np.tile(mean_1d, (out_channels, 1))
  return(tiled)


def copy_weights_tl(model_orig, custom_model, layer_modify):
  layer_to_modify = [layer_modify]
  input_channel = 12

  conf = custom_model.get_config()
  layer_names = [conf['layers'][x]['name'] for x in range(len(conf['layers']))]

  for layer in model_orig.layers:
    if layer.name in layer_names:
      if layer.get_weights() != []:
        target_layer = custom_model.get_layer(layer.name)

        if layer.name in layer_to_modify:    
          kernels = layer.get_weights()[0]
          biases  = layer.get_weights()[1]

          kernels_extra_channel = np.concatenate((kernels,
                                                  multify_weights(kernels, input_channel - 3)),
                                                  axis=-2)
                                                  
          target_layer.set_weights([kernels_extra_channel, biases])
          #target_layer.trainable = False

        else:
          target_layer.set_weights(layer.get_weights())
          #target_layer.trainable = False

# multidy_wights and copy_weights_tl from: 
# https://towardsdatascience.com/implementing-transfer-learning-from-rgb-to-multi-channel-imagery-f87924679166

def print_trainable(model):
    print("Printing trainable layers")
    for layer in model.layers:
        if layer.trainable: print(layer.name)
    


class CustomMAE(tf.keras.losses.Loss):
  def __init__(self):
    super().__init__()
    self.mae = tf.keras.losses.MeanAbsoluteError()
  def call(self, y_true, y_pred):
    mvalue = 99/MAG_MAX
    mask = K.cast(K.not_equal(y_true, mvalue), K.floatx())
    return self.mae(y_true*mask, y_pred*mask)


class Trainer:
    def __init__(self, backbone, n_channels, output_type, base_dir, weights, model_name, save_checkpoints=True, l2=None):
        if backbone not in BACKBONES:
            raise ValueError('backbone should be one of %s, but %s was given' % (
                BACKBONES, backbone))

        if n_channels not in N_CHANNELS:
            raise ValueError('n_channels should be one of %s, but %s was given' % (
                N_CHANNELS, n_channels))

        if output_type not in OUTPUT_TYPES:
            raise ValueError('output_type should be one of %s, but %s was given' % (
                OUTPUT_TYPES, output_type))

        if weights is not None and weights != 'imagenet' and not os.path.exists(weights):
            raise ValueError('weights must be: None, imagenet, or a valid path to a h5 file')

        self.backbone = backbone
        self.n_channels = n_channels
        self.output_type = output_type
        self.weights = weights
        self.model_name = model_name
        self.save_checkpoints = save_checkpoints
        self.l2=l2

        self.clf = None
        self.history = None
        self.run = -1

        self.input_shape = (32, 32, n_channels)
        self.max_norm = None

        self.base_dir = base_dir
        self.data_dir = os.path.join(base_dir, 'data')
        self.save_dir = os.path.join(base_dir, 'trained_models')

        if self.output_type == 'class':
            self.activation = 'softmax'
            self.loss = 'categorical_crossentropy'
            self.metrics = ['accuracy'] 
            self.n_outputs = 3
        else:
            self.activation = relu_saturated
            self.loss = CustomMAE() #'mae'
            self.metrics = None
            self.n_outputs = 12 if self.n_channels != 5 else 5

        tf.compat.v1.disable_eager_execution()
        os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

        if self.backbone is not None:
            self.build_model()

    def load_data(self, dataset, split, return_y=True):
        channels = 12 if self.n_channels==5 else self.n_channels

        X = np.load(os.path.join(
            self.data_dir,
            f'{dataset}_{channels}_X_{split}.npy'))
        if self.n_channels==5:
            X = X[:, :, :, BROAD_BANDS]

        if not return_y:
            return X

        y = np.load(os.path.join(
            self.data_dir,
            f'{dataset}_{channels}_y_{split}.npy'))
        if self.n_channels==5 and self.output_type in ['magnitudes', 'mockedmagnitudes']:
            y = y[:, BROAD_BANDS]

        return X, y

    def preprocess_input(self, X):
        if self.n_channels==3 and X.dtype=='uint8':
            preprocessing_fn = PREPROCESSING_FN.get(self.backbone)
            Xp = preprocessing_fn(
                X,
                backend=tf.keras.backend,
                layers=tf.keras.layers,
                models=tf.keras.models,
                utils=tf.keras.utils
            )
        elif self.n_channels==5 and X.shape[-1]>5:
            Xp = X[:, :, :, BROAD_BANDS]
        else:
            Xp = X

        if Xp.dtype != 'float32':
            raise ValueError('Xp data type should be float32')

        return Xp

    def preprocess_output(self, y):
        if y.dtype!='uint8' and self.output_type!='class':
            yp = y / MAG_MAX
        else:
            yp = y
        return yp

    def build_model(self, learning_rate=1e-4, freeze_backbone=False, skip_mismatch=False):
        tf.keras.backend.clear_session()

        architecture_fn = BACKBONE_FN.get(self.backbone)
        weights0 = 'imagenet' if self.weights == 'imagenet' else None

        if self.weights == 'imagenet' and self.n_channels == 12:
            p_net = architecture_fn(
                input_shape = (32,32,3),
                include_top=False,
                weights=weights0,
                backend=tf.keras.backend,
                layers=tf.keras.layers,
                models=tf.keras.models,
                utils=tf.keras.utils
            )
            config = p_net.get_config()
            config["layers"][0]["config"]["batch_input_shape"] = (None, 32, 32, 12)
            model = tf.keras.models.Model.from_config(config)
            id = 2 if self.backbone == "resnet" else 1
            f_conv =  model.get_layer(index=id).name
            copy_weights_tl(p_net, model, f_conv)

        else:
            model = architecture_fn(
                input_shape = self.input_shape,
                include_top=False,
                weights=weights0,
                backend=tf.keras.backend,
                layers=tf.keras.layers,
                models=tf.keras.models,
                utils=tf.keras.utils
            )

        x = model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(
            1024,
            kernel_initializer='glorot_uniform',
            kernel_constraint=self.max_norm)(x)
        x = LeakyReLU()(x)

        # top layer
        y = Dropout(0.5)(x) 
        y = Dense(self.n_outputs, activation=self.activation)(y)

        self.top_layer_idx = -4
        self.model = Model(inputs=model.input, outputs=y)

        self.embedder = Model(inputs=self.model.input, outputs=x)
        self.embedder_yx = Model(inputs=self.model.input, outputs=[y, x])

        if self.weights not in [None, 'imagenet']:
            self.load_weights(self.weights, skip_mismatch)

        if freeze_backbone:
            for layer in self.model.layers[:self.top_layer_idx]:
                layer.trainable = False

        opt = Adam(lr=learning_rate)

        # Here, I'll try to add an r2 regularization. (Please don't break)
        # l2 best = 0.0007
        if self.l2 != None:
            for i in range(len(self.model.layers)):
                if self.model.layers[i].get_weights() != []:
                #if isinstance(self.model.layers[i], tf.keras.layers.Conv2D) or isinstance(self.model.layers[i], tf.keras.layers.Dense):
                    print('Adding regularizer to layer {}'.format(self.model.layers[i].name))
                    self.model.layers[i].kernel_regularizer = tf.keras.regularizers.l2(self.l2)

            model_json = self.model.to_json()
            self.model.save_weights("./temp_weights.h5")
            self.model = tf.keras.models.model_from_json(model_json)
            self.model.load_weights("./temp_weights.h5", by_name=False, skip_mismatch=False)
            os.remove("./temp_weights.h5")

        # Changes end here
        self.model.compile(loss=self.loss, optimizer=opt, metrics=self.metrics)

    def build_top_clf(self, inpt_dim, learning_rate=1e-4):
        inpt = Input(shape=(inpt_dim,))
        x = Dense(12, activation=LeakyReLU())(inpt)
        x = Dense(N_CLASSES, activation=self.activation)(x)
        self.clf = Model(inpt, x)

        opt = Adam(lr=learning_rate)
        self.clf.compile(loss=self.loss, optimizer=opt, metrics=self.metrics)

    def describe(self, verbose=False):
        if verbose:
            self.model.summary()

        if self.model is not None:
            trainable = np.sum([
                tf.keras.backend.count_params(w) for w in self.model.trainable_weights])
            non_trainable = np.sum([
                tf.keras.backend.count_params(w) for w in self.model.non_trainable_weights])
            print('******************************')
            print('total params\t', f'{int(trainable + non_trainable):,}')
            print('trainable\t', f'{int(trainable):,}')
            print('non-trainable\t', f'{int(non_trainable):,}')
            print()

        print('backbone\t', self.backbone)
        print('n_channels\t', self.n_channels)
        print('output\t\t', self.output_type)
        print('weights\t\t', self.weights)
        print('******************************')

    def load_weights(self, weights_file, skip_mismatch=False):
        if skip_mismatch: # use for training
            self.model.load_weights(weights_file, by_name=True, skip_mismatch=True)
        else: # use for inference
            self.model.load_weights(weights_file)
        print('loaded .h5 weights')

    def pick_best_model(self, metric='val_loss'):
        if self.history is None:
            raise ValueError('no training history available.')
        min_metrics = [min(self.history[i][metric]) for i in range(len(self.history))]
        argmin = np.argmin(min_metrics)
        shutil.copy2(
            os.path.join(self.save_dir, f'{self.model_name}_{argmin}.h5'),
            os.path.join(self.save_dir, f'{self.model_name}.h5'))

    def set_callbacks(self):
        if self.save_checkpoints:
            self.callbacks = [
                ModelCheckpoint(
                    os.path.join(self.save_dir, f'{self.model_name}_{self.run}.h5'),
                    monitor='val_loss', save_best_only=True, save_weights_only=True, mode='min')
            ]

    def set_class_weights(self, y):
        if self.output_type != 'class':
            self.class_weights = None
        else:
            if len(y.shape) > 1:
                yy = np.argmax(y, axis=1)
            self.class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(yy), y=yy)
            print('set class weights to', self.class_weights)

    def train(self, X_train, y_train, X_val, y_val, mode='from_scratch', epochs=100, runs=3):
        # available modes: from_scratch, finetune, top_clf
        if mode!='from_scratch' and self.weights is None:
            raise ValueError(f'{mode} not available for weights=None')
        self.set_class_weights(y_train)

        if mode=='top_clf':
            Xp_train = self.extract_features(X_train)
            Xp_val = self.extract_features(X_val)
        else:
            Xp_train = self.preprocess_input(X_train)
            Xp_val = self.preprocess_input(X_val)
        yp_train = self.preprocess_output(y_train)
        yp_val = self.preprocess_output(y_val)

        if mode=='from_scratch':
            history = self.from_scratch(Xp_train, yp_train, Xp_val, yp_val, epochs, runs)
        elif mode=='finetune':
            history = self.finetune(Xp_train, yp_train, Xp_val, yp_val, epochs, runs)
        elif mode=='top_clf':
            history = self.train_top(Xp_train, yp_train, Xp_val, yp_val, epochs, runs)

        self.history = history

    def train_lowdata(self, X_train, y_train, X_val, y_val, mode='from_scratch', epochs=300, runs=3,
                      size_increment=100, n_subsets=20):
        if mode!='from_scratch' and self.weights is None:
            raise ValueError(f'{mode} not available for weights=None')

        self.set_class_weights(y_train)

        if mode=='top_clf':
            Xp_train = self.extract_features(X_train)
            Xp_val = self.extract_features(X_val)
        else:
            Xp_train = self.preprocess_input(X_train)
            Xp_val = self.preprocess_input(X_val)
        yp_train = self.preprocess_output(y_train)
        yp_val = self.preprocess_output(y_val)

        time_cb = TimeHistory()
        histories = []

        rnd = np.random.uniform(size=Xp_train.shape[0])
        percentages = np.linspace(size_increment, size_increment*n_subsets, n_subsets)/Xp_train.shape[0]

        for p in percentages:
            Xpp_train = Xp_train[rnd <= p]
            ypp_train = yp_train[rnd <= p]

            run_suffix = f'-{Xpp_train.shape[0]}'

            if mode=='from_scratch':
                ht = self.from_scratch(Xpp_train, ypp_train, Xp_val, yp_val, epochs, runs, run_suffix)
            elif mode=='finetune':
                ht = self.finetune(Xpp_train, ypp_train, Xp_val, yp_val, epochs, runs, run_suffix)
            elif mode=='top_clf':
                ht = self.train_top(Xpp_train, ypp_train, Xp_val, yp_val, epochs, runs, run_suffix)

            histories.append(ht)

        self.history = histories

    def from_scratch(self, X_train, y_train, X_val, y_val, epochs, runs, run_suffix=''):
        time_cb = TimeHistory()
        histories = []

        for run in range(runs):
            self.run = f'{run}{run_suffix}'
            self.build_model() 
            self.set_callbacks()
            history = self.model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                batch_size=BATCH_SIZE,
                epochs=epochs,
                callbacks=self.callbacks + [time_cb],
                class_weight=self.class_weights,
                verbose=2
            )
            ht = history.history
            ht['times'] = time_cb.times
            histories.append(ht)

            print("RUN #", run)
            if 'val_accuracy' in  ht.keys():
                print('val acc', np.max(ht['val_accuracy']))
            print('val loss', np.min(ht['val_loss']))

        return histories

    def finetune(self, X_train, y_train, X_val, y_val, epochs, runs, run_suffix='', learning_rate=1e-5):
        time_cb = TimeHistory()
        histories = []

        for run in range(runs):
            self.run = f'{run}{run_suffix}'
            self.build_model(freeze_backbone=True, skip_mismatch=True)
            self.set_callbacks()
            #print_trainable(self.model) # Check if right layers are trainable
            history0 = self.model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                batch_size=BATCH_SIZE,
                epochs=10,
                callbacks=self.callbacks,
                class_weight=self.class_weights,
                verbose=2
            )
            for l in self.model.layers:
                l.trainable = True

            opt = Adam(lr=learning_rate)

            self.model.compile(
                loss=self.loss,
                optimizer=opt,
                metrics=self.metrics)

           # print_trainable(self.model) # Check if right layers are trainable

            history = self.model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                batch_size=BATCH_SIZE,
                epochs=epochs,
                callbacks=self.callbacks + [time_cb],
                class_weight=self.class_weights,
                verbose=2
            )
            ht = history.history
            ht['times'] = time_cb.times
            histories.append(ht)

            print('RUN #', run)
            print('val acc', np.max(ht['val_accuracy']))

        return histories

    def train_top(self, X_train, y_train, X_val, y_val, epochs, runs, run_suffix=''):
        inpt_dim = X_train.shape[1]
        time_cb = TimeHistory()
        histories = []

        for run in range(runs):
            self.run = f'{run}{run_suffix}'
            self.build_top_clf(inpt_dim)
            self.set_callbacks()
            history = self.clf.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                batch_size=BATCH_SIZE,
                epochs=epochs,
                callbacks=self.callbacks + [time_cb],
                class_weight=self.class_weights,
                verbose=2
            )
            ht = history.history
            ht['times'] = time_cb.times
            histories.append(ht)

            print("RUN #", run)
            print('val acc', np.max(ht['val_accuracy']))

        return histories

    def print_history(self):
        print(self.history)

    # TODO refactor
    def dump_history(self, history_dir):
        if type(self.history) == list:
            hist_tmp = []
            for h in self.history:
                if type(h) == list:
                    hist_tmp2 = []
                    for h_perc in h:
                        hist_tmp2.append(serialize(h_perc))
                    hist_tmp.append(hist_tmp2)
                else:
                    hist_tmp.append(serialize(h))
            self.history = hist_tmp
        else:
            self.history = serialize(self.history)

        if not os.path.exists(os.path.join(self.base_dir, history_dir)):
            os.makedirs(os.path.join(self.base_dir, history_dir))
        with open(os.path.join(self.base_dir, history_dir, self.model_name+'.json'), 'w') as f:
            json.dump(self.history, f)
        print('dumped history to', os.path.join(self.base_dir, history_dir, f'{self.model_name}.json'))

    def evaluate(self, X, y, print_cm=True):
        yp = self.preprocess_output(y)
        y_hat = []
        if self.backbone is not None:
            Xp = self.preprocess_input(X)
            y_hat = self.model.predict(Xp)
        elif self.clf is not None:
            Xp = self.preprocess_output(X)
            y_hat = self.clf.predict(Xp)
        else:
            print('no model to use on evaluation')
            return
        compute_metrics(y_hat, yp, target=self.output_type, print_cm=print_cm)

    def predict(self, X):
        if self.backbone is not None:
            Xp = self.preprocess_input(X)
            yhat = self.model.predict(Xp)
            if self.output_type != 'class':
                return yhat * MAG_MAX
            return yhat
        else:
            Xp = self.preprocess_output(X)
            return self.clf.predict(Xp)

    def extract_features(self, X):
        Xp = self.preprocess_input(X)
        return self.embedder.predict(Xp)

    def extract_features_and_predict(self, X):
        # TODO apply MAG_MAX when output_type != class
        Xp = self.preprocess_input(X)
        yhat, X_features = self.embedder_yx.predict(Xp)
        if self.output_type != 'class':
            return yhat * MAG_MAX, X_features
        return yhat, X_features
