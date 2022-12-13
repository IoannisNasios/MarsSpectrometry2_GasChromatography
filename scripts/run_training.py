# -*- coding: utf-8 -*-

#from matplotlib import pyplot as plt#, cm
import numpy as np
import pandas as pd

#from sklearn.metrics import log_loss
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm
import os
#os.environ["CUDA_VISIBLE_DEVICES"]="0"
import gc
import random

import warnings
warnings.filterwarnings("ignore")
import scipy.stats as sp
import joblib

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import RidgeClassifier
# Import keras models architectures
from keras_models_arch import get_model_simple, get_model_cnn
import tensorflow.keras.backend as K
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.callbacks import LearningRateScheduler#, EarlyStopping
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

#Import configured paths
from config_paths import processed_data_path, model_path, RAW_DATA_PATH#, subs_path


if not os.path.exists(model_path):
   os.makedirs(model_path)

def set_seed(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    # session_conf = tf.ConfigProto(intra_op_parallelism_threads=num_cores,
#                                       inter_op_parallelism_threads=num_cores, 
#                                       allow_soft_placement=True,
#                                       device_count = {'CPU' : num_CPU, 'GPU' : num_GPU})

#         sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
#         K.set_session(sess)


######################
#Load data
metadata = pd.read_csv(RAW_DATA_PATH + "metadata.csv", index_col="sample_id")
submission=pd.read_csv('submission_format.csv')
train_labels = pd.read_csv(RAW_DATA_PATH + "train_labels.csv", index_col="sample_id")
val_labels = pd.read_csv(RAW_DATA_PATH + "val_labels.csv", index_col="sample_id")


train_files = metadata[metadata["split"] == "train"]["features_path"].to_dict()
val_files = metadata[metadata["split"] == "val"]["features_path"].to_dict()
test_files = metadata[metadata["split"] == "test"]["features_path"].to_dict()

print("Number of training samples: ", len(train_files))
print("Number of validation samples: ", len(val_files))
print("Number of testing samples: ", len(test_files))


# Load datasets
train_img_data1=np.load(processed_data_path+'DS1_train.npz')['a']
test_img_data1=np.load(processed_data_path+'DS1_test.npz')['a']
train_img_data1b=np.load(processed_data_path+'DS1s1_train.npz')['a']
test_img_data1b=np.load(processed_data_path+'DS1s1_test.npz')['a']
train_img_data2=np.load(processed_data_path+'DS2_train.npz')['a']
test_img_data2=np.load(processed_data_path+'DS2_test.npz')['a']
gc.collect()


# Read One extra feature, whether sample is derivatized or not - train
der=[]
for key in tqdm(train_files):
    if metadata.loc[metadata.index==key,'derivatized'].values[0]==1:
        der.append(1)
    else:
        der.append(0)
der=np.array(der)


# Read labels
labels=np.zeros(train_labels.shape, dtype='uint8')
counter=0
for key in tqdm(train_files):
    labels[counter, : ] = train_labels.iloc[np.where(train_labels.index==
          train_files[key][:-4].split('/')[-1])[0][0],:].values
    counter += 1
# make list for stratified split
labelsOne=[('').join(x.astype(str)) for x in labels]
print(labels.sum(), train_labels.values.sum(),
      np.abs(train_labels.values-labels).sum())


# Read One extra feature, whether sample is derivatized or not - test
dertest=[]
for key in tqdm(submission.sample_id.values):
    if metadata.loc[metadata.index==key,'derivatized'].values[0]==1:
        dertest.append(1)
    else:
        dertest.append(0)
dertest=np.array(dertest)

# make list for stratified split
labelsOnetest=[('').join(x.astype(str)) for x in val_labels.values]


# EXTEND TRAIN DATA WITH PUPLIC LB 
train_img_data1=np.concatenate((train_img_data1, 
                                test_img_data1[:len(val_labels),...]))
train_img_data1b=np.concatenate((train_img_data1b, 
                                 test_img_data1b[:len(val_labels),...]))
train_img_data2=np.concatenate((train_img_data2, 
                                test_img_data2[:len(val_labels),...]))
labels=np.concatenate((labels, val_labels))
der=np.concatenate((der, dertest[:len(val_labels),...]))
labelsOne=np.concatenate((np.array(labelsOne), 
                          np.array(labelsOnetest)[:len(val_labels),...]))

print(train_img_data1.shape, train_img_data1b.shape, train_img_data2.shape, 
      labels.shape, der.shape, labelsOne.shape)
######################



######################

# make a statistical features dataset for logistic regression model
data_maxes2=np.concatenate((
        np.mean(train_img_data1,-1).astype('float')/255-0.5,
        np.std(train_img_data1,-1).astype('float')/255-0.5,
        sp.skew(train_img_data1,-1).astype('float')/255-0.5,
        np.mean(train_img_data1,-2).astype('float')/255-0.5, 
        np.expand_dims(der,-1)
                    ),1)

model_name='PH2L1Rlogreg'
allfolds=[]
oof=np.zeros((len(data_maxes2), 9))
NUM_FOLDS=5
all_test_preds=np.zeros((submission.shape[0],9))
skf = StratifiedKFold(n_splits=NUM_FOLDS, random_state=42, shuffle=True)
fold=0
for train_index, test_index in skf.split(data_maxes2, labelsOne):
    test_preds=[]
    X_train, X_test = data_maxes2[train_index,:], data_maxes2[test_index,:]
    y_train = labels[train_index].astype(float)
    y_test = labels[test_index].astype(float)


    loss_pre_class=[]
    for i in range(y_train.shape[1]):
        clf = LogisticRegression(random_state=0, max_iter=1000, tol=0.001, C=80.0).fit(X_train, y_train[:,i])
        # Save model
        joblib.dump(clf, model_path+model_name+'_fold'+str(fold)+'_col'+str(i))
    fold+=1
######################


######################
# make a statistical features dataset for first set of random forest and 
# ridge classification  model
data_maxes2=np.concatenate((
        np.mean(train_img_data1,-1).astype('float')/255,
        np.max(train_img_data1,-1).astype('float')/255,
        np.std(train_img_data1,-1).astype('float')/255,
        np.mean(train_img_data1,-2).astype('float')/255, 
        np.expand_dims(der,-1)),1)

model_name='PH2L1Rrf'
allfolds=[]
oof=np.zeros((len(data_maxes2),9))
all_test_preds=np.zeros((submission.shape[0],9))
all_feat_imps=[]
fold=0
for train_index, test_index in skf.split(data_maxes2, labelsOne):
    test_preds=[]
    # print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = data_maxes2[train_index,:], data_maxes2[test_index,:]
    y_train, y_test = labels[train_index].astype(float), labels[test_index].astype(float)


    loss_pre_class=[]
    for i in range(y_train.shape[1]):
        rf = RandomForestClassifier(n_estimators=300, criterion='gini', 
            max_depth=None, min_samples_split=2, min_samples_leaf=1, 
            min_weight_fraction_leaf=0.0, max_features='sqrt', 
            max_leaf_nodes=None, min_impurity_decrease=0.0, bootstrap=True, 
            oob_score=False, n_jobs=8, random_state=42, verbose=0, 
            warm_start=False, class_weight=None, ccp_alpha=0.0, max_samples=None)
        rf.fit(X_train, y_train[:,i])    

        # Save model
        joblib.dump(rf, model_path+model_name+'_fold'+str(fold)+'_col'+str(i))
        rf=joblib.load(model_path+model_name+'_fold'+str(fold)+'_col'+str(i))

        all_feat_imps.append(rf.feature_importances_)
    fold+=1
all_feat_imps=np.vstack(all_feat_imps)

model_name='PH2L1Rridge'
cols=np.where(np.mean(all_feat_imps,0)>0.0001)[0]
# cols=np.arange(data_maxes2.shape[1])
print(len(cols))

ridge = RidgeClassifier(alpha=0.1, fit_intercept=True, normalize=False, 
                        copy_X=True, max_iter=None, tol=0.01, class_weight=None, 
                        solver='auto',  random_state=42)
allfolds=[]
oof=np.zeros((len(data_maxes2),9))
all_test_preds=np.zeros((submission.shape[0],9))
fold=0
for train_index, test_index in skf.split(data_maxes2, labelsOne):
    test_preds=[]
    X_train = data_maxes2[train_index,:][:,cols]
    X_test  = data_maxes2[test_index,:][:,cols]
    y_train = labels[train_index].astype(float)
    y_test  = labels[test_index].astype(float)

    loss_pre_class=[]
    for i in range(y_train.shape[1]):
        model = ridge.fit(X_train, y_train[:,i])#,

        # Save model
        joblib.dump(model, model_path+model_name+'_fold'+str(fold)+'_col'+str(i))
    fold+=1
######################


######################
# make a statistical features dataset for first set of random forest and 
# ridge classification  model
data_maxes2=np.concatenate((
        np.mean(train_img_data1b,-1).astype('float')/255,
        np.max(train_img_data1b,-1).astype('float')/255,
        np.std(train_img_data1b,-1).astype('float')/255,
        np.mean(train_img_data1b,-2).astype('float')/255, 
        np.expand_dims(der,-1)),1)

model_name='PH2L1Rrfb'
allfolds=[]
oof=np.zeros((len(data_maxes2),9))
all_test_preds=np.zeros((submission.shape[0],9))
all_feat_impsb=[]
fold=0
for train_index, test_index in skf.split(data_maxes2, labelsOne):
    test_preds=[]
    X_train = data_maxes2[train_index,:]
    X_test  = data_maxes2[test_index,:]
    y_train = labels[train_index].astype(float)
    y_test  = labels[test_index].astype(float)

    loss_pre_class=[]
    for i in range(y_train.shape[1]):
        rf = RandomForestClassifier(n_estimators=300, criterion='gini', 
                max_depth=None, min_samples_split=2, min_samples_leaf=1, 
                min_weight_fraction_leaf=0.0, max_features='sqrt', 
                max_leaf_nodes=None, min_impurity_decrease=0.0, bootstrap=True, 
                oob_score=False, n_jobs=8, random_state=42, verbose=0, 
                warm_start=False, class_weight=None, ccp_alpha=0.0, max_samples=None)

        rf.fit(X_train, y_train[:,i])    

        # Save model
        joblib.dump(rf, model_path+model_name+'_fold'+str(fold)+'_col'+str(i))
        rf=joblib.load(model_path+model_name+'_fold'+str(fold)+'_col'+str(i))
        
        all_feat_impsb.append(rf.feature_importances_)
    fold+=1
all_feat_impsb=np.vstack(all_feat_impsb)


model_name='PH2L1Rridge0'
#cols=np.where(np.mean(all_feat_impsb,0)>0.00005)[0]
cols=np.where(np.mean(all_feat_impsb,0)>0.0002)[0]
print(len(cols))

ridge = RidgeClassifier(alpha=0.2, fit_intercept=True, normalize=False, 
# ridge = RidgeClassifier(alpha=0.25, fit_intercept=True, normalize=False, 
                        copy_X=True, max_iter=None, tol=0.01, class_weight=None, 
                        solver='auto',  random_state=42)

allfolds=[]
oof=np.zeros((len(data_maxes2),9))
all_test_preds=np.zeros((submission.shape[0],9))
fold=0
for train_index, test_index in skf.split(data_maxes2, labelsOne):
    test_preds=[]
    X_train = data_maxes2[train_index,:][:,cols]
    X_test  = data_maxes2[test_index,:][:,cols]
    y_train = labels[train_index].astype(float)
    y_test  = labels[test_index].astype(float)

    loss_pre_class=[]
    for i in range(y_train.shape[1]):
        model = ridge.fit(X_train, y_train[:,i])#,

        # Save model
        joblib.dump(model, model_path+model_name+'_fold'+str(fold)+'_col'+str(i))
    fold+=1
######################


######################
# Train Simple Keras model
def train_model(epochs=20, SNAPSHOTS=1, BATCH_SIZE=16, 
                lr_0=0.0001, SEEDNUM=42, verbose=1):

    # LR schedule
    def _cosine_anneal_schedule(t):
        cos_inner = np.pi * (t % (epochs // SNAPSHOTS))
        cos_inner /= epochs // SNAPSHOTS
        cos_out = np.cos(cos_inner) + 1
        return float(lr_0 / 2 * cos_out)
    lr_anneal = LearningRateScheduler(schedule=_cosine_anneal_schedule, verbose=False)
    np.array([_cosine_anneal_schedule(t) for t in range( int(epochs/SNAPSHOTS))])

    # Data Preprocess    
    data_maxes2=np.concatenate((
            np.mean(train_img_data,-1).astype('float')/255,
            np.max(train_img_data,-1).astype('float')/255, 
            np.std(train_img_data,-1).astype('float')/255,
            np.mean(train_img_data,-2).astype('float')/255, 
            np.std(train_img_data,-2).astype('float')/255, 
            np.expand_dims(der,-1)),1)
             

    # training
#    allfolds=[]
    skf = StratifiedKFold(n_splits=NUM_FOLDS, random_state=42, shuffle=True)
#    oof=np.zeros((len(data_maxes2),9))
#    all_test_preds=np.zeros((submission.shape[0],9))
    fold=0
    for train_index, test_index in skf.split(data_maxes2, labelsOne):
#        loss_per_class=[]
        set_seed(seed=SEEDNUM)
        # test_preds=[]
        # print("TRAIN:", train_index, "TEST:", test_index)
        X_train, X_test = data_maxes2[train_index,:], data_maxes2[test_index,:]
        y_train = labels[train_index].astype(float)
        y_test  = labels[test_index].astype(float)
    
        model=get_model_simple(dim=(data_maxes2.shape[1],))     
#             model.compile(loss=keras.losses.binary_crossentropy, metrics=["accuracy"], optimizer=keras.optimizers.Adam(learning_rate=lr_0, beta_1=0.8))
        model.compile(loss=keras.losses.binary_crossentropy, metrics=['binary_crossentropy'],#metrics=["accuracy"],
                        optimizer=keras.optimizers.SGD(learning_rate=lr_0, momentum=0.9))#, clipvalue=1.0))#,  nesterov=True))


        model.fit(X_train,  y_train,
                            batch_size=BATCH_SIZE,
                            # epochs=40,
                            epochs=epochs,
                            shuffle=True,
                            validation_data=(X_test, y_test),
                            callbacks=[#lr_sch, 
            lr_anneal,
            tf.keras.callbacks.ModelCheckpoint(model_path+model_name+
                           '_fold'+str(fold)+'_seed'+str(SEEDNUM)+'.h5',
                            monitor='val_loss', verbose=0, save_best_only=True,
                            save_weights_only=True, mode='auto', 
                            save_freq='epoch')
            # ,EarlyStopping(monitor="val_loss", min_delta=0, patience=30, verbose=0,
            #                mode="auto", baseline=None, restore_best_weights=False)
                            ],
                            verbose = verbose )
        del  model
        K.clear_session()
        gc.collect()
        fold+=1


model_name='PH2L1Rgpukeras'
oof_simpleA, all_test_preds_simpleA=[],[] 
num_mz = 500    
train_img_data=train_img_data2[:,:num_mz,:]
test_img_data=test_img_data2[:,:num_mz,:]

train_model(epochs=80, SNAPSHOTS=1, BATCH_SIZE=32, lr_0=0.01, SEEDNUM=42, verbose=0)
train_model(epochs=80, SNAPSHOTS=1, BATCH_SIZE=32, lr_0=0.01, SEEDNUM=2022, verbose=0)
train_model(epochs=80, SNAPSHOTS=1, BATCH_SIZE=32, lr_0=0.01, SEEDNUM=22, verbose=0)
######################


######################
# Train CNN Keras model
def train_model_cnn(model_type, epochs=20, SNAPSHOTS=1, BATCH_SIZE=16, 
                    lr_0=0.0001, SEEDNUM=42, verbose=1):

    # LR schedule
#    epochspersnapshot = int(epochs/SNAPSHOTS)
    def _cosine_anneal_schedule(t):
        cos_inner = np.pi * (t % (epochs // SNAPSHOTS))
        cos_inner /= epochs // SNAPSHOTS
        cos_out = np.cos(cos_inner) + 1
        return float(lr_0 / 2 * cos_out)
    lr_anneal = LearningRateScheduler(schedule=_cosine_anneal_schedule, verbose=False)

#    # Data Preprocess  
    data_maxes2 = train_img_data.astype('float')
    
    fold=0
    for train_index, test_index in skf.split(data_maxes2, labelsOne):
        set_seed(seed=SEEDNUM)
        X_train, X_test = data_maxes2[train_index,:], data_maxes2[test_index,:]
        y_train = labels[train_index].astype(float) 
        y_test  = labels[test_index].astype(float)
        
        # Preprocess - switch time-ions dimensions
        X_train = np.transpose(X_train.astype('float'), axes = (0,2,1))
        X_test  = np.transpose(X_test.astype('float'), axes = (0,2,1))
        
        
        # Load model
        if model_type=='CNN0':
            model=get_model_cnn(9, dim=(data_maxes2.shape[1:][::-1]), EFF=0 )
        elif model_type=='CNN1':
            model=get_model_cnn(9, dim=(data_maxes2.shape[1:][::-1]), EFF=1 )
        elif model_type=='CNN2':
            model=get_model_cnn(9, dim=(data_maxes2.shape[1:][::-1]), EFF=2 )

        # model.compile(loss=keras.losses.binary_crossentropy, metrics=["accuracy"], optimizer=keras.optimizers.Adam(learning_rate=lr_0, beta_1=0.8))
        model.compile(loss=keras.losses.binary_crossentropy, metrics=["accuracy"], 
              optimizer=keras.optimizers.SGD(learning_rate=lr_0, momentum=0.9))#, clipvalue=1.0))#,  nesterov=True))

        model.fit(X_train,  y_train,
                batch_size=BATCH_SIZE,
                # epochs=40,
                epochs=epochs,
                shuffle=True,
                validation_data=(X_test, y_test),
                callbacks=[#lr_sch, 
                    lr_anneal,
                    tf.keras.callbacks.ModelCheckpoint(model_path+model_name+
                       '_fold'+str(fold)+'_seed'+str(SEEDNUM)+'.h5',
                        monitor='val_loss', verbose=0, save_best_only=True,
                        save_weights_only=True, mode='auto', 
                        save_freq='epoch')
                ],
                verbose = verbose )
      
        del  model
        K.clear_session()
        gc.collect()
        fold+=1


model_name='PH2L1RgpukerasEFF012'
oof_simpleA, all_test_preds_simpleA=[],[] 
num_mz = 500
train_img_data=train_img_data2[:,:num_mz,:]
test_img_data=test_img_data2[:,:num_mz,:]
train_model_cnn(model_type='CNN0', epochs=40, SNAPSHOTS=1, BATCH_SIZE=32, 
                lr_0=0.01, SEEDNUM=42, verbose=1)
train_model_cnn(model_type='CNN1', epochs=40, SNAPSHOTS=1, BATCH_SIZE=32, 
                lr_0=0.01, SEEDNUM=2022, verbose=0)
train_model_cnn(model_type='CNN2', epochs=40, SNAPSHOTS=1, BATCH_SIZE=32, 
                lr_0=0.01, SEEDNUM=22, verbose=0)


