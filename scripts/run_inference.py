# -*- coding: utf-8 -*-

#Imports
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss
from tqdm import tqdm
#import os
#os.environ["CUDA_VISIBLE_DEVICES"]="0"
import gc
import warnings
warnings.filterwarnings("ignore")
import scipy.stats as sp
import joblib

# Import keras models architectures
from keras_models_arch import get_model_simple, get_model_cnn
import tensorflow.keras.backend as K
#Import configured paths
from config_paths import processed_data_path, model_path, subs_path, RAW_DATA_PATH


######################
#Load datasets
metadata = pd.read_csv(RAW_DATA_PATH + "metadata.csv", index_col="sample_id")
val_labels = pd.read_csv(RAW_DATA_PATH + "val_labels.csv", index_col="sample_id")
submission=pd.read_csv(RAW_DATA_PATH+'submission_format.csv')

metadata.groupby('split').count()

train_files = metadata[metadata["split"] == "train"]["features_path"].to_dict()
val_files = metadata[metadata["split"] == "val"]["features_path"].to_dict()
test_files = metadata[metadata["split"] == "test"]["features_path"].to_dict()

print("Number of training samples: ", len(train_files))
print("Number of validation samples: ", len(val_files))
print("Number of testing samples: ", len(test_files))


# Read all test datasets
test_img_data1=np.load(processed_data_path+'DS1_test.npz')['a']
test_img_data1b=np.load(processed_data_path+'DS1s1_test.npz')['a']
test_img_data2=np.load(processed_data_path+'DS2_test.npz')['a']
gc.collect()

# Read One extra feature, whether sample is derivatized or not
dertest=[]
for key in tqdm(submission.sample_id.values):
    if metadata.loc[metadata.index==key,'derivatized'].values[0]==1:
        dertest.append(1)
    else:
        dertest.append(0)
dertest=np.array(dertest)
######################



######################
# make a statistical features dataset for logistic regression model
data_test_maxes2=np.concatenate((
         np.mean(test_img_data1,-1).astype('float')/255-0.5,
         np.std(test_img_data1,-1).astype('float')/255-0.5,
         sp.skew(test_img_data1,-1).astype('float')/255-0.5,
         np.mean(test_img_data1,-2).astype('float')/255-0.5, 
         np.expand_dims(dertest,-1)),1)

model_name='PH2L1Rlogreg'
NUM_FOLDS=5
all_test_preds=np.zeros((submission.shape[0],9))
for fold in range(NUM_FOLDS):
    test_preds=[]
    for i in range(9):
        # Load model for fold and column
        clf=joblib.load(model_path+model_name+'_fold'+str(fold)+'_col'+str(i))
        # make predictions
        test_preds.append(clf.predict_proba(data_test_maxes2)[:,1])
    test_preds=np.stack(test_preds).T   
    all_test_preds += test_preds/NUM_FOLDS
test_logreg=all_test_preds.copy()
print('log reg',np.mean([log_loss(val_labels.values[:,x], 
                      test_logreg[:len(val_labels),x]) for x in range(9)]))
######################


######################
# make a statistical features dataset for first set of random forest and 
# ridge classification  model
data_test_maxes2=np.concatenate((
        np.mean(test_img_data1,-1).astype('float')/255,
        np.max(test_img_data1,-1).astype('float')/255,
        np.std(test_img_data1,-1).astype('float')/255,
        np.mean(test_img_data1,-2).astype('float')/255, 
        np.expand_dims(dertest,-1)),1)

# Use random forest for feature selection for ridge classification model
model_name='PH2L1Rrf'
all_test_preds=np.zeros((submission.shape[0],9))
all_feat_imps=[]
fold=0
for fold in range(NUM_FOLDS):
    for i in range(9):
        rf=joblib.load(model_path+model_name+'_fold'+str(fold)+'_col'+str(i))
        all_feat_imps.append(rf.feature_importances_)
all_feat_imps=np.vstack(all_feat_imps)


model_name='PH2L1Rridge'
cols=np.where(np.mean(all_feat_imps,0)>0.0001)[0]
print(len(cols))
all_test_preds=np.zeros((submission.shape[0],9))
for fold in range(NUM_FOLDS):    
    test_preds=[]
    for i in range(9):
        # Load model for fold and column
        model=joblib.load(model_path+model_name+'_fold'+str(fold)+'_col'+str(i))
        # make predictions
        test_preds.append(model.predict(data_test_maxes2[:,cols]))
    test_preds=np.stack(test_preds).T
    all_test_preds += test_preds/NUM_FOLDS   
all_test_preds_ridge=all_test_preds.copy()
print('ridge reg',np.mean([log_loss(val_labels.values[:,x], 
                all_test_preds_ridge[:len(val_labels),x]) for x in range(9)]))
######################


######################
# make a statistical features dataset for second set of random forest and 
# ridge classification  model
data_test_maxes2=np.concatenate((
        np.mean(test_img_data1b,-1).astype('float')/255,
        np.max(test_img_data1b,-1).astype('float')/255,
        np.std(test_img_data1b,-1).astype('float')/255,
        np.mean(test_img_data1b,-2).astype('float')/255, 
        np.expand_dims(dertest,-1)),1)

model_name='PH2L1Rrfb'
all_feat_impsb=[]
for fold in range(NUM_FOLDS):    
    for i in range(9):
        rf=joblib.load(model_path+model_name+'_fold'+str(fold)+'_col'+str(i))
        all_feat_impsb.append(rf.feature_importances_)
all_feat_impsb=np.vstack(all_feat_impsb)


model_name='PH2L1Rridge0'
cols=np.where(np.mean(all_feat_impsb,0)>0.0002)[0]
print(len(cols))
all_test_preds=np.zeros((submission.shape[0],9))
for fold in range(NUM_FOLDS):
    test_preds=[]
    for i in range(9):
        # Load model for fold and column
        model=joblib.load(model_path+model_name+'_fold'+str(fold)+'_col'+str(i))   
        # make predictions
        test_preds.append(model.predict(data_test_maxes2[:,cols]))
    test_preds=np.stack(test_preds).T
    all_test_preds += test_preds/NUM_FOLDS   
all_test_preds_ridge0=all_test_preds.copy()
print('ridge0 reg',np.mean([log_loss(val_labels.values[:,x], 
             all_test_preds_ridge0[:len(val_labels),x]) for x in range(9)]))
######################


#####################
def infer_model(SEEDNUM=42):

    # Get statistical features
    data_test_maxes2=np.concatenate((
            np.mean(test_img_data,-1).astype('float')/255,
            np.max(test_img_data,-1).astype('float')/255, 
            np.std(test_img_data,-1).astype('float')/255,
            np.mean(test_img_data,-2).astype('float')/255, 
            np.std(test_img_data,-2).astype('float')/255, 
            np.expand_dims(dertest,-1)),1)    

    all_test_preds=np.zeros((submission.shape[0],9))
    for fold in range(NUM_FOLDS):
    
        model=get_model_simple(dim=(data_test_maxes2.shape[1],))     
        model.load_weights(model_path+model_name+'_fold'+str(fold)+
                           '_seed'+str(SEEDNUM)+'.h5')

        test_preds=model.predict(data_test_maxes2)
        all_test_preds += test_preds/NUM_FOLDS                  
            
        del  model
        K.clear_session()
        gc.collect()
    return all_test_preds


#model_name='PH2L1Rkeras'
model_name='PH2L1Rgpukeras'

#all_test_preds_simpleA=[]
#for num_mz in [500]:
num_mz=500
#    print(num_mz)
#test_img_data=test_img_data2.copy()
test_img_data=test_img_data2[:,:num_mz,:]

all_test_preds_simple = infer_model(SEEDNUM=42)
print('simple keras',np.mean([log_loss(val_labels.values[:,x], 
             all_test_preds_simple[:len(val_labels),x]) for x in range(9)]))
all_test_preds_simple2 = infer_model(SEEDNUM=2022)
print('simple keras 2',np.mean([log_loss(val_labels.values[:,x], 
             all_test_preds_simple2[:len(val_labels),x]) for x in range(9)]))
all_test_preds_simple3 = infer_model(SEEDNUM=22)
print('simple keras 3',np.mean([log_loss(val_labels.values[:,x], 
             all_test_preds_simple3[:len(val_labels),x]) for x in range(9)]))
#all_test_preds_simpleA.append([all_test_preds_simple,all_test_preds_simple2,all_test_preds_simple3])

#all_test_preds = np.mean(np.stack(all_test_preds_simpleA[0]),0)
#all_test_preds_keras=all_test_preds.copy()
all_test_preds_keras=(all_test_preds_simple+all_test_preds_simple2+
                      all_test_preds_simple3)/3
######################


#####################
def infer_model_cnn(model_type, SEEDNUM=42):

    all_test_preds=np.zeros((submission.shape[0],9))
    for fold in range(NUM_FOLDS):
        # Load model Arch
        if model_type=='CNN0':
            model=get_model_cnn(9, dim=(test_img_data.shape[1:][::-1]), EFF=0 )
        elif model_type=='CNN1':
            model=get_model_cnn(9, dim=(test_img_data.shape[1:][::-1]), EFF=1 )
        elif model_type=='CNN2':
            model=get_model_cnn(9, dim=(test_img_data.shape[1:][::-1]), EFF=2 )
#        elif model_type=='CNN3':
#            model=get_model_cnn(9, dim=(test_img_data.shape[1:][::-1]), EFF=3 )            
        
        #Load model weights
        model.load_weights(model_path+model_name+'_fold'+str(fold)+
                           '_seed'+str(SEEDNUM)+'.h5')

        # Predictions - switch time-ions
        test_preds=model.predict(np.transpose(test_img_data, axes = (0,2,1)))
        all_test_preds += test_preds/NUM_FOLDS                  
            
        del  model
        K.clear_session()
        gc.collect()
    return all_test_preds


#model=get_model_cnn()

model_name='PH2L1RgpukerasEFF012'
#all_test_preds_simpleA=[]
#for num_mz in [500]:
num_mz = 500
#print(num_mz)
test_img_data=test_img_data2[:,:num_mz,:]
all_test_preds_simple = infer_model_cnn(model_type='CNN0', SEEDNUM=42)
print('cnn keras',np.mean([log_loss(val_labels.values[:,x],
            all_test_preds_simple[:len(val_labels),x]) for x in range(9)]))
all_test_preds_simple2 = infer_model_cnn(model_type='CNN1', SEEDNUM=2022)
print('cnn keras 2',np.mean([log_loss(val_labels.values[:,x], 
            all_test_preds_simple2[:len(val_labels),x]) for x in range(9)]))
all_test_preds_simple3 = infer_model_cnn(model_type='CNN2', SEEDNUM=22)
print('cnn keras 3',np.mean([log_loss(val_labels.values[:,x], 
            all_test_preds_simple3[:len(val_labels),x]) for x in range(9)]))
#all_test_preds_simpleA.append([all_test_preds_simple,all_test_preds_simple2,all_test_preds_simple3])
gc.collect()

#all_test_preds_keras_cnn=np.mean(np.stack(all_test_preds_simpleA[0]),0)
all_test_preds_keras_cnn=(all_test_preds_simple+all_test_preds_simple2+
                          all_test_preds_simple3)/3
######################

                          
# Make ensemble - Final predictions
ensemble=(all_test_preds_keras_cnn+all_test_preds_keras+test_logreg+
          (all_test_preds_ridge+all_test_preds_ridge0)/2)/4
print('ensemble',np.mean([log_loss(val_labels.values[:,x], 
            ensemble[:len(val_labels),x]) for x in range(9)]))
# create submission file
submission2=submission.copy()
submission2.iloc[:,1:] = ensemble
submission2.to_csv(subs_path+'L1_KKcnnLrRc_v18.csv', index=False)

#print(np.mean([log_loss(val_labels.values[:,x], all_test_preds_keras[:len(val_labels),x]) for x in range(9)]))
#print(np.mean([log_loss(val_labels.values[:,x], test_logreg[:len(val_labels),x]) for x in range(9)]))
#print(np.mean([log_loss(val_labels.values[:,x], all_test_preds_ridge[:len(val_labels),x]) for x in range(9)]))
#print(np.mean([log_loss(val_labels.values[:,x], all_test_preds_ridge0[:len(val_labels),x]) for x in range(9)]))
