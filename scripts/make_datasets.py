# coding: utf-8

# Import libraries
import numpy as np
import pandas as pd
from tqdm import tqdm
import os
import gc

# set helper functions
#def array2length(array1D, length=500):
#    lenarray=len(array1D)
#    t=lenarray/length
#    # ps=[min(int(round(x*t)),lenarray-1) for x in range(length)]
#    ps=[int(x*t) for x in range(length)]
#    return array1D[ps]

def array2lengthM(array1D, length=500):
    """
    Turns an array of any length to length equal to length variable.
    If original length > desired length, then select the higher value.
    Args:
        array1D: 1 dimension numpy array 
        length: desired output length
    Returns:
        1 dimension numpy array
    """
    t=len(array1D)/length
    ps=[int(x*t) for x in range(length)]
    ps2=ps[1:]+[ps[-1]]
    
    return np.max(np.stack((array1D[ps],array1D[ps2])),0)
    
#def scale_mass(array):
#    narray=array.copy()
#    return ((narray/narray.max())*255).astype('uint8')
def scale_mass_p(array, p=99.9):
    """
    Upclip numpy array to p percentile. 0-255 scale and convert to uint8 dtype.
    Args:
        array: a numpy array of any size
        p: percentile for upclipping array
    Returns:
        a numpy array of same size 
    """    
    narray=array.copy()
    perMax=np.percentile(narray, p)
    # print(perMax,narray.max())
    narray=np.clip(narray, narray.min(), perMax)
    return ((narray/perMax)*255).astype('uint8')    

#def file_features(paths):
#    data = np.zeros((len(paths),7))
#    j=0
#    for path in tqdm(paths):
#        tf = pd.read_csv(path)
#        tf['mass_int']=tf['mass'].round(0).astype(int)
#
#        data[j,:] = np.array([len(tf),len(np.unique(tf['mass_int'])),tf.intensity.min(),np.log1p(tf.intensity.max())**0.5,tf.time.max(),tf.time.min(),tf.time.max()-tf.time.min()])
#        j+=1
#        # break
#    return data


#def remove_background_abundance(df):
#    """
#    Subtracts minimum abundance value
#    Args:
#        df: dataframe with 'm/z' and 'abundance' columns
#    Returns:
#        dataframe with minimum abundance subtracted for all observations
#    """
#
#    df["intensity"] = df.groupby(["mass_int"])["intensity"].transform(
#        lambda x: (x - x.min())
#    )
#    return df

def smooth_ar2d(ar):
    """
    Smooths a 2-d numpy array in last dimension 
    Args:
        ar: a 2-d numpy array
    Returns:
        a smoothed numpy array of same size 
    """        
    retar=np.zeros_like(ar)
    retar[:,1:-1] = ((ar[:,1:-1].astype(float)+ar[:,:-2].astype(float)+ar[:,2:].astype(float))/3)#.astype('uint8')
    retar[...,0]=retar[...,1]
    retar[...,-1]=retar[...,-2]
    return retar
        
def create_dataset(paths, timesteps=500, ions=500, sq=True, log=True, 
                   smooth=True, smooth_times=2, scale=True, perc=99.9):
    
    """
    Creates a 3-d dataset of ions abundance with length equal to number of 
    paths (samples).
    Args:
        paths: a list of strings, paths of sample csv files
        timesteps: the number of desired length of every sample (last dimension)
        ions: the number of ions to be used in dataset (second dimension)
        sq: boolean, whether square root transform will be applied.
        log: boolean, whether log transform will be applied.
        smooth: boolean, whether log transform will be applied.
        smooth_times: how many times smooth will be applied.
        scale: boolean, whether scale will be applied.
        perc: percintile for upclipping sample. Meaningful only when scale 
            is set to True.
        
    Returns:
        a 3-d numpy array of shape (num samples, num ions, num timesteps)
    """  
    
    img_data=np.zeros((len(paths), ions, timesteps), dtype='uint8')
    j=0
    for path in tqdm(paths):
        tf = pd.read_csv(path)
        tf['mass_int']=tf['mass'].round(0).astype(int)
        
        # tf = remove_background_abundance(tf)
        if sq==True:
            tf.intensity= tf.intensity.values**0.5
        if log==True:
            tf.intensity=np.log1p(tf.intensity.values)
            


        img=np.zeros((ions,timesteps))
        uniquemassint=np.unique(tf.mass_int)
        for i in range(ions):
            if i in uniquemassint:
                a = tf.intensity.loc[tf.mass_int==i]
                img[i,:] = array2lengthM(a.values,  length=timesteps )

        if smooth==True:
            for s in range(smooth_times):
                img = smooth_ar2d(img)
                
        if scale==True:
            img=scale_mass_p(img, p=perc)
            
        img_data[j,...]=img
        j+=1
        # break
    return img_data

from config_paths import RAW_DATA_PATH, processed_data_path
#RAW_DATA_PATH = ''

metadata = pd.read_csv(RAW_DATA_PATH + "metadata.csv", index_col="sample_id")
metadata.head()


train_files = metadata[metadata["split"] == "train"]["features_path"].to_dict()
val_files = metadata[metadata["split"] == "val"]["features_path"].to_dict()
test_files = metadata[metadata["split"] == "test"]["features_path"].to_dict()

print("Number of training samples: ", len(train_files))
print("Number of validation samples: ", len(val_files))
print("Number of testing samples: ", len(test_files))



train_labels = pd.read_csv(RAW_DATA_PATH + "train_labels.csv", index_col="sample_id")
gc.collect()


train_paths=[RAW_DATA_PATH+'train_features/'+train_files[key] for key in train_files]


submission=pd.read_csv('submission_format.csv')
val_files=os.listdir(RAW_DATA_PATH+'val_features/val_features')
val_test_paths=[RAW_DATA_PATH+'val_features/val_features/'+x+'.csv' if x+'.csv' in val_files else RAW_DATA_PATH+'test_features/'+x+'.csv' for x in submission.sample_id.values ]



print('make DS2 train and test datasets')
train_img_data=create_dataset(train_paths, timesteps=500, ions=600, sq=True, smooth=True, 
                              smooth_times=2, log=False, scale=True, perc=99.8)
test_img_data=create_dataset(val_test_paths, timesteps=500, ions=600, sq=True, smooth=True, 
                             smooth_times=2, log=False, scale=True, perc=99.8)
np.savez_compressed(processed_data_path+'DS2_train', a=train_img_data)
np.savez_compressed(processed_data_path+'DS2_test', a=test_img_data)



print('make DS1 train and test datasets')
train_img_data=create_dataset(train_paths, timesteps=500, ions=600, sq=True, smooth=False, 
                              smooth_times=2, log=True, scale=True, perc=99.9)
test_img_data=create_dataset(val_test_paths, timesteps=500, ions=600, sq=True, smooth=False, 
                             smooth_times=2, log=True, scale=True, perc=99.9)

np.savez_compressed(processed_data_path+'DS1_train', a=train_img_data)
np.savez_compressed(processed_data_path+'DS1_test', a=test_img_data)



print('make DS1s1 train and test datasets')
train_img_data=create_dataset(train_paths, timesteps=500, ions=600, sq=True, smooth=True, 
                              smooth_times=1, log=True, scale=True, perc=99.9)
test_img_data=create_dataset(val_test_paths, timesteps=500, ions=600, sq=True, smooth=True, 
                             smooth_times=1, log=True, scale=True, perc=99.9)
np.savez_compressed(processed_data_path+'DS1s1_train', a=train_img_data)
np.savez_compressed(processed_data_path+'DS1s1_test', a=test_img_data)

