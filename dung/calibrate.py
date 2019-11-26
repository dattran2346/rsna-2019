import pandas as pd
import numpy as np
import torch
import tqdm
import os
import tensorflow as tf

# supress warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# use gpu
physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

if __name__ == "__main__":
    if not os.path.exists('calib_pred'):
        os.makedirs('calib_pred')
    val_files = ['EfficientnetB2_yp_valid_fold0_epoch5.csv',
            'EfficientnetB2_yp_valid_fold1_epoch11.csv',
            'EfficientnetB2_yp_valid_fold2_epoch11.csv',
            'EfficientnetB2_yp_valid_fold3_epoch11.csv',
            'EfficientnetB2_yp_valid_fold4_epoch11.csv',
            'EfficientnetB5_yp_valid_fold0_epoch11.csv',
            'EfficientnetB5_yp_valid_fold1_epoch11.csv',
            'EfficientnetB5_yp_valid_fold2_epoch11.csv',
            'EfficientnetB5_yp_valid_fold3_epoch11.csv',
            'EfficientnetB5_yp_valid_fold4_epoch10.csv']
    for fold in range(5):
        for mfile in val_files:
            if 'fold{}'.format(fold) not in mfile:
                continue
            csv_name = 'raw_pred/{}'.format(mfile)
            df = pd.read_csv(csv_name).reindex(['image', 'any', 'epidural', \
                                'intraparenchymal', 'intraventricular', 'subarachnoid','subdural'], axis=1)

            # load metadata to get studyids (studyids in metadata df must be sorted)
            metadata = pd.read_csv('dataset/train_metadata.csv')
            metadata = metadata[metadata['image'].isin(list(df['image'].values))]
            metadata.drop([ 'any','epidural','intraparenchymal','intraventricular', 'subarachnoid', \
                        'subdural','BrainPresence'], axis=1, inplace=True)

            # merge to sort id
            df = pd.merge(metadata, df, how='inner', on='image')
            df.drop(['ImagePositionPatient_2','PatientID','StudyInstanceUID','WindowCenter','WindowWidth'],\
            axis=1,inplace=True)

            # extract slices suiting for window cnn
            features_idcs = []
            studyid_values = metadata['StudyInstanceUID'].values
            for i in range(2,len(studyid_values)-2):
                if set(studyid_values[i-2:i+3]) == set(studyid_values[i+3:i-2:-1]):
                    features_idcs.append(i)

            features = np.ndarray(shape=(len(features_idcs), 5, 6, 1))
            prediction = np.array(df.iloc[:,1:].values, dtype=float)
            for i,idcs in enumerate(features_idcs):
                features[i,:,:,0] = prediction[idcs-2:idcs+3,:]

            window_cnn = tf.keras.models.load_model(f'windows/studyid_fold{fold}.h5')
            windowed_prediction = window_cnn.predict(features)

            prediction[features_idcs] = windowed_prediction
            df.iloc[:,1:7] = prediction

            df.to_csv(csv_name.replace('raw_pred', 'calib_pred'), index=False)

    test_files = ['EfficientnetB2_yp_test_fold0_epoch5.csv',
            'EfficientnetB2_yp_test_fold1_epoch11.csv',
            'EfficientnetB2_yp_test_fold2_epoch11.csv',
            'EfficientnetB2_yp_test_fold3_epoch11.csv',
            'EfficientnetB2_yp_test_fold4_epoch11.csv',
            'EfficientnetB5_yp_test_fold0_epoch11.csv',
            'EfficientnetB5_yp_test_fold1_epoch11.csv',
            'EfficientnetB5_yp_test_fold2_epoch11.csv',
            'EfficientnetB5_yp_test_fold3_epoch11.csv',
            'EfficientnetB5_yp_test_fold4_epoch10.csv']
    for fold in range(5):
        for mfile in test_files:
            if 'fold{}'.format(fold) not in mfile:
                continue
            csv_name = 'raw_pred/{}'.format(mfile)
            df = pd.read_csv(csv_name).reindex(['image', 'any', 'epidural', \
                                'intraparenchymal', 'intraventricular', 'subarachnoid','subdural'], axis=1)

            # load metadata to get studyids (studyids in metadata df must be sorted)
            metadata = pd.read_csv('dataset/test_metadata.csv')
            metadata = metadata[metadata['image'].isin(list(df['image'].values))]

            # merge to sort id
            df = pd.merge(metadata, df, how='inner', on='image')
            df.drop(['ImagePositionPatient_2','PatientID','StudyInstanceUID','WindowCenter','WindowWidth'],\
            axis=1,inplace=True)

            # extract slices suiting for window cnn
            features_idcs = []
            studyid_values = metadata['StudyInstanceUID'].values
            for i in range(2,len(studyid_values)-2):
                if set(studyid_values[i-2:i+3]) == set(studyid_values[i+3:i-2:-1]):
                    features_idcs.append(i)

            features = np.ndarray(shape=(len(features_idcs), 5, 6, 1))
            prediction = np.array(df.iloc[:,1:].values, dtype=float)
            for i,idcs in enumerate(features_idcs):
                features[i,:,:,0] = prediction[idcs-2:idcs+3,:]

            window_cnn = tf.keras.models.load_model(f'windows/studyid_fold{fold}.h5')
            windowed_prediction = window_cnn.predict(features)

            prediction[features_idcs] = windowed_prediction
            df.iloc[:,1:7] = prediction

            df.to_csv(csv_name.replace('raw_pred', 'calib_pred'), index=False)