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

for fold in [0,1,2,3,4]:
    for csv_name_family, studypatient in zip(['3_seresnext50', '4_b3', '4_b4'], ['study', 'patient', 'patient']):
        csv_name = 'datnt_version' + csv_name_family + f'_fold{fold}'
        valtest = 'test'

        # input 1 df with format image - disease - disease - disease - disease - disease - disease
        df = pd.read_csv(f'../datnt/stage2result/{csv_name}.csv').reindex(['image', 'any', 'epidural', \
                                'intraparenchymal', 'intraventricular', 'subarachnoid','subdural'], axis=1)

        # load metadata to get studyids (studyids in metadata df must be sorted)
        metadata = pd.read_csv('../data/stage_2_test_metadata.csv')
        # drop redundant columns as NhanNT was too lazy to do it
        # dropped_cols = ['BitsAllocated', 'BitsStored',
        #     'Columns', 'HighBit', 'ImageOrientationPatient_0',
        #     'ImageOrientationPatient_1', 'ImageOrientationPatient_2',
        #     'ImageOrientationPatient_3', 'ImageOrientationPatient_4',
        #     'ImageOrientationPatient_5', 'ImagePositionPatient_0',
        #     'ImagePositionPatient_1', 'Modality',
        #     'PhotometricInterpretation', 'PixelRepresentation',
        #     'PixelSpacing_0', 'PixelSpacing_1', 'RescaleIntercept', 'RescaleSlope',
        #     'Rows', 'SOPInstanceUID', 'SamplesPerPixel', 'SeriesInstanceUID',
        #     'StudyID',]
        dropped_cols = ['ImageName','BrainPresence']
        metadata.drop(dropped_cols, axis=1, inplace=True)
        metadata = metadata[metadata['image'].isin(list(df['image'].values))]

        # merge to sort id
        df = pd.merge(metadata, df, how='inner', on='image')

        # drop columns not needed in df
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
        window_cnn = tf.keras.models.load_model(f'../datnt/windows/{studypatient}id_fold{fold}.h5')
        windowed_prediction = window_cnn.predict(features)
        prediction[features_idcs] = windowed_prediction
        df.iloc[:,1:7] = prediction
        df.to_csv(f'../output_stage2/{csv_name}.csv', index=False)
