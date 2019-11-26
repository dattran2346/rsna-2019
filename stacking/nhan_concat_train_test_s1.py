import pandas as pd 
import os

if __name__ == "__main__":
    if not os.path.exists('../dung/stage2_train/'):
        os.makedirs('../dung/stage2_train/')

    model_train_dict1 = {
        0: ['../nhannt/rsna_outputs_stage1/val_blseresnext50_32x4d_a2_b4_bilstm_fold0.csv',
            '../nhannt/rsna_outputs_stage1/test_blseresnext50_32x4d_a2_b4_bilstm_fold0.csv'],

        1: ['../nhannt/rsna_outputs_stage1/val_blseresnext50_32x4d_a2_b4_bilstm_fold1.csv',
            '../nhannt/rsna_outputs_stage1/test_blseresnext50_32x4d_a2_b4_bilstm_fold1.csv'],

        2: ['../nhannt/rsna_outputs_stage1/val_blseresnext50_32x4d_a2_b4_bilstm_fold2.csv',
            '../nhannt/rsna_outputs_stage1/test_blseresnext50_32x4d_a2_b4_bilstm_fold2.csv'],

        3: ['../nhannt/rsna_outputs_stage1/val_blseresnext50_32x4d_a2_b4_bilstm_fold3.csv',
            '../nhannt/rsna_outputs_stage1/test_blseresnext50_32x4d_a2_b4_bilstm_fold3.csv'],

        4: ['../nhannt/rsna_outputs_stage1/val_blseresnext50_32x4d_a2_b4_bilstm_fold4.csv',
            '../nhannt/rsna_outputs_stage1/test_blseresnext50_32x4d_a2_b4_bilstm_fold4.csv'],
    }

    model_train_dict2 = {
        0: ['../nhannt/rsna_outputs_stage1/val_blseresnext101_32x4d_a2_b4_bilstm_fold0.csv',
            '../nhannt/rsna_outputs_stage1/test_blseresnext101_32x4d_a2_b4_bilstm_fold0.csv'],

        1: ['../nhannt/rsna_outputs_stage1/val_blseresnext101_32x4d_a2_b4_bilstm_fold1.csv',
            '../nhannt/rsna_outputs_stage1/test_blseresnext101_32x4d_a2_b4_bilstm_fold1.csv'],

        2: ['../nhannt/rsna_outputs_stage1/val_blseresnext101_32x4d_a2_b4_bilstm_fold2.csv',
            '../nhannt/rsna_outputs_stage1/test_blseresnext101_32x4d_a2_b4_bilstm_fold2.csv'],

        3: ['../nhannt/rsna_outputs_stage1/val_blseresnext101_32x4d_a2_b4_bilstm_fold3.csv',
            '../nhannt/rsna_outputs_stage1/test_blseresnext101_32x4d_a2_b4_bilstm_fold3.csv'],

        4: ['../nhannt/rsna_outputs_stage1/val_blseresnext101_32x4d_a2_b4_bilstm_fold4.csv',
            '../nhannt/rsna_outputs_stage1/test_blseresnext101_32x4d_a2_b4_bilstm_fold4.csv'],
    }

    model_train_dict3 = {
        0: ['../nhannt/rsna_outputs_stage1/val_tf_efficientnet_b3_bigru_patient_fold0.csv',
            '../nhannt/rsna_outputs_stage1/test_tf_efficientnet_b3_bigru_patient_fold0.csv'],

        1: ['../nhannt/rsna_outputs_stage1/val_tf_efficientnet_b3_bigru_patient_fold1.csv',
            '../nhannt/rsna_outputs_stage1/test_tf_efficientnet_b3_bigru_patient_fold1.csv'],

        2: ['../nhannt/rsna_outputs_stage1/val_tf_efficientnet_b3_bigru_patient_fold2.csv',
            '../nhannt/rsna_outputs_stage1/test_tf_efficientnet_b3_bigru_patient_fold2.csv'],

        3: ['../nhannt/rsna_outputs_stage1/val_tf_efficientnet_b3_bigru_patient_fold3.csv',
            '../nhannt/rsna_outputs_stage1/test_tf_efficientnet_b3_bigru_patient_fold3.csv'],

        4: ['../nhannt/rsna_outputs_stage1/val_tf_efficientnet_b3_bigru_patient_fold4.csv',
            '../nhannt/rsna_outputs_stage1/test_tf_efficientnet_b3_bigru_patient_fold4.csv'],
    }

    test_s1_with_label = pd.read_csv('../dung/dataset/testset_stage1_labels.csv')
    for fold in range(5):
        print('*'*40 + ' FOLD {} '.format(fold) + '*'*40)
        test_s1_fold = test_s1_with_label.loc[test_s1_with_label['fold'] == fold]
        valid_df = pd.read_csv(model_train_dict1[fold][0])
        valid_df = valid_df[['image','any','epidural','intraparenchymal','intraventricular','subarachnoid','subdural']]
        test_df = pd.read_csv(model_train_dict1[fold][1])
        test_df = test_df[['image','any','epidural','intraparenchymal','intraventricular','subarachnoid','subdural']]
        test_df = test_df[test_df['image'].isin(list(test_s1_fold.image.values))].reset_index(drop=True)
        new_valid_df = pd.concat([valid_df, test_df])
        print(valid_df.shape, test_df.shape, new_valid_df.shape)

        new_valid_df.to_csv(model_train_dict1[fold][0].replace('rsna_outputs_stage1', 'stage2_train'), index=False)

    for fold in range(5):
        print('*'*40 + ' FOLD {} '.format(fold) + '*'*40)
        test_s1_fold = test_s1_with_label.loc[test_s1_with_label['fold'] == fold]
        valid_df = pd.read_csv(model_train_dict2[fold][0])
        valid_df = valid_df[['image','any','epidural','intraparenchymal','intraventricular','subarachnoid','subdural']]
        test_df = pd.read_csv(model_train_dict2[fold][1])
        test_df = test_df[['image','any','epidural','intraparenchymal','intraventricular','subarachnoid','subdural']]
        test_df = test_df[test_df['image'].isin(list(test_s1_fold.image.values))].reset_index(drop=True)
        new_valid_df = pd.concat([valid_df, test_df])
        print(valid_df.shape, test_df.shape, new_valid_df.shape)

        new_valid_df.to_csv(model_train_dict2[fold][0].replace('rsna_outputs_stage1', 'stage2_train'), index=False)

    for fold in range(5):
        print('*'*40 + ' FOLD {} '.format(fold) + '*'*40)
        test_s1_fold = test_s1_with_label.loc[test_s1_with_label['fold'] == fold]
        valid_df = pd.read_csv(model_train_dict3[fold][0])
        valid_df = valid_df[['image','any','epidural','intraparenchymal','intraventricular','subarachnoid','subdural']]
        test_df = pd.read_csv(model_train_dict3[fold][1])
        test_df = test_df[['image','any','epidural','intraparenchymal','intraventricular','subarachnoid','subdural']]
        test_df = test_df[test_df['image'].isin(list(test_s1_fold.image.values))].reset_index(drop=True)
        new_valid_df = pd.concat([valid_df, test_df])
        print(valid_df.shape, test_df.shape, new_valid_df.shape)

        new_valid_df.to_csv(model_train_dict3[fold][0].replace('rsna_outputs_stage1', 'stage2_train'), index=False)