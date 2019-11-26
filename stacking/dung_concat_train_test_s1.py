import pandas as pd 
import os

if __name__ == "__main__":
    if not os.path.exists('../dung/stage2_train/'):
        os.makedirs('../dung/stage2_train/')

    model_train_dict1 = {
        0: ['../dung/calib_pred/EfficientnetB5_yp_valid_fold0_epoch11.csv',
            '../dung/calib_pred/EfficientnetB5_yp_test_fold0_epoch11.csv'],

        1: ['../dung/calib_pred/EfficientnetB5_yp_valid_fold1_epoch11.csv',
            '../dung/calib_pred/EfficientnetB5_yp_test_fold1_epoch11.csv'],

        2: ['../dung/calib_pred/EfficientnetB5_yp_valid_fold2_epoch11.csv',
            '../dung/calib_pred/EfficientnetB5_yp_test_fold2_epoch11.csv'],

        3: ['../dung/calib_pred/EfficientnetB5_yp_valid_fold3_epoch11.csv',
            '../dung/calib_pred/EfficientnetB5_yp_test_fold3_epoch11.csv'],

        4: ['../dung/calib_pred/EfficientnetB5_yp_valid_fold4_epoch10.csv',
            '../dung/calib_pred/EfficientnetB5_yp_test_fold4_epoch10.csv'],
    }

    model_train_dict2 = {
        0: ['../dung/calib_pred/EfficientnetB2_yp_valid_fold0_epoch5.csv',
            '../dung/calib_pred/EfficientnetB2_yp_test_fold0_epoch5.csv'],

        1: ['../dung/calib_pred/EfficientnetB2_yp_valid_fold1_epoch11.csv',
            '../dung/calib_pred/EfficientnetB2_yp_test_fold1_epoch11.csv'],

        2: ['../dung/calib_pred/EfficientnetB2_yp_valid_fold2_epoch11.csv',
            '../dung/calib_pred/EfficientnetB2_yp_test_fold2_epoch11.csv'],

        3: ['../dung/calib_pred/EfficientnetB2_yp_valid_fold3_epoch11.csv',
            '../dung/calib_pred/EfficientnetB2_yp_test_fold3_epoch11.csv'],

        4: ['../dung/calib_pred/EfficientnetB2_yp_valid_fold4_epoch11.csv',
            '../dung/calib_pred/EfficientnetB2_yp_test_fold4_epoch11.csv'],
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

        new_valid_df.to_csv(model_train_dict1[fold][0].replace('calib_pred', 'stage2_train'), index=False)

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

        new_valid_df.to_csv(model_train_dict2[fold][0].replace('calib_pred', 'stage2_train'), index=False)