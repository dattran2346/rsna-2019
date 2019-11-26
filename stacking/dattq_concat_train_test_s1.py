import pandas as pd 
import os

if __name__ == "__main__":
    if not os.path.exists('../dattq/stage2_train/'):
        os.makedirs('../dattq/stage2_train/')

    model_train_dict1 = {
        0: ['../dattq/stacking/val_resnext50_32x4d_fold0_x256_tta3.csv',
            '../dattq/stacking/test_resnext50_32x4d_fold0_x256_tta3.csv'],

        1: ['../dattq/stacking/val_resnext50_32x4d_fold1_x256_tta3.csv',
            '../dattq/stacking/test_resnext50_32x4d_fold1_x256_tta3.csv'],

        2: ['../dattq/stacking/val_resnext50_32x4d_fold2_x256_tta3.csv',
            '../dattq/stacking/test_resnext50_32x4d_fold2_x256_tta3.csv'],

        3: ['../dattq/stacking/val_resnext50_32x4d_fold3_x256_tta3.csv',
            '../dattq/stacking/test_resnext50_32x4d_fold3_x256_tta3.csv'],

        4: ['../dattq/stacking/val_resnext50_32x4d_fold4_x256_tta3.csv',
            '../dattq/stacking/test_resnext50_32x4d_fold4_x256_tta3.csv'],
    }

    model_train_dict2 = {
        0: ['../dattq/stacking/val_resnext101_32x4d_fold0_x384_tta3.csv',
            '../dattq/stacking/test_resnext101_32x4d_fold0_x384_tta3.csv'],

        1: ['../dattq/stacking/val_resnext101_32x4d_fold1_x384_tta3.csv',
            '../dattq/stacking/test_resnext101_32x4d_fold1_x384_tta3.csv'],

        2: ['../dattq/stacking/val_resnext101_32x4d_fold2_x384_tta3.csv',
            '../dattq/stacking/test_resnext101_32x4d_fold2_x384_tta3.csv'],

        3: ['../dattq/stacking/val_resnext101_32x4d_fold3_x384_tta3.csv',
            '../dattq/stacking/test_resnext101_32x4d_fold3_x384_tta3.csv'],

        4: ['../dattq/stacking/val_resnext101_32x4d_fold4_x384_tta3.csv',
            '../dattq/stacking/test_resnext101_32x4d_fold4_x384_tta3.csv'],
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

        new_valid_df.to_csv(model_train_dict1[fold][0].replace('stacking', 'stage2_train'), index=False)

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

        new_valid_df.to_csv(model_train_dict2[fold][0].replace('stacking', 'stage2_train'), index=False)