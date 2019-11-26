import pandas as pd 
import os

if __name__ == "__main__":
    if not os.path.exists('../datnt/stage2_train/'):
        os.makedirs('../datnt/stage2_train/')

    model_train_dict1 = {
        0: ['../datnt/windowed_csv/val/datnt_version3_seresnext50_fold0.csv',
            '../datnt/windowed_csv/test/datnt_version3_seresnext50_fold0.csv'],

        1: ['../datnt/windowed_csv/val/datnt_version3_seresnext50_fold1.csv',
            '../datnt/windowed_csv/test/datnt_version3_seresnext50_fold1.csv'],

        2: ['../datnt/windowed_csv/val/datnt_version3_seresnext50_fold2.csv',
            '../datnt/windowed_csv/test/datnt_version3_seresnext50_fold2.csv'],

        3: ['../datnt/windowed_csv/val/datnt_version3_seresnext50_fold3.csv',
            '../datnt/windowed_csv/test/datnt_version3_seresnext50_fold3.csv'],

        4: ['../datnt/windowed_csv/val/datnt_version3_seresnext50_fold4.csv',
            '../datnt/windowed_csv/test/datnt_version3_seresnext50_fold4.csv'],
    }

    model_train_dict2 = {
        0: ['../datnt/windowed_csv/val/datnt_version4_b3_fold0.csv',
            '../datnt/windowed_csv/test/datnt_version4_b3_fold0.csv'],

        1: ['../datnt/windowed_csv/val/datnt_version4_b3_fold1.csv',
            '../datnt/windowed_csv/test/datnt_version4_b3_fold1.csv'],

        2: ['../datnt/windowed_csv/val/datnt_version4_b3_fold2.csv',
            '../datnt/windowed_csv/test/datnt_version4_b3_fold2.csv'],

        3: ['../datnt/windowed_csv/val/datnt_version4_b3_fold3.csv',
            '../datnt/windowed_csv/test/datnt_version4_b3_fold3.csv'],

        4: ['../datnt/windowed_csv/val/datnt_version4_b3_fold4.csv',
            '../datnt/windowed_csv/test/datnt_version4_b3_fold4.csv'],
    }

    model_train_dict3 = {
        0: ['../datnt/windowed_csv/val/datnt_version4_b4_fold0.csv',
            '../datnt/windowed_csv/test/datnt_version4_b4_fold0.csv'],

        1: ['../datnt/windowed_csv/val/datnt_version4_b4_fold1.csv',
            '../datnt/windowed_csv/test/datnt_version4_b4_fold1.csv'],

        2: ['../datnt/windowed_csv/val/datnt_version4_b4_fold2.csv',
            '../datnt/windowed_csv/test/datnt_version4_b4_fold2.csv'],

        3: ['../datnt/windowed_csv/val/datnt_version4_b4_fold3.csv',
            '../datnt/windowed_csv/test/datnt_version4_b4_fold3.csv'],

        4: ['../datnt/windowed_csv/val/datnt_version4_b4_fold4.csv',
            '../datnt/windowed_csv/test/datnt_version4_b4_fold4.csv'],
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

        new_valid_df.to_csv(model_train_dict1[fold][0].replace('windowed_csv/val', 'stage2_train'), index=False)

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

        new_valid_df.to_csv(model_train_dict2[fold][0].replace('windowed_csv/val', 'stage2_train'), index=False)
    
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

        new_valid_df.to_csv(model_train_dict3[fold][0].replace('windowed_csv/val', 'stage2_train'), index=False)