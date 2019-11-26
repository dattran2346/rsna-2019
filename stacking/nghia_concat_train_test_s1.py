import pandas as pd 
import os

if __name__ == "__main__":
    if not os.path.exists('../nghia/stage2_train/'):
        os.makedirs('../nghia/stage2_train/')
    
    model_train_dict = {
        0: ['../nghia/raw_pred_s1/valid_exp25_fold0_lstm.csv',
            '../nghia/raw_pred_s1/test_exp25_fold0_lstm.csv'],

        1: ['../nghia/raw_pred_s1/valid_exp25_fold1_lstm.csv',
            '../nghia/raw_pred_s1/test_exp25_fold1_lstm.csv'],

        2: ['../nghia/raw_pred_s1/valid_exp25_fold2_lstm.csv',
            '../nghia/raw_pred_s1/test_exp25_fold2_lstm.csv'],

        3: ['../nghia/raw_pred_s1/valid_exp25_fold3_lstm.csv',
            '../nghia/raw_pred_s1/test_exp25_fold3_lstm.csv'],

        4: ['../nghia/raw_pred_s1/valid_exp25_fold4_lstm.csv',
            '../nghia/raw_pred_s1/test_exp25_fold4_lstm.csv'],
    }

    test_s1_with_label = pd.read_csv('../dung/dataset/testset_stage1_labels.csv')
    for fold in range(5):
        print('*'*40 + ' FOLD {} '.format(fold) + '*'*40)
        test_s1_fold = test_s1_with_label.loc[test_s1_with_label['fold'] == fold]
        valid_df = pd.read_csv(model_train_dict[fold][0])
        valid_df = valid_df[['image','any','epidural','intraparenchymal','intraventricular','subarachnoid','subdural']]
        test_df = pd.read_csv(model_train_dict[fold][1])
        test_df = test_df[['image','any','epidural','intraparenchymal','intraventricular','subarachnoid','subdural']]
        test_df = test_df[test_df['image'].isin(list(test_s1_fold.image.values))].reset_index(drop=True)
        new_valid_df = pd.concat([valid_df, test_df])
        print(valid_df.shape, test_df.shape, new_valid_df.shape)

        new_valid_df.to_csv(model_train_dict[fold][0].replace('raw_pred_s1', 'stage2_train'), index=False)