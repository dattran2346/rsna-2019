import pandas as pd 
import os

if __name__ == "__main__":
    if not os.path.exists('s2train_patient_df'):
        os.makedirs('s2train_patient_df')

    test_s1_with_label = pd.read_csv('../dung/dataset/testset_stage1_labels.csv')
    for fold in range(5):
        print('*'*40 + ' FOLD {} '.format(fold) + '*'*40)
        valid_df = pd.read_csv('patient_df/valid_patient_df_fold{}.csv'.format(fold))
        valid_df = valid_df[['image','any','epidural','intraparenchymal','intraventricular','subarachnoid','subdural']]
        test_df = test_s1_with_label.loc[test_s1_with_label['fold'] == fold]
        test_df = test_df[['image','any','epidural','intraparenchymal','intraventricular','subarachnoid','subdural']]

        new_valid_df = pd.concat([valid_df, test_df])

        new_valid_df.to_csv('s2train_patient_df/valid_patient_df_fold{}.csv'.format(fold), index=False)