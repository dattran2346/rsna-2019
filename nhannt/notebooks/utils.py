import numpy as np
import pandas as pd
from tqdm import tqdm_notebook


def compute_corr(csv_file1, csv_file2):
    corr_matrix = np.corrcoef(
        csv_file1.iloc[:, 1:7].values.reshape(-1), 
        csv_file2.iloc[:, 1:7].values.reshape(-1))
    print("Correlation {:.5f}".format(corr_matrix[0, 1]))
    
def smoothing_prediction(pred, k=1):
    # moving average
    total = k * 2 + 1
    w_k = np.zeros((len(pred), 1))
    w_0 = np.ones((len(pred), 1))
    w_k[k*1:k*-1] = 1. / (total * k * 2)
    w_0[k*1:k*-1] = 2. * k / total
    
# def smoothing_prediction(pred, k=1, alpha=2./3):
#     total = k * 2 + 1
#     w_k = np.zeros((len(pred), 1))
#     w_0 = np.ones((len(pred), 1))
#     w_k[k*1:k*-1] = (1. - alpha) / (2. * k)
#     w_0[k*1:k*-1] = alpha
    
    smoothed_pred = w_0 * pred
    for i in range(1, k + 1):
        smoothed_pred += w_k * pred.shift(periods=-i, fill_value=0)
        smoothed_pred += w_k * pred.shift(periods=i, fill_value=0)
    return smoothed_pred

def watersmooth(y_hat):
    output = y_hat
    _dtype = output.dtype
    for d in range(output.shape[1]):
        res = output[:, d].copy()
        n = res.shape[0]
        for i in range(1, n-1, 1):
            # left_idx = np.argmax(res[:i])
            # right_idx = i + 1 + np.argmax(res[i+1:])
            left_idx, right_idx = i-1, i+1
            # if i > 1 and i < n-2:
            #     left_idx, right_idx = i-2, i+2
            to_fill = np.ones(right_idx-left_idx+1, dtype=_dtype) * min(res[left_idx], res[right_idx])
            res[left_idx:right_idx+1] = np.max([res[left_idx:right_idx+1], to_fill], 0)
        output[:, d] = res
    return output

def postprocess_csv_file(orig_csv, test_meta):
    orig_w_meta = pd.merge(orig_csv, test_meta, on="image", how="inner")
    smoothed_probs = []
    image_ids = []

    for i, study_id in enumerate(tqdm_notebook(orig_w_meta["StudyInstanceUID"].unique())):
        study_df = orig_w_meta[orig_w_meta["StudyInstanceUID"]==study_id]
        image_ids.append(study_df.iloc[:, 0].values)
        pred = study_df.iloc[:, 1:7]
        # nhan's
        smoothed_pred = smoothing_prediction(pred, k=1)
        # nghia's
        smoothed_prob = smoothed_pred.values
        smoothed_prob = watersmooth(smoothed_prob)
        smoothed_probs.append(smoothed_prob)
        
    smoothed_probs = np.concatenate(smoothed_probs, 0)
    image_ids = np.concatenate(image_ids, 0)
    return image_ids, smoothed_probs