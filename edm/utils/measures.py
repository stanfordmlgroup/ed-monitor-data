import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.metrics import roc_curve, auc
from edm.utils.delong import delong_roc_variance

def perf_measure(y_actual, y_hat):
    y_hat = [round(a) for a in y_hat]

    TP = 0
    FP = 0
    TN = 0
    FN = 0

    for i in range(len(y_hat)): 
        if y_actual[i]==y_hat[i]==1:
            TP += 1
        if y_hat[i]==1 and y_actual[i]!=y_hat[i]:
            FP += 1
        if y_actual[i]==y_hat[i]==0:
            TN += 1
        if y_hat[i]==0 and y_actual[i]!=y_hat[i]:
            FN += 1

    return (TP, FP, TN, FN)

def calculate_output_statistics(y_actual, y_pred, show_plots=True):

    tp, fp, tn, fn = perf_measure(y_actual, y_pred)
    print(f"fp={fp}, fn={fn}, tn={tn}, tp={tp}")
    actual_positives = tp + fn
    actual_negatives = tn + fp

    fpr_keras, tpr_keras, thresholds_keras = roc_curve(y_actual, y_pred)
    auc_keras = auc(fpr_keras, tpr_keras)
    y_pred_vals = np.array(y_pred) >= 0.5

    if show_plots:
        a4_dims = (5, 5)
        fig, ax = plt.subplots(figsize=a4_dims)
        ax.plot([0, 1], [0, 1], 'k--')
        ax.plot(fpr_keras, tpr_keras, label='Model (area = {:.3f})'.format(auc_keras))
        ax.set_xlabel('False positive rate')
        ax.set_ylabel('True positive rate')
        ax.set_title('ROC curve')
        ax.legend(loc='best')
        plt.show()
    
    return auc_keras

def calculate_confidence_intervals(y_actual, y_pred, alpha = 0.95):
    auc, auc_cov = delong_roc_variance(np.array(y_actual), np.array(y_pred))
    
    auc_std = np.sqrt(auc_cov)
    lower_upper_q = np.abs(np.array([0, 1]) - (1 - alpha) / 2)

    ci = stats.norm.ppf(
        lower_upper_q,
        loc=auc,
        scale=auc_std)

    ci[ci > 1] = 1

    print(f"AUC={auc}, AUC COV={auc_cov}, 95% CI={ci}")
    return ci
