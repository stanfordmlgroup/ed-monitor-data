import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.metrics import roc_curve, auc
from edm.utils.delong import delong_roc_variance
from tqdm import tqdm

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
    actual_positives = tp + fn
    actual_negatives = tn + fp

    fpr_keras, tpr_keras, thresholds_keras = roc_curve(y_actual, y_pred)
    auc_keras = auc(fpr_keras, tpr_keras)

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

def get_bootstrap_metrics(orig_preds, orig_actual, bootstrap_samples=10000):
    bootstrap_aucs = []
    for j in tqdm(range(bootstrap_samples)):
        bootstrap_indices = np.random.choice(range(len(orig_preds)), size=len(orig_preds), replace=True)
        preds = [orig_preds[i] for i in bootstrap_indices]
        actual = [orig_actual[i] for i in bootstrap_indices]

        preds = np.array(preds)
        actual = np.array(actual)
        auc = calculate_output_statistics(actual, preds, show_plots=False)
        
        bootstrap_aucs.append(auc)

    return bootstrap_aucs

def calculate_confidence_intervals(y_actual, y_pred, ci_type="bootstrap", alpha=0.95):
    if ci_type == "bootstrap":
        bootstrap_aucs = get_bootstrap_metrics(y_pred, y_actual)

        p = ((1.0-alpha)/2.0) * 100
        lower = max(0.0, np.percentile(bootstrap_aucs, p))
        p = (alpha+((1.0-alpha)/2.0)) * 100
        upper = min(1.0, np.percentile(bootstrap_aucs, p))

        ci = [lower, upper]

        print(f"[Bootstrap] 95% CI={ci}")
        return ci
    else:
        auc, auc_cov = delong_roc_variance(np.array(y_actual), np.array(y_pred))

        auc_std = np.sqrt(auc_cov)
        lower_upper_q = np.abs(np.array([0, 1]) - (1 - alpha) / 2)

        ci = stats.norm.ppf(
            lower_upper_q,
            loc=auc,
            scale=auc_std)

        ci[ci > 1] = 1

        print(f"[DeLong] AUC={auc}, AUC COV={auc_cov}, 95% CI={ci}")
        return ci
