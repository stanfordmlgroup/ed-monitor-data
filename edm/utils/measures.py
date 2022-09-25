import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.metrics import roc_curve, auc, precision_recall_curve, roc_auc_score
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

    precision, recall, _ = precision_recall_curve(y_actual, y_pred)
    auprc_alt = auc(recall, precision)

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
    
    return auc_keras, auprc_alt

def get_bootstrap_metrics(orig_preds, orig_actual, bootstrap_samples=10000):
    bootstrap_aucs = []
    bootstrap_auprcs = []
    np.random.seed(bootstrap_samples)
    for j in tqdm(range(bootstrap_samples)):
        bootstrap_indices = np.random.choice(range(len(orig_preds)), size=len(orig_preds), replace=True)
        preds = [orig_preds[i] for i in bootstrap_indices]
        actual = [orig_actual[i] for i in bootstrap_indices]

        preds = np.array(preds)
        actual = np.array(actual)
        auroc, auprc = calculate_output_statistics(actual, preds, show_plots=False)
        
        bootstrap_aucs.append(auroc)
        bootstrap_auprcs.append(auprc)

    return bootstrap_aucs, bootstrap_auprcs

def calculate_confidence_intervals(y_actual, y_pred, ci_type="bootstrap", alpha=0.95):
    if ci_type == "bootstrap":
        bootstrap_aucs, bootstrap_auprcs = get_bootstrap_metrics(y_pred, y_actual)

        p = ((1.0-alpha)/2.0) * 100
        lower = max(0.0, np.percentile(bootstrap_aucs, p))
        p = (alpha+((1.0-alpha)/2.0)) * 100
        upper = min(1.0, np.percentile(bootstrap_aucs, p))

        auc_ci = [round(lower, 3), round(upper, 3)]

        fpr_keras, tpr_keras, thresholds_keras = roc_curve(y_actual, y_pred)
        auc_keras = auc(fpr_keras, tpr_keras)

        p = ((1.0-alpha)/2.0) * 100
        lower = max(0.0, np.percentile(bootstrap_auprcs, p))
        p = (alpha+((1.0-alpha)/2.0)) * 100
        upper = min(1.0, np.percentile(bootstrap_auprcs, p))

        auprc_ci = [round(lower, 3), round(upper, 3)]

        precision, recall, _ = precision_recall_curve(y_actual, y_pred)
        auprc = auc(recall, precision)

        print(f"[Bootstrap] AUC={round(auc_keras, 3)}, 95% CI={auc_ci}; AUPRC={round(auprc, 3)}, 95% CI={auprc_ci}")
        return auc_ci, auprc_ci
    else:
        auroc, auc_cov = delong_roc_variance(np.array(y_actual), np.array(y_pred))

        auc_std = np.sqrt(auc_cov)
        lower_upper_q = np.abs(np.array([0, 1]) - (1 - alpha) / 2)

        ci = stats.norm.ppf(
            lower_upper_q,
            loc=auroc,
            scale=auc_std)

        if ci[0] > 1:
            ci[0] = 1
        if ci[1] > 1:
            ci[1] = 1

        ci = [round(ci[0], 3), round(ci[1], 3)]
        
        precision, recall, _ = precision_recall_curve(y_actual, y_pred)
        auprc = auc(recall, precision)

        print(f"[DeLong] AUC={round(auroc, 3)}, AUC COV={round(auc_cov, 3)}, 95% CI={ci}; AUPRC={round(auprc, 3)}, 95% CI=N/A")
        return ci, None

def permutation_test_between_clfs(y_test, pred_proba_1, pred_proba_2, nsamples=1000):
    auc_differences = []
    orig_auc1 = roc_auc_score(y_test, pred_proba_1)
    orig_auc2 = roc_auc_score(y_test, pred_proba_2)
    observed_difference = orig_auc1 - orig_auc2
    for _ in tqdm(range(nsamples)):
        mask = np.random.randint(2, size=len(pred_proba_1))
        p1 = np.where(mask, pred_proba_1, pred_proba_2)
        p2 = np.where(mask, pred_proba_2, pred_proba_1)
        auc1 = roc_auc_score(y_test, p1)
        auc2 = roc_auc_score(y_test, p2)
        auc_differences.append(auc1 - auc2)
    p_diff = np.mean(auc_differences >= observed_difference)
    print(f"AUC1 = {orig_auc1}; AUC2 = {orig_auc2}; p_diff = {p_diff}")
    return observed_difference, p_diff
