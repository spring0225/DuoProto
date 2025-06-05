import monai
import torch
from monai.data import decollate_batch
from monai.metrics import ROCAUCMetric
from sklearn.metrics import classification_report, cohen_kappa_score, confusion_matrix, precision_score, recall_score, f1_score, roc_auc_score, roc_curve
from sklearn.metrics import balanced_accuracy_score, average_precision_score, accuracy_score
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sn
import pandas as pd


class ScalarContainer(object):
    def __init__(self):
        self.scalar_list = []

    def write(self, s):
        self.scalar_list.append(float(s))

    def read(self):
        ave = np.mean(np.array(self.scalar_list))
        self.scalar_list = []
        return ave
    
def bootstrap_f1_threshold_evaluation(labels, probs, n_iterations=100):
    """
    Bootstrap evaluation using F1-optimal threshold.
    Returns sensitivity, specificity, F1, precision, accuracy, and threshold.
    """
    metrics = {
        'Sensitivity': [],
        'Specificity': [],
        'F1': [],
        'Precision': [],
        'Accuracy': [],
        'Threshold': [],
    }

    for _ in range(n_iterations):
        indices = np.random.choice(len(labels), len(labels), replace=True)
        resampled_labels = labels[indices]
        resampled_probs = probs[indices]

        thresholds = np.linspace(0.01, 0.99, 100)
        best_thresh = 0.5
        best_f1 = 0

        for thresh in thresholds:
            preds = (resampled_probs >= thresh).astype(int)
            f1 = f1_score(resampled_labels, preds)
            if f1 > best_f1:
                best_f1 = f1
                best_thresh = thresh

        resampled_preds = (resampled_probs >= best_thresh).astype(int)
        sens = recall_score(resampled_labels, resampled_preds)
        tn, fp, fn, tp = confusion_matrix(resampled_labels, resampled_preds, labels=[0, 1]).ravel()
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0
        precision = precision_score(resampled_labels, resampled_preds)
        accuracy = accuracy_score(resampled_labels, resampled_preds)

        metrics['Sensitivity'].append(sens)
        metrics['Specificity'].append(spec)
        metrics['F1'].append(best_f1)
        metrics['Precision'].append(precision)
        metrics['Accuracy'].append(accuracy)
        metrics['Threshold'].append(best_thresh)

    results = {}
    for metric, values in metrics.items():
        mean_val = np.mean(values)
        std_val = np.std(values)
        interval = 1.96 * std_val
        results[metric] = {'mean': mean_val, '± interval': interval}

    return results

    
def find_youden_threshold(probs, labels):
    fpr, tpr, thresholds = roc_curve(labels, probs)
    youden_index = tpr - fpr
    best_idx = youden_index.argmax()
    best_threshold = thresholds[best_idx]
    return best_threshold, tpr[best_idx], 1 - fpr[best_idx]

from sklearn.metrics import roc_curve, recall_score, f1_score, confusion_matrix, precision_score, accuracy_score

def bootstrap_youden_evaluation(labels, probs, n_iterations=100):
    """
    Bootstrap evaluation using Youden Index threshold.
    Returns sensitivity, specificity, F1, precision, accuracy, and threshold.
    """
    metrics = {
        'Sensitivity': [],
        'Specificity': [],
        'F1': [],
        'Precision': [],
        'Accuracy': [],
        'Threshold': [],
    }

    for _ in range(n_iterations):
        indices = np.random.choice(len(labels), len(labels), replace=True)
        resampled_labels = labels[indices]
        resampled_probs = probs[indices]

        # Find best threshold using Youden Index
        fpr, tpr, thresholds = roc_curve(resampled_labels, resampled_probs)
        youden_index = tpr - fpr
        best_idx = np.argmax(youden_index)
        best_thresh = thresholds[best_idx]

        # Apply threshold to get predicted labels
        resampled_preds = (resampled_probs >= best_thresh).astype(int)

        # Compute metrics
        sens = recall_score(resampled_labels, resampled_preds)
        tn, fp, fn, tp = confusion_matrix(resampled_labels, resampled_preds, labels=[0, 1]).ravel()
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0
        f1 = f1_score(resampled_labels, resampled_preds)
        precision = precision_score(resampled_labels, resampled_preds)
        accuracy = accuracy_score(resampled_labels, resampled_preds)

        # Store results
        metrics['Sensitivity'].append(sens)
        metrics['Specificity'].append(spec)
        metrics['F1'].append(f1)
        metrics['Precision'].append(precision)
        metrics['Accuracy'].append(accuracy)
        metrics['Threshold'].append(best_thresh)

    # Compute mean ± 95% CI
    results = {}
    for metric, values in metrics.items():
        mean_val = np.mean(values)
        std_val = np.std(values)
        interval = 1.96 * std_val
        results[metric] = {'mean': mean_val, '± interval': interval}

    return results


def bootstrap_evaluation(preds, labels, probs, n_iterations=100):
    """
    Perform bootstrap evaluation of model metrics.
    
    Args:
        preds (array): array of predictions. Dimension is N.
        labels (array): array of labels. Dimension is N.
        probs (array): array of probabilities. Dimension is N x num_classes.
        n_iterations (int): number of bootstrap iterations.
    
    Returns:
        dict: dictionary containing mean and 95% confidence interval for each metric.
    """
    metrics = {
        'kappa': [],
        'precision': [],
        'Sensitivity (Recall)': [],  # Same as sensitivity
        'Specificity': [],
        'f1': [],
        'AUROC': [],
        'AUPRC': [],
        'accuracy': [],
        'balanced_accuracy': [],
        'mcc': []
    }
    
    for _ in range(n_iterations):
        indices = np.random.choice(len(labels), len(labels), replace=True)
        resampled_preds = preds[indices]
        resampled_labels = labels[indices]
        resampled_probs = probs[indices]
        
        kappa = cohen_kappa_score(resampled_labels, resampled_preds)
        precision = precision_score(resampled_labels, resampled_preds, average='binary')
        recall = recall_score(resampled_labels, resampled_preds, average='binary')
        f1 = f1_score(resampled_labels, resampled_preds, average='binary')
        auroc = roc_auc_score(resampled_labels, resampled_probs)
        auprc = average_precision_score(resampled_labels, resampled_probs)
        accuracy = np.mean(resampled_preds == resampled_labels)
        balanced_acc = balanced_accuracy_score(resampled_labels, resampled_preds)
        
        # Calculate specificity
        tn, fp, fn, tp = confusion_matrix(resampled_labels, resampled_preds, labels=[0, 1]).ravel()
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        
        # Calculate Matthews Correlation Coefficient
        numerator = (tp * tn) - (fp * fn)
        denominator = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
        mcc = numerator / denominator if denominator > 0 else 0
        
        
       
        metrics['AUPRC'].append(auprc)
        metrics['AUROC'].append(auroc)
        metrics['accuracy'].append(accuracy)
        metrics['precision'].append(precision)
        metrics['Sensitivity (Recall)'].append(recall)
        metrics['Specificity'].append(specificity)
        metrics['f1'].append(f1)
        metrics['kappa'].append(kappa)
        metrics['balanced_accuracy'].append(balanced_acc)
        metrics['mcc'].append(mcc)
    
    results = {}
    for metric, values in metrics.items():
        mean_val = np.mean(values)
        std_val = np.std(values)
        interval = 1.96 * std_val  # 95% confidence interval
        results[metric] = {'mean': mean_val, '± interval': interval}
    
    return results


def get_metrics(model, val_loader, device, logging, args):

    if args.checkpoint != "complete":
        model.load_state_dict(torch.load(args.checkpoint))
        model.eval()
        logging.info("Model loaded from checkpoint")
    
    Val_ER_Acc = ScalarContainer()
    
    er_gts, er_y_hat, er_probs = [], [], []

    
    with torch.no_grad():
        for val_data in val_loader:
            if args.phase == "late" or args.dataset == "NEW":
                PREimgs = val_data["preimg"].to(device)
                Aimgs = val_data["Aimg"].to(device)
                PVimgs = val_data["PVimg"].to(device)
                Delayimgs = val_data["Delayimg"].to(device)
                input = (PREimgs, Aimgs, PVimgs, Delayimgs)
                print("input shape:", input[0].shape, input[1].shape, input[2].shape, input[3].shape)
            else:
                input = val_data[args.phase].to(device)
                
            er_labels = val_data["label"].to(device)
            
            # Forward pass
            outputs = model(input)
            
            er_preds = outputs
            
            # Process ER predictions
            er_prob_preds = F.softmax(er_preds, dim=1)
            er_pred_label = torch.argmax(er_prob_preds, dim=1).item()
            er_gt = er_labels.item()
            
            er_gts.append(er_gt)
            er_y_hat.append(er_pred_label)
            er_probs.append(er_prob_preds[0, 1].cpu().detach().numpy())
            
            # Compute ER Accuracy
            er_val_acc = (er_pred_label == er_labels).float().mean().item() * 100
            Val_ER_Acc.write(er_val_acc)

    # Process ER results
    ER_acc = Val_ER_Acc.read()
    er_gts, er_y_hat, er_probs = np.asarray(er_gts), np.asarray(er_y_hat), np.asarray(er_probs)
    
    er_bootstrap_results = bootstrap_evaluation(er_y_hat, er_gts, er_probs)
    logging.info("ER VALIDATION RESULTS:")
    for metric, values in er_bootstrap_results.items():
        logging.info(f"ER {metric}: mean={values['mean']:.4f} ± {values['± interval']:.4f}")
    
    # === Youden Index bootstrap for ER ===
    youden_results = bootstrap_youden_evaluation(er_gts, er_probs)
    logging.info("====== ER Bootstrap (Youden Threshold) ======")
    for metric, value in youden_results.items():
        logging.info(f"ER {metric}: mean={value['mean']:.4f} ± {value['± interval']:.4f}")


    f1_results = bootstrap_f1_threshold_evaluation(er_gts, er_probs)
    logging.info("====== ER Bootstrap (F1 Threshold) ======")
    for metric, value in f1_results.items():
        logging.info(f"ER {metric}: mean={value['mean']:.4f} ± {value['± interval']:.4f}")

    
    er_cm = confusion_matrix(er_gts, er_y_hat)
    logging.info(f"ER Confusion matrix: \n{er_cm}")
    
    # Generate and save ER confusion matrix visualization
    df_er_cm = pd.DataFrame(er_cm, index=["No (0)", "ER (1)"], columns=["No (0)", "ER (1)"])
    f_er, ax_er = plt.subplots(figsize=(10, 8))
    sn.set(font_scale=1.5)
    sn.heatmap(df_er_cm, annot=True, cmap="YlGnBu", fmt="d", annot_kws={"fontsize": 20})
    ax_er.tick_params(labelsize=20)
    ax_er.set_xlabel("Predict", fontsize=22)
    ax_er.set_ylabel("Actual", fontsize=22)
    f_er.savefig(f"path/er_confusion_matrix.png", bbox_inches='tight')
    plt.close(f_er)
    

    if hasattr(val_loader.dataset, "data") and "PID" in val_loader.dataset.data[0]:
        pids = [entry["PID"] for entry in val_loader.dataset.data]
        save_path = f"path/er_predictions.csv"
        save_er_predictions_csv(
            pids=pids,
            er_gts=er_gts,
            er_y_hat=er_y_hat,
            er_probs=er_probs,
            save_path=save_path
        )


import csv
import os

def save_er_predictions_csv(pids, er_gts, er_y_hat, er_probs, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["PID", "ER_True", "ER_Pred", "ER_Prob"])
        for i in range(len(pids)):
            writer.writerow([pids[i], int(er_gts[i]), int(er_y_hat[i]), float(er_probs[i])])
    
    print(f"Prediction results saved to: {save_path}")
    
