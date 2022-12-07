import os
import csv
import numpy as np
from sklearn.metrics import matthews_corrcoef, accuracy_score, confusion_matrix
from sklearn.metrics import roc_auc_score, average_precision_score, cohen_kappa_score, balanced_accuracy_score


# save results
def save_result(dictionary, save_dir="/content/drive/My Drive/Predict_task/result", filename='Result.csv'):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    path = os.path.join(save_dir, filename)
    if not (os.path.exists(path)):
        logfile = open(path, 'a')
        logwriter = csv.DictWriter(logfile, fieldnames=list(dictionary.keys()))
        logwriter.writeheader()
        logwriter = csv.DictWriter(logfile, fieldnames = dictionary.keys())
        logwriter.writerow(dictionary)
    else:
        logfile = open(path, 'a')
        logwriter = csv.DictWriter(logfile, fieldnames=dictionary.keys())
        logwriter.writerow(dictionary)
    logfile.close()


# Get probabilities 
def get_prob(prob_list, best_epoch):
    bestE_problist = []
    for batch_y in prob_list[best_epoch]:
        for i in batch_y:
            bestE_problist.append(i.detach().cpu().numpy())
    bestE_problist = np.array(bestE_problist)
    return bestE_problist


# Get model performance
def performance(labels, probs, thresold=0.5, name='test_dataset', path_save=None):
    #------------------------------------
    if thresold != 0.5:
        predicted_labels = []
        for prob in probs:
            if prob >= thresold:
                predicted_labels.append(1)
            else:
                predicted_labels.append(0)
    else:
        predicted_labels = np.round(probs)
    #------------------------------------
    tn, fp, fn, tp  = confusion_matrix(labels, predicted_labels).ravel()
    acc             = np.round(accuracy_score(labels, predicted_labels), 4)
    ba              = np.round(balanced_accuracy_score(labels, predicted_labels), 4)
    roc_auc         = np.round(roc_auc_score(labels, probs),4)
    pr_auc          = np.round(average_precision_score(labels, probs), 4)
    mcc             = np.round(matthews_corrcoef(labels, predicted_labels), 4)
    sensitivity     = np.round(tp / (tp + fn), 4)
    specificity     = np.round(tn / (tn + fp), 4)
    precision       = np.round(tp / (tp + fp), 4)
    f1              = np.round(2*precision*sensitivity / (precision + sensitivity), 4)
    ck              = np.round(cohen_kappa_score(labels, predicted_labels), 4)
    print('Performance for {}'.format(name))
    print(f'AUC-ROC: {roc_auc}, AUC-PR: {pr_auc}, Accuracy: {acc}, B_ACC : {ba}, \
        MCC: {mcc}, Sensitivity/Recall: {sensitivity}, Specificity: {specificity}, \
        Precision: {precision}, F1-score: {f1}, CK-score {ck}')
    result = {}
    result['Dataset'] = name
    result['AUC'] = roc_auc
    result['PR_AUC'] = pr_auc
    result['ACC'] = acc
    result['B_ACC'] = ba
    result['MCC'] = mcc
    result['Sensitivity'] = sensitivity
    result['F1'] = f1
    result['Pre'] = precision
    result['Spe'] = specificity
    result['Ck']  = ck
    if path_save:
        save_result(result, save_dir= path_save, filename='result_test_cut.csv')
    return roc_auc, pr_auc, acc, ba, mcc, sensitivity, specificity, precision, f1, ck
