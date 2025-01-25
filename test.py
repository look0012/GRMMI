import os
import numpy as np
import keras
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score, f1_score, precision_recall_curve, auc,matthews_corrcoef
import matplotlib.pyplot as plt
import csv

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_curve
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_recall_curve, auc, matthews_corrcoef, \
    confusion_matrix


def GenerateBehaviorFeature(InteractionPair, NodeBehavior):
    SampleFeature1 = []
    SampleFeature2 = []
    not_found_counter = 0  

    for i in range(len(InteractionPair)):
        Pair1 = InteractionPair[i][0]
        Pair2 = InteractionPair[i][1]

        found1 = False
        for j in range(len(NodeBehavior)):
            if Pair1 == NodeBehavior[j][0]:
                SampleFeature1.append(NodeBehavior[j][1:])
                found1 = True
                break

        if not found1:
            print("Pair not found for Pair1:", Pair1)
            not_found_counter += 1

        found2 = False
        for k in range(len(NodeBehavior)):
            if Pair2 == NodeBehavior[k][0]:
                SampleFeature2.append(NodeBehavior[k][1:])
                found2 = True
                break

        if not found2:
            print("Pair not found for Pair2:", Pair2)
            not_found_counter += 1

    SampleFeature1 = np.array(SampleFeature1).astype('float32')
    SampleFeature2 = np.array(SampleFeature2).astype('float32')

    print("Number of pairs not found:", not_found_counter)

    return SampleFeature1, SampleFeature2

def ReadMyCsv1(SaveList, fileName):
    csv_reader = csv.reader(open(fileName))
    for row in csv_reader:  
        SaveList.append(row)
    return


Data_dir='data/'
val=np.load(Data_dir+'test.npz')
X_mi_val,X_M_val, y_val=val['X_mi_test'], val['X_M_test'], val['y_test']

AllNodeBehavior = []
ReadMyCsv1(AllNodeBehavior, 'AllNodeBehavior.csv')
PositiveSample_Validation = []
ReadMyCsv1(PositiveSample_Validation, Data_dir+'PositiveSample_Test.csv')
NegativeSample_Validation = []
ReadMyCsv1(NegativeSample_Validation, Data_dir+'NegativeSample_Test.csv')
x_validation_pair = []
x_validation_pair.extend(PositiveSample_Validation)
x_validation_pair.extend(NegativeSample_Validation)

x_validation_1_Behavior, x_validation_2_Behavior = GenerateBehaviorFeature(x_validation_pair, AllNodeBehavior)



kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)



accuracy_scores = []
f1_scores = []
roc_auc_scores = []
aupr_scores = []
mcc_scores = []
sensitivity_scores = []
precision_scores = []
specificity_scores = []


for train_idx, val_idx in kfold.split(X_mi_val, y_val):
    X_mi_train, X_mi_val_fold = X_mi_val[train_idx], X_mi_val[val_idx]
    X_M_train, X_M_val_fold = X_M_val[train_idx], X_M_val[val_idx]
    y_train, y_val_fold = y_val[train_idx], y_val[val_idx]
    behavior_train_1, behavior_val_1 = x_validation_1_Behavior[train_idx], x_validation_1_Behavior[val_idx]
    behavior_train_2, behavior_val_2 = x_validation_2_Behavior[train_idx], x_validation_2_Behavior[val_idx]


    model_dir = 'DNN'
    model_path = os.path.join(model_dir, 'model-ATT2__01_val_acc_0.8568.h5')
    # model_path = os.path.join('model_db.h5')
    best_model = keras.models.load_model(model_path)


    val_pred = best_model.predict([X_mi_val_fold, X_M_val_fold, behavior_val_1, behavior_val_2])
    threshold = 0.5  
    val_pred_binary = (val_pred > threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_val_fold, val_pred_binary).ravel()
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    precision = tp / (tp + fp)

    accuracy = accuracy_score(y_val_fold, val_pred_binary)
    f1 = f1_score(y_val_fold, val_pred_binary)
    roc_auc = roc_auc_score(y_val_fold, val_pred)
    precision_curve, recall_curve, _ = precision_recall_curve(y_val_fold, val_pred)
    aupr = auc(recall_curve, precision_curve)
    mcc = matthews_corrcoef(y_val_fold, val_pred_binary)


    accuracy_scores.append(accuracy)
    f1_scores.append(f1)
    roc_auc_scores.append(roc_auc)
    aupr_scores.append(aupr)
    mcc_scores.append(mcc)
    sensitivity_scores.append(sensitivity)
    precision_scores.append(precision)
    specificity_scores.append(specificity)



print(f"\n：")
print(f" (Accuracy): {np.mean(accuracy_scores):.4f}")
print(f" (F1 Score): {np.mean(f1_scores):.4f}")
print(f" {np.mean(roc_auc_scores):.4f}")
print(f" {np.mean(aupr_scores):.4f}")
print(f"{np.mean(mcc_scores):.4f}")
print(f" (Sensitivity): {np.mean(sensitivity_scores):.4f}")
print(f" (Precision): {np.mean(precision_scores):.4f}")
print(f"(Specificity): {np.mean(specificity_scores):.4f}")