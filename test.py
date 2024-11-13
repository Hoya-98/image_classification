import pandas as pd
import numpy as np
from tqdm import tqdm

import torch
import torch.nn.functional as F

from sklearn.metrics import f1_score, classification_report, confusion_matrix, accuracy_score, roc_auc_score

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

####################################################################################################################################################################################

def test_report(cm_df, cm_report, _test_acc, _test_f1_score, _test_auc, _test_classwise_auc):

    print('Test_Confusion_Matrix')
    print(cm_df)
    print('Test_Classification_Report')
    print(cm_report)
    print()
    print(f"Test_Accuracy : {_test_acc:.4f}"), 
    print(f"Test_F1_Score : {_test_f1_score:.4f}"), 
    print(f"Test_AucRoc_Score : {_test_auc:.4f}")

    for key, value in _test_classwise_auc.items():
        print(f"{key} AucRoc_Score : {value}")


def test_result(CFG, preds, labels, probs):

    cm = confusion_matrix(labels, preds)
    cm_df = pd.DataFrame(cm, index=CFG['Target_Names'], columns=CFG['Target_Names'])
    cm_report = classification_report(labels, preds, target_names=CFG['Target_Names'], digits=4, output_dict=False)
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average='weighted')
    auc = roc_auc_score(y_true=labels, y_score=probs, average='weighted', multi_class='ovr')

    classwise_auc = {}
    one_hot_labels = pd.get_dummies(labels).values

    for i in range(CFG['Target_Nums']):
        class_auc = roc_auc_score(one_hot_labels[: , i], probs[:, i])
        classwise_auc[CFG['Target_Names'][i]] = class_auc
    
    test_report(cm_df, cm_report, acc, f1, auc, classwise_auc)


def test(CFG, model, test_loader):

    model.eval()
    preds, probs, labels = [], [], []
    
    with torch.no_grad():
        for input, label in tqdm(test_loader):

            input = input.to(CFG['Device'])
            label = label.to(CFG['Device'])
                
            output = model(input)

            preds += output.argmax(1).detach().tolist()
            labels += label.detach().tolist()
            probs += F.softmax(output, dim=1).tolist()

        test_result(CFG, np.array(preds), np.array(labels), np.array(probs))
