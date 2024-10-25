import os
import sys

import pandas as pd
import numpy as np
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvsion.transforms as T

from sklearn.metrics import f1_score, classification_report, confusion_matrix, accuracy_score, auc, roc_auc_score

from utils.dataset import cus_Dataset
from utils.options import Tee
from utils.model import ConvNext

####################################################################################################################################################################################

def test_report(cm_df, cm_report, _test_acc, _test_f1_score, _test_auc, _test_classwise_auc):
        
    print("Test_Confusion_Matrix:")
    print(cm_df)
    print("Test_Classification_Report")
    print(cm_report)
    print()
    print(f"Test_Accuracy : {_test_acc:.4f}")
    print(f"Test_F1_Score : {_test_f1_score:.4f}")
    print(f"Test_AucRoc_Score : {_test_auc:.4f}")

    for key, value in _test_classwise_auc:
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
    preds, probs, labels = np.array([]), np.array([]), np.array([])

    with torch.no_grad():
        for input, label in tqdm(test_loader):

            input = input.to(CFG['Device'])
            label = label.to(CFG['Device'])

            pred = model(input)

            preds += pred.argmax(1).detach().cpu().numpy()
            labels += label.detach().cpu().numpy()
            probs += F.softmax(pred, dim=1).cpu().numpy()
        
        test_result(preds, labels, probs)
        

def main(CFG):

    log_path = f"./log/{CFG['Target_Nums']}classes_{CFG['Model_Name']}_test_log_{CFG['Today_Date']}.txt"
    
    if os.path.exists(log_path):
        log_path = f"./log/{CFG['Target_Nums']}classes_{CFG['Model_Name']}_test_log_{CFG['Today_Date']}_{CFG["Current_Time"]}.txt"

    logfile = open(log_path, 'a')
    original_stdout = sys.stdout
    sys.stdout = Tee(original_stdout, logfile)

    print('Log 기록을 위한 Memo')
    print(CFG)

    model = ConvNext(n_classes=CFG['Target_Nums'])
    model.load_state_dict(torch.load(f"./model/{CFG['Weight']}"), map_location=CFG['Device'])
    model.to(CFG['Device'])

    valid_transform = T.Compose([
        T.Resize((CFG['Img_Size'], CFG['Img_Size'])),
        T.ToTensor(),
        T.Normalize(mean=[0.485], std=[0.229])
    ])

    test_csv_path = f"./csv_files/{CFG['Target_Nums']}classes_{CFG['Model_Name']}_Test.csv"
    test_dataset = cus_Dataset(csv_path=test_csv_path, transform=valid_transform)
    test_dataloader = DataLoader(test_dataset, batch_size=CFG['Batch_Size'], shuffle=False, num_workers=CFG['Num_Workers'])

    test(CFG, model, test_dataloader)

    sys.stdout = original_stdout
    logfile.close()

####################################################################################################################################################################################

if __name__ == '__main__':

    CFG = {
        'Device' : torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        'Weight' : '4classes_ConvNext.pth',
        
        'Img_Size' : 384,
        'Batch_Size' : 32,
        'Num_Workers' : 4,

        'Model_Name' : 'ConvNext',
        'Target_Nums' : 4,
        'Target_Names' : ['1도', '표재성2도', '심재성2도', '3도'],
        'Target_Label' : 'all'
    }

    main(CFG)