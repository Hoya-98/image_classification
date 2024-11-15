import random

import pandas as pd
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.metrics import f1_score, classification_report, confusion_matrix, accuracy_score, auc, roc_auc_score
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

####################################################################################################################################################################################

def set_seed(CFG):

    random.seed(CFG['Random_Seed'])
    np.random.seed(CFG['Random_Seed'])
    torch.manual_seed(CFG['Random_Seed'])
    if CFG['Device'] == 'cuda':
        torch.cuda.manual_seed(CFG['Random_Seed'])

####################################################################################################################################################################################

def validation_report(cm_df, cm_report, _val_acc, _val_f1_score, _val_auc, _val_classwise_auc):
        
    print("Validation_Confusion_Matrix:")
    print(cm_df)
    print("Validation_Classification_Report")
    print(cm_report)
    print()
    print(f"Validation_Accuracy : {_val_acc:.4f}")
    print(f"Validation_F1_Score : {_val_f1_score:.4f}")
    print(f"Validation_AucRoc_Score : {_val_auc:.4f}")

    for key, value in _val_classwise_auc.items():
        print(f"{key} AucRoc_Score : {value}")


def validation_result(CFG, preds, labels, probs):

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
    
    validation_report(cm_df, cm_report, acc, f1, auc, classwise_auc)

    return f1


def validation(CFG, model, criterion, val_loader):

    model.eval()
    val_loss = []
    preds = []
    probs = []
    labels = []

    with torch.no_grad():
        for input, label in tqdm(val_loader):

            input = input.to(CFG['Device'])
            label = label.to(CFG['Device'])

            pred = model(input)
            loss = criterion(pred, label)

            preds += pred.argmax(1).tolist()
            labels += label.tolist()
            probs += F.softmax(pred, dim=1).tolist()

            val_loss.append(loss.item())

        _val_f1_score = validation_result(np.array(preds), np.array(labels), np.array(probs))
        _val_loss = np.mean(val_loss)
        
    return _val_loss, _val_f1_score


def train(CFG, model, optimizer, scheduler, train_loader, val_loader):
    
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1).to(CFG['Device'])
    early_stop_counter = 0
    best_score = 0
    best_model = None

    train_loss_list = []
    val_loss_list = []
    val_f1_score_list = []

    try:
        for epoch in range(CFG['Epochs']):
            
            model.train()
            train_loss = []

            for input, label in tqdm(train_loader):

                input = input.to(CFG['Device'])
                label = label.to(CFG['Device'])

                pred = model(input)
                loss = criterion(pred, label)

                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                train_loss.append(loss.item())

            _train_loss = np.mean(train_loss)
            _val_loss, _val_f1_score = validation(CFG, model, criterion, val_loader)

            print(f"Epoch[{epoch+1}]")
            print(f"Train_Loss: [{_train_loss:.4f}]")   
            print(f"Val_Loss: [{_val_loss:.4f}]")
            print('#' * 200)

            train_loss_list.append(_train_loss)
            val_loss_list.append(_val_loss)
            val_f1_score_list.append(_val_f1_score)

            if scheduler is not None:
                scheduler.step(_val_loss)

            if best_score < _val_f1_score:
                early_stop_counter = 0
                best_score = _val_f1_score
                best_model = model
                model_save_path = f"./model/{CFG['Today_Date']}_{CFG['Current_Time']}.pth"
                torch.save(best_model.to('cpu').state_dict(), model_save_path)
                best_model.to(CFG['Device'])

                print(f"{epoch+1} ::::::::::::::: update the best model best score(f1) {best_score} :::::::::::::::")
                print('#' * 200)

            else:
                early_stop_counter += 1
                if early_stop_counter >= CFG['Early_Stop']:
                    print(f"Early Stop : validation score did not improve for {CFG['Early_Stop']} epochs")
                    print('#' * 200)
                    break

    except KeyboardInterrupt as e:
        print(f"{e} 그래프를 그립니다")

    plt.figure(figsize=(12, 6))
    plt.plot(train_loss_list, label='Train Loss')
    plt.plot(val_loss_list, label='Validation Loss')
    plt.title('Loss History')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(f"./history/{CFG['Today_Date']}_{CFG['Current_Time']}_Loss.png")

    plt.figure(figsize=(12, 6))
    plt.plot(val_f1_score_list, label='Validation F1 Score')
    plt.title("F1 Score History")
    plt.xlabel('Epochs')
    plt.ylabel('F1 score')
    plt.legend()
    plt.savefig(f"./history/{CFG['Today_Date']}_{CFG['Current_Time']}_F1_Score.png")
