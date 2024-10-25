import os
import random
import sys
import datetime

import pandas as pd
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision.transforms as T

from sklearn.metrics import f1_score, classification_report, confusion_matrix, accuracy_score, auc, roc_auc_score
import matplotlib.pyplot as plt

from utils.dataset import cus_Dataset
from utils.options import Tee
from utils.model import ConvNext

####################################################################################################################################################################################

def set_seed(CFG):

    random.seed(CFG['Random_Seed'])
    np.random.seed(CFG['Random_Seed'])
    torch.manual_seed(CFG['Random_Seed'])
    if CFG['Device'] == 'cuda':
        torch.cuda.manual_seed(CFG['Random_Seed'])


def validation_report(cm_df, cm_report, _val_acc, _val_f1_score, _val_auc, _val_classwise_auc):
        
    print("Validation_Confusion_Matrix:")
    print(cm_df)
    print("Validation_Classification_Report")
    print(cm_report)
    print()
    print(f"Validation_Accuracy : {_val_acc:.4f}")
    print(f"Validation_F1_Score : {_val_f1_score:.4f}")
    print(f"Validation_AucRoc_Score : {_val_auc:.4f}")

    for key, value in _val_classwise_auc:
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
    preds, probs, labels = [], [], []

    with torch.no_grad():
        for input, label in tqdm(val_loader):

            input = input.to(CFG['Device'])
            label = label.to(CFG['Device'])

            pred = model(input)
            loss = criterion(pred, label)

            preds += pred.argmax(1).detach().cpu().numpy()
            labels += label.detach().cpu().numpy()
            probs += F.softmax(pred, dim=1).cpu().numpy()

            val_loss.append(loss.item())

        _val_f1_score = validation_result(np.array(preds), np.array(labels), np.array(probs))
        _val_loss = np.mean(val_loss)
        
    return _val_loss, _val_f1_score


def train(CFG, model, optimizer, train_loader, val_loader, scheduler):
    
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1).to(CFG['Device'])
    early_stopping_counter = 0
    best_score = 0
    best_model = None

    train_losses = []
    val_losses = []
    val_f1_scores = []

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
            _val_loss, _val_f1_score = validation(model, criterion, val_loader)

            print(f"Epoch[{epoch+1}]")
            print(f"Train_Loss: [{_train_loss:.4f}]")   
            print(f"Val_Loss: [{_val_loss:.4f}]")   

            train_losses.append(_train_loss)
            val_losses.append(_val_loss)
            val_f1_scores.append(_val_f1_score)

            if scheduler is not None:
                scheduler.step(_val_loss)

            if best_score < _val_f1_score:
                best_score = _val_f1_score
                model_save_path = f"./model/{CFG['Target_Nums']}classes_{CFG['Model_Name']}_{CFG['Today_Date']}.pth"
                torch.save(best_model.to('cpu').state_dict(), model_save_path)
                best_model.to(CFG['Device'])

                print(f"{epoch+1} ::::::::::::::: update the best model best score(f1) {best_score} :::::::::::::::")
            
            else:
                early_stopping_counter += 1
                if early_stopping_counter >= CFG['Early_Stop']:
                    print(f"Early Stopping : validation score did not improve for {CFG['Early_Stop']} epochs")

    except KeyboardInterrupt as e:
        print(f"{e} 그래프를 그립니다")

    plt.figure(figsize=(12, 6))

    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Loss History')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(f"./history/train_loss_{CFG['Target_Nums']}classes_{CFG['Model_Name']}_{CFG['Today_Date']}.png")

    plt.figure(figsize=(12, 6))
    plt.plot(val_f1_scores, label='Validation F1 Score')
    plt.title("F1 Score History")
    plt.xlabel('Epochs')
    plt.ylabel('F1 score')
    plt.legend()
    plt.savefig(f"./history/train_score_{CFG['Target_Nums']}classes_{CFG['Model_Name']}_{CFG['Today_Date']}.png")

    return best_model


def main(CFG):

    log_path = f"./log/{CFG['Target_Nums']}classes_{CFG['Model_Name']}_train_log_{CFG['Today_Date']}.txt"

    if os.path.exists(log_path):
        log_path = f"./log/{CFG['Target_Nums']}classes_{CFG['Model_Name']}_train_log_{CFG['Today_Date']}_{CFG["Current_Time"]}.txt"

    logfile = open(log_path, 'a')
    original_stdout = sys.stdout
    sys.stdout = Tee(original_stdout, logfile)

    print('Log 기록을 위한 Memo')
    print(CFG)
    set_seed(CFG)
    
    train_transform = T.Compose([
        T.Resize((CFG['Img_Size'], CFG['Img_Size'])),
        T.RandomVerticalFlip(p=0.5),
        T.RandomHorizontalFlip(p=0.5),
        T.ToTensor(),
        T.Normalize(mean=[0.485], std=[0.229])
    ])

    valid_transform = T.Compose([
        T.Resize((CFG['Img_Size'], CFG['Img_Size'])),
        T.ToTensor(),
        T.Normalize(mean=[0.485], std=[0.229])
    ])

    train_csv_path = f"./csv_files/{CFG['Target_Nums']}classes_{CFG['Target_Label']}_Training_Single.csv"
    valid_csv_path = f"./csv_files/{CFG['Target_Nums']}classes_{CFG['Target_Label']}_Training_Single.csv"
    train_datasets = cus_Dataset(csv_path=train_csv_path, transform=train_transform)
    valid_datasets = cus_Dataset(csv_path=valid_csv_path, transform=valid_transform)
    train_dataloader = DataLoader(train_datasets, batch_size=CFG['Batch_Size'], shuffle=True, num_workers=CFG['Num_Workers'])
    valid_dataloader = DataLoader(valid_datasets, batch_size=CFG['Batch_Size'], shuffle=False, num_workers=CFG['Num_Workers'])

    model = ConvNext(n_classes=CFG['Target_Nums'])
    model.to(CFG['Device'])
    
    optimizer = optim.AdamW(params = model.parameters(), lr=CFG['Learning_Rate'])
    scheduler = None
    # scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=CFG['Patience'], threshold_mode='rel', min_lr=1e-9, cooldown=3, verbose=True)
    
    train(CFG, model, optimizer, train_dataloader, valid_dataloader, scheduler)

    sys.stdout = original_stdout
    logfile.close()

####################################################################################################################################################################################

if __name__ == "__main__":

    CFG = {
        'Device' : torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        'Today_Date' : datetime.date.today(),
        'Current_Time' : datetime.datetime.now().strftime('%H%M'),

        'Img_Size' : 384,
        'Epochs' : 50,
        'Radnom_Seed' : 42,
        'Learning_Rate' : 1e-5,
        'Batch_Size' : 64,
        'Patience' : 10,
        'Early_Stop' : 20,
        'Num_Workers' : 4,

        'Model_Name' : 'ConvNext',
        'Target_Nums' : 4,
        'Target_Names' : ['1도', '표재성2도', '심재성2도', '3도'],
        'Target_Label' : 'all'
    }

    main(CFG)