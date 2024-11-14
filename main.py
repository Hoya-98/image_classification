import os
import datetime
import sys

import pandas as pd

import torch
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
import torchvision.transforms as T

from sklearn.model_selection import train_test_split

from utils.cus_dataset import cus_Dataset
from utils.options import Tee
from utils.SwinV2 import VanillaSwinV2
from utils.ConvNeXt import ConvNeXt
from train import set_seed, train
from test import test

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

########################################################################################################################################################

def main(CFG):
    
    log_path = f"./log/{CFG['Today_Date']}_{CFG['Current_Time']}.txt"    
    logfile = open(log_path, 'a')
    original_stdout = sys.stdout
    sys.stdout = Tee(original_stdout, logfile)
    print('Log 기록을 위한 Memo')
    print(CFG)
    
    set_seed(CFG)
    
    train_transform = T.Compose([
        T.Resize((CFG['Resize'], CFG['Resize'])),
        T.RandomVerticalFlip(p=0.5),
        T.RandomHorizontalFlip(p=0.5),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std =[0.229, 0.224, 0.225])
    ])
    valid_transform = T.Compose([
        T.Resize((CFG['Resize'], CFG['Resize'])),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std =[0.229, 0.224, 0.225])
    ])

    data = pd.read_csv(CFG['Meta_Path'])
    train_data, tmp_data = train_test_split(data, test_size=0.2, random_state=CFG['Random_Seed'])
    valid_data, test_data = train_test_split(tmp_data, test_size=0.5, random_state=CFG['Random_Seed'])
    train_dataset = cus_Dataset(CFG, train_data, train_transform)
    valid_dataset = cus_Dataset(CFG, valid_data, valid_transform)
    test_dataset = cus_Dataset(CFG, test_data, valid_transform)
    train_dataloader = DataLoader(train_dataset, batch_size=CFG['Batch_Size'], shuffle=True, num_workers=CFG['Num_Workers'])
    valid_dataloader = DataLoader(valid_dataset, batch_size=CFG['Batch_Size'], shuffle=False, num_workers=CFG['Num_Workers'])
    test_dataloader = DataLoader(test_dataset, batch_size=CFG['Batch_Size'], shuffle=False, num_workers=CFG['Num_Workers'])
        
    model = ConvNeXt(n_classes=CFG['Target_Nums'], img_size=CFG['Resize'], size=CFG['Model_Size'])
    optimizer = optim.AdamW(params=model.parameters(), lr=CFG['Learning_Rate'], weight_decay=0.001)
    scheduler = None
    model.to(CFG['Device'])
    train(CFG, model, optimizer, scheduler, train_dataloader, valid_dataloader)
    
    weight = f"./model/{CFG['Today_Date']}_{CFG['Current_Time']}.pth"
    model.load_state_dict(torch.load(weight, weights_only=True))
    model.to(CFG['Device'])
    test(CFG, model, test_dataloader)

    sys.stdout = original_stdout
    logfile.close()


########################################################################################################################################################

if __name__ == "__main__":
    
    CFG = {
    'Device' : torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
    'Today_Date' : datetime.date.today(),
    'Current_Time': datetime.datetime.now().strftime("%H%M"),
    'Image_Dir' :  '',
    'Meta_Path' : '',

    'Epochs': 50,
    'Random_Seed' : 42,
    'Learning_Rate' : 1e-5,
    'Resize' : 512,
    'Batch_Size' : 32,
    'Early_Stop' : 30,
    'Num_Workers' : 8,

    'Model_Name' : 'ConvNeXt',
    'Model_Size' : 'Large',
    'Target_Nums': 4,
    'Target_Names' : ['Class0', 'Class1', 'Class2', 'Class3'],
    }

    main(CFG)