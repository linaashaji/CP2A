from torch.utils.data import Dataset, DataLoader
import pickle
from sklearn.preprocessing import OneHotEncoder
from data.preprocessing import pad_sequence
import numpy as np
import pandas as pd
import torch

class SimulatedPIEDataset(Dataset):
    
    def __init__(self,
                 path, split, data_type=1):


        self.data_type = data_type
            
        if(data_type == 2):
            self.path_data_enc = path + '/' + split + '_enc.pkl'
            self.path_data_dec = path + '/' + split + '_dec.pkl'
            self.data = pickle.load(open(self.path_data_enc,'rb'))
            self.data_dec = pickle.load(open(self.path_data_dec,'rb'))
        else:
            self.path_data = path + '/' + split + '_data.pkl'
            self.data = pickle.load(open(self.path_data,'rb'))

        self.path_labels= path + '/' + split + '_labels.pkl'
        self.labels = pickle.load(open(self.path_labels,'rb'))

            
        for i, df in enumerate(self.data):
            df["bbox_right_x"] = df["bbox_right_x"] / 1920
            df["bbox_left_x"] = df["bbox_left_x"] / 1920
            df["bbox_up_y"] = df["bbox_up_y"] / 1080
            df["bbox_down_y"] = df["bbox_down_y"] / 1080
            

            self.data[i] = df[['bbox_left_x','bbox_up_y', 'bbox_right_x', 'bbox_down_y']]

        if (data_type == 2):
            for i, df in enumerate(self.data_dec):
                df["bbox_right_x"] = df["bbox_right_x"] / 1920
                df["bbox_left_x"] = df["bbox_left_x"] / 1920
                df["bbox_up_y"] = df["bbox_up_y"] / 1080
                df["bbox_down_y"] = df["bbox_down_y"] / 1080
                

                self.data_dec[i] = df[['bbox_left_x','bbox_up_y', 'bbox_right_x', 'bbox_down_y']]
            
    def __getitem__(self, index):
        return self.getitem(index)
    
    def __len__(self):
        return len(self.data)
    
    def getitem(self, index):
        
        df = self.data[index]
        labels = self.labels[index]
        
        intent = torch.tensor(labels, dtype=torch.float32)
        
        sample = torch.tensor(df.to_numpy(), dtype=torch.float32)

        if self.data_type == 2:
            df_dec = self.data_dec[index]
            sample_dec = torch.tensor(pad_sequence([df_dec.values.tolist()[0:60]], 60), dtype=torch.float32).reshape(60,4)
         
        
        if self.data_type == 1:
            return sample, intent
        else:
            return sample, sample_dec, intent





        
