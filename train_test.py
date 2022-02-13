#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 9 13:35:55 2021

@author: lachaji
"""

import numpy as np
import time
import argparse

import torch 
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from data.dataloader import SimulatedPIEDataset

from models.torch_transformer_1d import TransformerClassifier

from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, roc_auc_score
import json

torch.cuda.empty_cache()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def binary_acc(y_pred, y_test):
    y_pred_tag = torch.round(y_pred)

    correct_results_sum = (y_pred_tag == y_test).sum().float()
    acc = correct_results_sum / y_test.shape[0]
    
    return acc


def train(train_loader, valid_loader, model, s_epoch, num_steps_wo_improvement,
 valid_loss, global_step, critirion, optimizer, writer, checkpoint_filepath):

    best_valid_loss = np.inf
    improvement_ratio = 0.005

    l_steps = global_step
    start_time = time.time()

    for epoch in range(s_epoch, epochs):
        nb_batches_train = len(train_loader)
        train_acc = 0
        model.train()
        losses = 0.0

        for step, (x, y) in enumerate(train_loader):
            l_steps +=1
            x = x.to(device)
            y = y.reshape(-1,1).to(device)
            
            out = model(x)  # ①

            loss = critirion(out, y)  # ②
            
            model.zero_grad()  # ③

            loss.backward()  # ④
            losses += loss.item()

            optimizer.step()  # ⑤
                        
            train_acc += binary_acc(y, torch.round(out))

                
            if(step % 500 == 0):
                print(f"[Step {(step + 1):04d}]  | Train_Loss {(losses / (step+1)):0.4f}")
            
                writer.add_scalar('training loss',
                    losses / (step+1),
                    l_steps)


        writer.add_scalar('training Acc',
            train_acc / nb_batches_train,
            epoch + 1)
            
        print(f"Epoch: {epoch:02d}\t |Train_Loss: {(losses / nb_batches_train):0.4f}\t |Train_Acc: {(train_acc / nb_batches_train):0.4f}\t\tTime: {(time.time() - start_time)/(60):0.2f} minutes")
        valid_loss, val_acc = evaluate(valid_loader, model, critirion)
        writer.add_scalar('validation loss',
                          valid_loss,
                          epoch + 1)
        writer.add_scalar('validation Acc',
                          val_acc,
                          epoch + 1)
        
        if (best_valid_loss - valid_loss) > np.abs(best_valid_loss * improvement_ratio):
            num_steps_wo_improvement = 0
        else:
            num_steps_wo_improvement += 1
            torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'LOSS': losses / nb_batches_train,
            'best_validation_loss': best_valid_loss,
            'num_steps_wo_improvement': num_steps_wo_improvement,
            'e_stop': False,
            'global_step':l_steps,
            }, checkpoint_filepath)   

        if num_steps_wo_improvement == 7:
            print("Early stopping on epoch:{}".format(str(epoch)))
            torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'LOSS': losses / nb_batches_train,
            'best_validation_loss': best_valid_loss,
            'num_steps_wo_improvement': num_steps_wo_improvement,
            'e_stop': True,
            'global_step':l_steps,
            }, checkpoint_filepath) 
            break;

        if valid_loss <= best_valid_loss:
            best_valid_loss = valid_loss  

            torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'LOSS': losses / nb_batches_train,
            'best_validation_loss': best_valid_loss,
            'num_steps_wo_improvement': num_steps_wo_improvement,
            'e_stop': False,
            'global_step':l_steps,
            }, checkpoint_filepath)

def evaluate(data_loader, model, critirion):
    nb_batches = len(data_loader)
    val_losses = 0.0
    with torch.no_grad():
        model.eval()
        acc = 0 
        for x, y in data_loader:
            x = x.to(device)
            y = y.reshape(-1,1).to(device)
            
            out = model(x)   # ①
            val_loss = critirion(out, y)
            val_losses += val_loss.item()
            
            acc += binary_acc(y, torch.round(out))

    print(f"\t\t   Validation_Loss {(val_losses / nb_batches):0.4f} \t| Val_Acc {(acc / nb_batches):0.4f} \n")
    return val_losses / nb_batches, acc / nb_batches
    
    
def test(data_loader, model):
    with torch.no_grad():
        model.eval()
        step = 0
        for x, y in data_loader:
            x = x.to(device)
            y = y.reshape(-1,1).to(device)
            
            out = model(x)   # ①   
            if(step == 0):
                pred = out
                labels = y

            else:
                pred = torch.cat((pred, out), 0)
                labels = torch.cat((labels, y), 0)
            step +=1

    return pred, labels




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default=False)
    parser.add_argument("--path_annotation", default='data/CP2A_bb_data')
    parser.add_argument("--train_test", default=0)
    args = parser.parse_args()
        
    Train_SimData = SimulatedPIEDataset(args.path_annotation, 'train')
    Val_SimData = SimulatedPIEDataset(args.path_annotation, 'val')
    Test_SimData = SimulatedPIEDataset(args.path_annotation, 'test')
    
    
    train_loader = DataLoader(
        Train_SimData,
        batch_size=32,
        shuffle = True,
        num_workers=4
    )

    val_loader = DataLoader(
        Val_SimData,
        batch_size=32,
        shuffle = True,
        num_workers=4
    )

    test_loader = DataLoader(
        Test_SimData,
        batch_size=32,
        shuffle = False,
        num_workers=4
    )
    
    for x, y in test_loader:
        print(x.shape)
        print(y.shape)
        break;
        
     
        
    epochs = 2000
    
           
    critirion = nn.BCELoss()
    
    if(args.model_name):
        #Either Continue Training or Testing
        
        checkpoint_filepath = "checkpoints/{}.pt".format(args.model_name) 
        
        print(checkpoint_filepath)
            
            
        input_opts = {'num_layers' : 8,
            'd_model': 128,
            'd_input':4,
            'num_heads' : 8,
            'dff': 128,
            'pos_encoding': 16,
            'batch_size': 32,
            'warmup_steps': False,
            'model_name' : time.strftime("%d%b%Y-%Hh%Mm%Ss"),
            'pooling' : False,
            'optimizer': 'Adam'
        }
        
        model = TransformerClassifier(num_layers= input_opts['num_layers'], d_model=input_opts['d_model'],
                                    d_input=input_opts['d_input'], num_heads=input_opts['num_heads'], 
                                    dff=input_opts['dff'], maximum_position_encoding= input_opts['pos_encoding'])
        model.to(device)


        
        critirion = nn.BCELoss()
        
        checkpoint = torch.load(checkpoint_filepath, map_location=device)

            
        if(int(args.train_test) == 0):
            #Continue Traning
            
            model.load_state_dict(checkpoint['model_state_dict_to_continue'])
            model.to(device)            

            s_epoch = checkpoint['epoch']
            num_steps_wo_improvement = checkpoint['num_steps_wo_improvement']
            in_model_state_dict = checkpoint['model_state_dict']

            if(checkpoint['e_stop']):
                print("Early Stopping on epoch {}\n".format(s_epoch))
            else:
                print("Continue Training Loop from epoch {}\n".format(s_epoch))
                optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)    
                
                
                
                writer = SummaryWriter('logs/{}'.format(args.model_name))
                
                train(train_loader, val_loader, model, s_epoch + 1, num_steps_wo_improvement,
                 critirion, optimizer, writer, in_model_state_dict, checkpoint_filepath)
            

        if(int(args.train_test) == 1):
            #Testing

            model.load_state_dict(checkpoint['model_state_dict'])
            model.to(device)    

            print("Testing Loop \n")
            pred, lab = test(test_loader, model)
            pred_cpu = torch.Tensor.cpu(pred)
            lab_cpu = torch.Tensor.cpu(lab)
            acc = accuracy_score(lab_cpu, np.round(pred_cpu))
            conf_matrix = confusion_matrix(lab_cpu, np.round(pred_cpu), normalize = 'true')
            f1 = f1_score(lab_cpu, np.round(pred_cpu))
            auc = roc_auc_score(lab_cpu, np.round(pred_cpu))

            input_opts['acc'] = acc
            input_opts['f1'] = f1
            input_opts['conf_matrix'] = str(conf_matrix)
            input_opts['auc'] = auc
            config = json.dumps(input_opts)


            f = open("paper_torch_checkpoints/{}.json".format(args.model_name),"w")
            f.write(config)
            f.close()

            print(f"Accuracy: {acc} \n f1: {f1} \n AUC: {auc} ")        
            
            
    else:
        if(int(args.train_test)== 0):
        #Train from Scratch
            input_opts = {'num_layers' : 8,
                'd_model': 128,
                'd_input':4,
                'num_heads' : 8,
                'dff': 128,
                'pos_encoding': 16,
                'batch_size': 32,
                'warmup_steps': False,
                'model_name' : time.strftime("%d%b%Y-%Hh%Mm%Ss"),
                'pooling' : False,
                'optimizer': 'Adam'
            }
            
            model = TransformerClassifier(num_layers= input_opts['num_layers'], d_model=input_opts['d_model'],
                                        d_input=input_opts['d_input'], num_heads=input_opts['num_heads'], 
                                        dff=input_opts['dff'], maximum_position_encoding= input_opts['pos_encoding'])
            model.to(device)

            optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)


            model_name = time.strftime("%d%b%Y-%Hh%Mm%Ss")
            model_folder_name = "Sim_Encoder_Only_" + input_opts['model_name'] 
            checkpoint_filepath = "paper_torch_checkpoints/{}_model.pt".format(model_folder_name)
            writer = SummaryWriter('torch_logs/{}'.format(model_folder_name))
            
            in_model_state_dict = model.state_dict()
            print("Start Training Loop from Scratch \n")

            
            train(train_loader, test_loader, model, 0, 0, np.inf, 0, critirion,
             optimizer, writer, checkpoint_filepath)
        
        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
