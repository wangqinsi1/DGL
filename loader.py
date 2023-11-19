####################################################################################################
# DGL: Device Generic Latency model for Neural Architecture Search
####################################################################################################

import os
import random
import numpy as np
import torch
from utils import *

class Data:
    def __init__(self, mode,
                    data_path, 
                    search_space,
                    train_devices,
                    test_devices,
                    ):
        
        self.mode = mode
        self.data_path = data_path
        self.search_space = search_space
        self.train_devices = train_devices
        self.test_devices = test_devices
         

    def load_archs(self):   
            archs = np.loadtxt(os.path.join(self.data_path,'net.txt'), dtype=np.float32, delimiter=' ') 
            archs=mean(archs)
            return archs
         
    def load_train_data(self):   
            A=np.loadtxt(os.path.join(self.data_path,'a.txt'), dtype=np.float32, delimiter=' ')
            A=normalization(A)
            return A
        
    def load_test_data(self,device):   
            latency= np.loadtxt(os.path.join(self.data_path,'latency',f'{device}.txt'),dtype=np.float32, delimiter=' ')
            latency = normalization(latency)
            return latency

    def load_hardware_data(self,device):   
            hardware= np.loadtxt(os.path.join(self.data_path,'hardware',f'{device}.txt'),dtype=np.float32, delimiter=' ')
            return hardware
        
    def load_nas_data(self):   
            data = np.loadtxt('./data/fbnetv3/NAS/dataxy.txt', dtype=np.float32, delimiter=' ')
            return data
        
    def load_tar_par(self,device):   
            par = np.loadtxt(os.path.join(self.data_path,'NAS',f'{device}.txt'), dtype=np.float32, delimiter=' ')
            return par
        
    def load_nas_arch(self):   
            acc = np.loadtxt('./data/nasbench201/NAS/net.txt', dtype=np.float32, delimiter=' ')
            return acc

    

    def load_acc(self):   
            acc = np.loadtxt('./data/nasbench201/NAS/acc.txt', dtype=np.float32, delimiter=' ')
            return acc
