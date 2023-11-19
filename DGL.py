####################################################################################################
# DGL: Device Generic Latency model for Neural Architecture Search
####################################################################################################
import os
import logging
from collections import OrderedDict
from collections import defaultdict
import csv
from tqdm import tqdm
import json
#import wandb

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
import numpy as np
import pandas as pd
from math import sqrt
import random


from net import MLP
from net import FBNetV3
from net import TinyNetwork
from net import Acctrain
from net import Latencytrain
from loader import Data
from utils import *

class DGL:
    def __init__(self, args):
        self.args = args
        self.mode = args.mode
        self.search_space = args.search_space
        self.load_path = args.load_path
        # Log
        self.save_path = args.save_path
        # Data  
        self.train_devices = args.train_devices
        self.test_devices = args.test_devices
        self.lr = args.lr
        self.epoch = args.epoch
        # NAS
        self.nas_target_device = args.nas_target_device
        self.latency_constraint = args.latency_constraint
        self.POPULATION_SIZE = args.POPULATION_SIZE
        self.EPOCH = args.EPOCH
        self.CROSSOVER_RATE = args.CROSSOVER_RATE
        self.GENE_LENGTH = args.GENE_LENGTH
        self.MUTATION_RATE = args.MUTATION_RATE
        
        # Data
        self.data = Data(args.mode,
                        args.data_path, 
                        args.search_space,
                        args.train_devices,                      
                        args.test_devices,
                        )
        
    def train(self):
        print('==> start training...')
        archs=self.data.load_archs()
        A=self.data.load_train_data()
        xtrain,xtest,ytrain,ytest=train_test_split(archs,A,test_size=0.2,random_state=None)
        xtrain = torch.torch.tensor(xtrain , dtype=torch.float);
        ytrain = torch.tensor(ytrain , dtype=torch.float);
        xtrain=xtrain.cuda()
        ytrain=ytrain.cuda()
        xtest = torch.torch.tensor(xtest , dtype=torch.float);
        ytest = torch.tensor(ytest , dtype=torch.float);
        xtest=xtest.cuda()
        ytest=ytest.cuda()
        self.net=MLP(self.search_space)
        self.net.cuda()
        self.net.fc1.weight.requires_grad = False
        self.net.fc1.bias.requires_grad = False
        self.net.fc2.weight.requires_grad = False
        self.net.fc2.bias.requires_grad = False
        self.net.fc3.weight.requires_grad = False
        self.net.fc31.bias.requires_grad = False
        self.net.fc31.weight.requires_grad = False
        self.net.fc3.bias.requires_grad = False
        self.net.fc4.weight.requires_grad = False
        self.net.fc4.bias.requires_grad = False
        losses = []
        optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, self.net.parameters()), lr=0.1)
        loss_func = torch.nn.SmoothL1Loss(reduction='mean')
        num_epoch =self.epoch
        for epoch in range(num_epoch):
            yp = self.net(xtrain)  # feed data and forward propagation
            t=len(ytrain)
            ytrain=ytrain.reshape((t,1))
            loss = loss_func(yp,ytrain) # Calculate the loss
            losses.append(loss.item())
            if torch.isnan(loss):
                break
            optimizer.zero_grad()   
            loss.backward()   
            optimizer.step()   
                #scheduler.step()
            if epoch % 1000 == 0:
                y_p= self.net(xtest)
                print('epoch: {}, loss: {}'.format(epoch, loss.data.item()))
                y_p=y_p.cpu()
                ytest=ytest.cpu()
                ytset=ytest.detach().numpy()
                y_p=y_p.detach().numpy()
                print("MAE = ", metrics.mean_absolute_error(ytest, y_p)) 
                print("MSE = ", metrics.mean_squared_error(ytest, y_p)) 
        print('==> end training')
        torch.save(self.net, 'result/'+str(self.search_space)+"/pretrainmodel.pth")
        print('==> save model')
          
        
    def test(self):
        net=torch.load('result/'+str(self.search_space)+'/pretrainmodel.pth')
        print('==> load model')
        archs=self.data.load_archs()
        archs = torch.torch.tensor(archs , dtype=torch.float);
        archs=archs.cuda()
        net.cuda()
        Apre=net(archs)
        Apre=Apre.cpu()
        Apre=Apre.detach().numpy()
        for device in self.test_devices:
            latency=self.data.load_test_data(device)
            hardware=self.data.load_hardware_data(device)
            latencypre=DGL(Apre,hardware)
            m=len(latency)
            latency=latency.reshape(1,m)
            latencypre=latencypre.reshape(1,m)
            print("device = ",device)
            print("R = ", np.corrcoef(latency,latencypre)[1,0])
            print("MSE = ", metrics.mean_squared_error(latency,latencypre)) 
            print("RMSE = ", np.sqrt(metrics.mean_squared_error(latency,latencypre))) 
            print("MAE = ", metrics.mean_absolute_error(latency,latencypre)) 
            np.savetxt('result/'+str(self.search_space)+'/'+str(device)+"_timepre.txt",latencypre,fmt='%d',delimiter=' ')
    
    def fbnetv3nas(self):
        data=self.data.load_nas_data()
        X = data[:,:28]
        latencynet=Latencytrain() 
        latencynet.load_state_dict(torch.load( "./data/fbnetv3/NAS/latency_predictor.pth"))
        accuracynet=Acctrain() 
        accuracynet.load_state_dict(torch.load( "./data/fbnetv3/NAS/accuracy_predictor.pth"))
        print('==> load model')
        print('==> init population')
        tar_hardware_par=self.data.load_tar_par(self.nas_target_device)
        parents, fitness = init_population(data,X,accuracynet,latencynet,tar_hardware_par,self.latency_constraint,self.POPULATION_SIZE)
        bestpopulation= parents
        bestfitness=fitness
        print('==> start search')
        for i in range(self.EPOCH):
            population = select(parents,fitness,self.POPULATION_SIZE)
            children = crossover(population,self.POPULATION_SIZE,self.CROSSOVER_RATE,self.GENE_LENGTH)
            children = mutation(children,self.MUTATION_RATE,self.GENE_LENGTH)
            parents, fitness=caculate_fitness(children,X,accuracynet,latencynet,tar_hardware_par,self.latency_constraint,self.POPULATION_SIZE)
            bestpopulation,bestfitness=save_fitness(parents, fitness,bestpopulation,bestfitness,self.POPULATION_SIZE)
            print("epoch:",i, " fitness: ", fitness)
        print("bestfitness: ",bestfitness)
        print('==> end search')
        print('==> predicted accuracy:',bestfitness[0]+70)
        if bestfitness[0]!=0: 
            Generate_pt_model(bestpopulation[0],FBNetV3,self.nas_target_device,self.latency_constraint)
        if bestfitness[0]==0: 
            print("==> No network meets latency constraints")
        
        
    def nasbench201nas(self):
        net=torch.load('result/'+str(self.search_space)+'/pretrainmodel.pth')
        print('==> load model')
        archs=self.data.load_nas_arch()
        archs = torch.torch.tensor(archs , dtype=torch.float);
        archs=archs.cuda()
        net.cuda()
        Apre=net(archs)
        Apre=Apre.cpu()
        Apre=Apre.detach().numpy()
        tar_hardware_par=self.data.load_tar_par(self.nas_target_device)
        latencypre=DGL(Apre,tar_hardware_par)
        accpre=self.data.load_acc()
        time=np.zeros((1,2))
        acc,lat,index=findnas(latencypre,accpre,self.latency_constraint)
        print("==> find net")
        print("==> accuracy:" ,acc)
        print("==> predicted latency:" ,lat)
        print("==> save model")
        if acc!=0:
            save_pt_model(archs[index],TinyNetwork,self.nas_target_device,self.latency_constraint)
        if acc==0:
            print("==> No network meets latency constraints")


 

