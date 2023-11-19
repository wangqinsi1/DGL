####################################################################################################
# DGL: Device Generic Latency model for Neural Architecture Search
####################################################################################################

import os
import random
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
import numpy as np
import pandas as pd
from math import sqrt
import random
import torchvision
from torch.utils.mobile_optimizer import optimize_for_mobile


def normalization(latency, index=None):
    if index != None:
        min_val = min(latency[index])
        max_val = max(latency[index])
    else :
        min_val = min(latency)
        max_val = max(latency)
    latency=latency/100
    return latency

def mean(arch):
    t=arch[0]
    for i in range(len(t)):
        arch[:,i] -= np.mean(arch[:,i],axis = 0)
    return arch

def DGL(A,hardware):
    latency=(A/hardware[1]+1/(hardware[2]**0.5)*(A/hardware[3]))*1/hardware[0]
    return latency

#NAS# 


def getacc(gene,accuracynet):
    gene1 = torch.tensor(gene , dtype=torch.float)
    acc=accuracynet(gene1)
    acc=acc.detach().numpy()
    acctopk1=acc[0,0]-70
    acctopk5=acc[0,1]-90
    return acctopk1

 
def getlatency(gene,latencynet,tar_hardware_par):
    gene1 = torch.tensor(gene , dtype=torch.float);
    a=latencynet(gene1)
    a=a.detach().numpy()
    a=a*100
    a1=a[0,0]
    frequency=tar_hardware_par[0]
    dispatch=tar_hardware_par[1]
    cache=tar_hardware_par[2]
    memoryspeed=tar_hardware_par[3]
    latency=(a1/dispatch+1/(cache**0.5)*a1/memoryspeed)*1/frequency+15
    return latency

 

def getfitness(gene,X,accuracynet,latencynet,tar_hardware_par,LATENCYNEED):
    sc = StandardScaler()
    X1 = sc.fit_transform(X)
    gene=gene.reshape((1,28))
    gene = sc.transform(gene)  
    Acc=getacc(gene,accuracynet)
    Lat=getlatency(gene,latencynet,tar_hardware_par)
    
    if Lat<= LATENCYNEED:
        fitness=Acc
    if Lat> LATENCYNEED:
        #fitness=Acc-alpha^beta
        fitness=0
     
    
    if fitness<0:
        fitness=0
    return fitness

def init_population(data,X,accuracynet,latencynet,tar_hardware_par,LATENCYNEED,POPULATION_SIZE):
    pool = data[:100,:28]
    value = np.zeros((100,29))
    fit = np.zeros((1,100))
    for i in range(100):
        gene=pool[i]
        fit[0,i] =getfitness(gene,X,accuracynet,latencynet,tar_hardware_par,LATENCYNEED)
        value[i,:28]=pool[i]
        value[i,28]=fit[0,i]
    
    fit=np.sort(-fit) 
    fit=-fit
    prerank=value[np.lexsort(-value.T)]
    population=prerank[:POPULATION_SIZE,:28]
    fitness =fit[0,:POPULATION_SIZE]
    print('Inited population fitness：', fitness)   
    return population, fitness


 
def select(population, fitness,POPULATION_SIZE):   
    idx = np.random.choice(np.arange(POPULATION_SIZE), size=POPULATION_SIZE, replace=True,
                           p=(fitness) / (fitness.sum()))
    return population[idx]



 
def cross(idx,population,POPULATION_SIZE,GENE_LENGTH):
    father = population[idx]
    child = father
    if idx==0:
        m=np.random.randint(1,POPULATION_SIZE)
    else:
        m=np.random.randint(0,POPULATION_SIZE)
    
    if m==idx:
        m=idx-1
    
    mother = population[m] 
    cross_points = np.random.randint(GENE_LENGTH)  # 随机产生交叉的点
    child[cross_points:] = mother[cross_points:]
    return child


def crossover(population,POPULATION_SIZE,CROSSOVER_RATE,GENE_LENGTH):
    new_population = []
    for i in range(POPULATION_SIZE):   
        child = population[i]
        if np.random.rand() < CROSSOVER_RATE:   
             child=cross(i,population,POPULATION_SIZE,GENE_LENGTH)
        new_population.append(child)
    return new_population
 
    
    
def mutation(population,MUTATION_RATE,GENE_LENGTH):
    k=[3,5]
    e1=[4,5,6,7]
    e2=[2,3,4,5]
    c1=[16,18,20,22,24]
    c2=[16,18,20,22,24]
    c3=[20,24,28,32]
    c4=[24,28,32,36,40,44,48]
    c5=[56,60,64,68,72,76,80,84]
    c6=[96,100,104,108,112,116,120,124,128,132,136,140,144]
    c7=[180,184,188,192,196,200,204,208,212,216,220,224]
    c8=[180,184,188,192,196,200,204,208,212,216,220,224]
    n2=[1,2,3,4]
    n3=[4,5,6,7]
    n4=[4,5,6,7]
    n5=[4,5,6,7,8]
    n6=[6,7,8,9,10]
    n7=[5,6,7,8,9]
    new_population = []
    for individual in population:
        if random.random() < MUTATION_RATE:
            rand = random.randint(0,GENE_LENGTH)
            if rand<7:
                individual[rand]=np.array(random.sample(k,1))
            if rand==7 or rand ==9 or rand ==11 or rand ==13:
                individual[rand]=np.array(random.sample(e1,1))   
            if rand==8 or rand ==10 or rand ==12:
                individual[rand]=np.array(random.sample(e2,1))  
            if rand==14:
                individual[rand]=np.array(random.sample(c1,1))  
            if rand==15:
                individual[rand]=np.array(random.sample(c2,1))
            if rand==16:
                individual[rand]=np.array(random.sample(c3,1))  
            if rand==17:
                individual[rand]=np.array(random.sample(c4,1))
            if rand==18:
                individual[rand]=np.array(random.sample(c5,1))  
            if rand==19:
                individual[rand]=np.array(random.sample(c6,1))
            if rand==20:
                individual[rand]=np.array(random.sample(c7,1))  
            if rand==21:
                individual[rand]=np.array(random.sample(c8,1))
            if rand==22:
                individual[rand]=np.array(random.sample(n2,1))
            if rand==23:
                individual[rand]=np.array(random.sample(n3,1))  
            if rand==24:
                individual[rand]=np.array(random.sample(n4,1))
            if rand==25:
                individual[rand]=np.array(random.sample(n5,1))  
            if rand==26:
                individual[rand]=np.array(random.sample(n6,1))
            if rand==27:
                individual[rand]=np.array(random.sample(n7,1)) 
    
        new_population.append(individual)
    return new_population

 
def caculate_fitness(children,X,accuracynet,latencynet,tar_hardware_par,LATENCYNEED,POPULATION_SIZE):
    childrenfitness=np.zeros((POPULATION_SIZE,1))
    
    for i in range(POPULATION_SIZE):
        individual=children[i]
        childrenfitness[i,0] =getfitness(individual,X,accuracynet,latencynet,tar_hardware_par,LATENCYNEED)
        
    childrenfit=np.hstack((children,childrenfitness))   
    value=childrenfit
    valuesort=value[np.lexsort(-value.T)]
    population=valuesort[:,:28]
    fitnessnew=valuesort[:,28]
    return population,fitnessnew



def save_fitness(population,fitness,bestpopulation,bestfitness,POPULATION_SIZE):
    fitness = fitness.reshape((POPULATION_SIZE,1))
    bestfitness = bestfitness.reshape((POPULATION_SIZE,1))
    best = np.hstack((bestpopulation,bestfitness))
    now = np.hstack((population,fitness))
    for i in range(POPULATION_SIZE):
        for j in range(POPULATION_SIZE):
            if (now[i]==best[j]).all():
                now[i,28]=0
     
    value=np.vstack((best,now))
    valuesort=value[np.lexsort(-value.T)]
    population=valuesort[:POPULATION_SIZE,:28]
    fitnessnew=valuesort[:POPULATION_SIZE,28]
    return population,fitnessnew



def Generate_pt_model(data,FBNetV3,device,const):
    pool=data
    cfgpool=np.zeros((7,5))
    print('==> start save models')
    pool=pool.reshape((1,28))
    i=0
    c=int(pool[i,14])
    cfgpool[0,0]=pool[i,15]
    cfgpool[1,0]=pool[i,16]
    cfgpool[2,0]=pool[i,17]
    cfgpool[3,0]=pool[i,18]
    cfgpool[4,0]=pool[i,19]
    cfgpool[5,0]=pool[i,20]
    cfgpool[6,0]=pool[i,21]
    cfgpool[0,1]=pool[i,0]
    cfgpool[1,1]=pool[i,1]
    cfgpool[2,1]=pool[i,2]
    cfgpool[3,1]=pool[i,3]
    cfgpool[4,1]=pool[i,4]
    cfgpool[5,1]=pool[i,5]
    cfgpool[6,1]=pool[i,6]
    cfgpool[0,2]=1
    cfgpool[0,3]=1
    cfgpool[1,2]=pool[i,7]
    cfgpool[1,3]=pool[i,8]
    cfgpool[2,2]=pool[i,9]
    cfgpool[2,3]=pool[i,10]
    cfgpool[3,2]=pool[i,11]
    cfgpool[3,3]=pool[i,12]
    cfgpool[4,2]=pool[i,11]
    cfgpool[4,3]=pool[i,12]
    cfgpool[5,2]=pool[i,13]
    cfgpool[5,3]=pool[i,13]
    cfgpool[6,2]=6
    cfgpool[6,3]=6
    cfgpool[0,4]=pool[i,22]
    cfgpool[1,4]=pool[i,23]
    cfgpool[2,4]=pool[i,24]
    cfgpool[3,4]=pool[i,25]
    cfgpool[4,4]=pool[i,26]
    cfgpool[5,4]=pool[i,27]
    cfgpool[6,4]=1
    cfgpool=cfgpool.astype(int)
    model= FBNetV3(c,cfgpool)
    model.eval()
    example = torch.rand(1, 3, 224, 224)
    traced_script_module = torch.jit.trace(model, example)
    optimized_traced_model = optimize_for_mobile(traced_script_module)
    optimized_traced_model._save_for_lite_interpreter('./result/fbnetv3/nas/' + str(device)+"_"+str(const)+"_model.pt")
        
    print('==> end save models')

def findnas(lat,acc,const):
    lat=lat.reshape((len(lat),1))
    acc=acc.reshape((len(acc),1))
    T= np.hstack((lat,acc))
    m=0
    w=0
    l=0
    k=0
    accbest=0
    latbest=0
    star=np.zeros((1,2))
    for i in range(len(lat)):
        l=T[i,0]
        if l<=const:
            w=T[i,1]
            if w>m:
                m=w
                accbest=T[i,1]
                latbest=T[i,0]
                k=i
                
    return accbest,latbest,k

def save_pt_model(archs,TinyNetwork,device,const):
    m0=[]
    m1=[]
    m2=[]
    m=[]
    t=archs
    if t[0]==1:
        m0.append(['none',0])
    if t[0]==2:
        m0.append(['skip_connect',0])
    if t[0]==3:
        m0.append(['nor_conv_1x1',0])
    if t[0]==4:
        m0.append(['nor_conv_3x3',0])
    if t[0]==5:
        m0.append(['avg_pool_3x3',0])
    if t[1]==1:
        m1.append(['none',0])
    if t[1]==2:
        m1.append(['skip_connect',0])
    if t[1]==3:
        m1.append(['nor_conv_1x1',0])
    if t[1]==4:
        m1.append(['nor_conv_3x3',0])
    if t[1]==5:
        m1.append(['avg_pool_3x3',0])
    if t[2]==1:
        m1.append(['none',1])
    if t[2]==2:
        m1.append(['skip_connect',1])
    if t[2]==3:
        m1.append(['nor_conv_1x1',1])
    if t[2]==4:
        m1.append(['nor_conv_3x3',1])
    if t[2]==5:
        m1.append(['avg_pool_3x3',1])
    if t[3]==1:
        m2.append(['none',0])
    if t[3]==2:
        m2.append(['skip_connect',0])
    if t[3]==3:
        m2.append(['nor_conv_1x1',0])
    if t[3]==4:
        m2.append(['nor_conv_3x3',0])
    if t[3]==5:
        m2.append(['avg_pool_3x3',0])
    if t[4]==1:
        m2.append(['none',1])
    if t[4]==2:
        m2.append(['skip_connect',1])
    if t[4]==3:
        m2.append(['nor_conv_1x1',1])
    if t[4]==4:
        m2.append(['nor_conv_3x3',1])
    if t[4]==5:
        m2.append(['avg_pool_3x3',1])
    if t[5]==1:
        m2.append(['none',2])
    if t[5]==2:
        m2.append(['skip_connect',2])
    if t[5]==3:
        m2.append(['nor_conv_1x1',2])
    if t[5]==4:
        m2.append(['nor_conv_3x3',2])
    if t[5]==5:
        m2.append(['avg_pool_3x3',2])
    m.append(m0)
    m.append(m1)
    m.append(m2) 
    m.append([])
    model= TinyNetwork(16,5,m,100)
    model.eval()
    example = torch.rand(1, 3, 224, 224)
    traced_script_module = torch.jit.trace(model, example)
    optimized_traced_model = optimize_for_mobile(traced_script_module)
    optimized_traced_model._save_for_lite_interpreter('./result/nasbench201/nas/' + str(device)+"_"+str(const)+"_model.pt")

  

