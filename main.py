####################################################################################################
# DGL: Device Generic Latency model for Neural Architecture Search
####################################################################################################
import os
import torch
from parser import get_parser  
from DGL import DGL

def main(args):
    set_seed(args)
    args = set_gpu(args)
    
    print(f'==> mode is [{args.mode}] ...')
    model = DGL(args)

    if args.mode == 'train':
        model.train()
    elif args.mode == 'test':
        model.test()
        
    elif args.mode == 'nas'and args.search_space=='fbnetv3':
        model.fbnetv3nas()   
        
    elif args.mode == 'nas'and args.search_space=='nasbench201':
        model.nasbench201nas()   

        
def set_seed(args):
    # Set the random seed for reproducible experiments
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

def set_gpu(args):
    os.environ['CUDA_VISIBLE_DEVICES']= '-1' if args.gpu == None else args.gpu
    args.gpu = int(args.gpu)
    return args 

 

if __name__ == '__main__':
    main(get_parser())



